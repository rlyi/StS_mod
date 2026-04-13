"""CombatEnv — Gymnasium-обёртка над живой игрой Slay the Spire.

Мост между callback-моделью spirecomm и синхронным интерфейсом Gymnasium:
  - Фоновый поток запускает Coordinator (читает stdin, пишет в stdout).
  - Основной поток (SB3-обучение) вызывает reset() / step().
  - Общение через две очереди: state_q и action_q.

Схема одного шага:
  Основной поток          Фоновый поток (spirecomm)
  ─────────────           ─────────────────────────
  step(action) ──action_q──► handle_state получает action
                             game обрабатывает ход
                             spirecomm присылает новое состояние
               ◄─state_q── handle_state кладёт ("state", game)
  возвращает obs, reward
"""

import queue
import threading
import time
import logging
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from spirecomm.communication.coordinator import Coordinator
from spirecomm.communication.action import (
    EndTurnAction, ProceedAction, ChooseAction, StateAction, StartGameAction
)
from spirecomm.spire.character import PlayerClass

from config import OBS_SIZE, ACTION_SIZE
from agents.combat_agent import game_to_obs, action_to_spirecomm, get_valid_actions
from environment.reward import calculate_reward

log = logging.getLogger("CombatEnv")


class CombatEnv(gym.Env):
    """Gymnasium env для боевой системы Slay the Spire (Акт 1, Ironclad)."""

    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(OBS_SIZE,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(ACTION_SIZE)

        self._state_q: queue.Queue = queue.Queue()
        self._action_q: queue.Queue = queue.Queue()
        self._coord_thread: threading.Thread | None = None
        self._coordinator: Coordinator | None = None

        self._prev_game = None
        self._done = False

    # ── Gymnasium interface ────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self._coord_thread is None or not self._coord_thread.is_alive():
            self._start_coordinator()

        # Очистить очередь от предыдущего эпизода
        _drain(self._state_q)

        self._done = False
        self._prev_game = None

        game = self._wait_for_combat_state(timeout=300)
        if game is None:
            raise RuntimeError(
                "Тайм-аут ожидания боя. Запустите Slay the Spire с CommunicationMod."
            )

        self._prev_game = game
        return game_to_obs(game), {}

    def step(self, action: int):
        if self._done:
            raise RuntimeError("Эпизод завершён. Вызовите reset().")

        self._action_q.put(int(action))

        try:
            msg = self._state_q.get(timeout=60)
        except queue.Empty:
            # Игра зависла — считаем ход проигрышем
            self._done = True
            return np.zeros(OBS_SIZE, dtype=np.float32), -2.0, True, False, {}

        kind = msg[0]

        if kind == "state":
            game = msg[1]
            reward = calculate_reward(self._prev_game, game)
            self._prev_game = game
            return game_to_obs(game), reward, False, False, {}

        elif kind == "terminal":
            _, outcome, game = msg
            reward = 2.0 if outcome == "win" else -2.0
            self._done = True
            obs = game_to_obs(game) if (game and game.player) else np.zeros(OBS_SIZE, dtype=np.float32)
            return obs, reward, True, False, {"outcome": outcome}

        # Ошибка или неизвестное сообщение
        self._done = True
        return np.zeros(OBS_SIZE, dtype=np.float32), 0.0, True, False, {}

    def close(self):
        pass  # Coordinator работает через stdin/stdout — ОС закроет при выходе

    # ── Внутренние методы ─────────────────────────────────────────────

    def _start_coordinator(self):
        self._coordinator = Coordinator()
        self._coord_thread = threading.Thread(
            target=self._coordinator_loop, daemon=True, name="spirecomm-coordinator"
        )
        self._coord_thread.start()

    def _coordinator_loop(self):
        """Фоновый поток: читает состояния игры, отправляет действия."""
        try:
            self._coordinator.register_state_change_callback(self._handle_state)
            self._coordinator.register_out_of_game_callback(self._handle_out_of_game)
            self._coordinator.register_command_error_callback(self._handle_error)
            self._coordinator.signal_ready()
            self._coordinator.run()
        except Exception as e:
            import traceback
            log.error("Coordinator упал: %s\n%s", e, traceback.format_exc())
            self._state_q.put(("error", str(e), None))

    def _handle_error(self, error):
        log.warning("Ошибка от игры: %s", error)
        return StateAction()

    def _handle_out_of_game(self):
        """Вне боя (главное меню) — автоматически стартуем новый ран."""
        log.info("Вне игры — запускаем новый ран Ironclad")
        return StartGameAction(PlayerClass.IRONCLAD, ascension_level=0)

    def _handle_state(self, game):
        """Callback spirecomm — вызывается в фоновом потоке на каждое обновление."""
        screen = _screen_str(game)
        player = game.player

        # ── Конец игры ────────────────────────────────────────────────
        if screen == "GAME_OVER":
            victory = getattr(getattr(game, "screen", None), "victory", False)
            outcome = "win" if victory else "lose"
            floor = getattr(game, "floor", "?")
            hp = f"{player.current_hp}/{player.max_hp}" if player else "?/?"
            log.info("РАН ЗАВЕРШЁН | исход=%-4s этаж=%-2s HP=%s", outcome, floor, hp)
            self._state_q.put(("terminal", outcome, game))
            return ProceedAction()

        # ── Конец боя (появился экран наград) ─────────────────────────
        if screen == "COMBAT_REWARD":
            floor = getattr(game, "floor", "?")
            hp = f"{player.current_hp}/{player.max_hp}" if player else "?/?"
            log.info("РАН ЗАВЕРШЁН | исход=win  этаж=%-2s HP=%s", floor, hp)
            self._state_q.put(("terminal", "win", game))
            return ProceedAction()

        # ── Не бой — навигируем автоматически ────────────────────────
        if screen != "NONE":
            action = _auto_navigate(screen, game)
            log.debug("Навигация: %-20s → %s", screen, type(action).__name__)
            return action

        # ── В бою: player может быть None на старте ───────────────────
        if player is None:
            return StateAction()

        # ── В бою: ждём действие от основного потока ─────────────────
        self._state_q.put(("state", game))
        try:
            action_int = self._action_q.get(timeout=60)
        except queue.Empty:
            return EndTurnAction()

        # Применяем маску: если PPO выбрал невалидное действие — берём первое допустимое
        mask = get_valid_actions(game)
        if not mask[action_int]:
            valid = [i for i, v in enumerate(mask) if v]
            action_int = valid[0] if valid else 15

        return action_to_spirecomm(action_int, game)

    def _wait_for_combat_state(self, timeout: int = 300):
        """Блокирует до получения первого состояния боя."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                msg = self._state_q.get(timeout=1.0)
                if msg[0] == "state":
                    return msg[1]
                # "terminal" или "error" до начала боя — ждём следующего эпизода
            except queue.Empty:
                pass
        return None


# ── Утилиты ───────────────────────────────────────────────────────────

def _screen_str(game) -> str:
    s = str(game.screen_type).upper()
    return s.split(".")[-1] if "." in s else s


def _drain(q: queue.Queue):
    while not q.empty():
        try:
            q.get_nowait()
        except queue.Empty:
            break


def _auto_navigate(screen: str, game):
    """Простые правила навигации для небоевых экранов (во время обучения PPO)."""
    if screen == "MAP":
        return ChooseAction(0)

    elif screen == "CARD_REWARD":
        return ProceedAction()  # Пропускаем карты при обучении

    elif screen in ("CHEST", "OPEN_CHEST"):
        return ProceedAction()

    elif screen in ("GRID", "HAND_SELECT"):
        screen_obj = getattr(game, "screen", None)
        if screen_obj and getattr(screen_obj, "confirm_up", False):
            return ProceedAction()
        return ChooseAction(0)

    elif screen == "BOSS_REWARD":
        return ChooseAction(0)

    elif screen == "REST":
        options = getattr(getattr(game, "screen", None), "rest_options", [])
        if not options:
            return ProceedAction()  # Отдых завершён — жмём Proceed
        player = game.player
        if player is not None:
            hp_pct = player.current_hp / max(player.max_hp, 1)
            opt_strs = [str(o).upper() for o in options]
            if hp_pct < 0.5:
                for i, o in enumerate(opt_strs):
                    if o in ("REST", "SLEEP"):
                        return ChooseAction(i)
            else:
                for i, o in enumerate(opt_strs):
                    if o == "SMITH":
                        return ChooseAction(i)
        return ChooseAction(0)

    elif screen == "EVENT":
        return ChooseAction(0)

    elif screen == "COMBAT_REWARD":
        return ProceedAction()

    elif screen == "GAME_OVER":
        return ProceedAction()

    else:
        return ProceedAction()
