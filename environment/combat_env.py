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
import random
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


def _make_meta_agent():
    """Загружает мета-агента согласно META_AGENT в config.py."""
    import importlib
    from config import META_AGENT
    _AGENTS = {
        "random": ("agents.meta_agent",        "RandomMetaAgent"),
        "tree":   ("agents.meta_tree_agent",  "DecisionTreeMetaAgent"),
        "forest": ("agents.meta_forest_agent", "RandomForestMetaAgent"),
        "llm":    ("agents.meta_llm_agent",    "LLMMetaAgent"),
    }
    module_name, class_name = _AGENTS.get(META_AGENT, _AGENTS["random"])
    cls = getattr(importlib.import_module(module_name), class_name)
    log.info("Мета-агент: %s (%s)", class_name, META_AGENT)
    return cls()

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
        self._error_count = 0
        self._meta_agent = _make_meta_agent()
        self._current_mask: list[bool] = [False] * ACTION_SIZE
        self._current_mask[25] = True  # EndTurn всегда валиден

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
        self._current_mask = get_valid_actions(game)
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
            self._current_mask = get_valid_actions(game)
            return game_to_obs(game), reward, False, False, {}

        elif kind == "terminal":
            _, outcome, game = msg
            self._done = True
            obs = game_to_obs(game) if (game and game.player) else np.zeros(OBS_SIZE, dtype=np.float32)
            if game and game.player:
                reward = calculate_reward(self._prev_game, game)
            else:
                reward = 2.0 if outcome == "win" else -2.0
            return obs, reward, True, False, {"outcome": outcome}

        # Ошибка или неизвестное сообщение
        self._done = True
        return np.zeros(OBS_SIZE, dtype=np.float32), 0.0, True, False, {}

    def close(self):
        pass  # Coordinator работает через stdin/stdout — ОС закроет при выходе

    def action_masks(self) -> np.ndarray:
        """MaskablePPO вызывает этот метод после каждого step/reset."""
        return np.array(self._current_mask, dtype=bool)

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
        self._error_count += 1
        log.warning("Ошибка от игры (#%d): %s", self._error_count, error)
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
            hp = f"{game.current_hp}/{game.max_hp}" if game.max_hp else "?/?"
            log.info("РАН ЗАВЕРШЁН | исход=%-4s этаж=%-2s HP=%s", outcome, floor, hp)
            self._state_q.put(("terminal", outcome, game))
            return ProceedAction()

        # ── Конец боя (появился экран наград) ─────────────────────────
        if screen == "COMBAT_REWARD":
            screen_obj = getattr(game, "screen", None)
            rewards = getattr(screen_obj, "rewards", [])
            # Подбираем все награды: карты, зелья (даже если слоты полны — игра покажет экран
            # выбора замены), золото, реликвии. Не пропускаем зелья — иначе proceed будет
            # отклонён и сессия зависнет.
            for i, reward in enumerate(rewards):
                rt = str(getattr(reward, "reward_type", "")).upper().split(".")[-1]
                if rt == "POTION":
                    potions = getattr(game, "potions", [])
                    if any(getattr(p, "potion_id", "Potion Slot") == "Potion Slot"
                           for p in potions):
                        log.debug("COMBAT_REWARD: берём зелье #%d", i)
                        return ChooseAction(i)
                    log.debug("COMBAT_REWARD: слоты зелий полны — пропускаем зелье")
                    continue
                if rt in ("CARD", "GOLD", "RELIC", "RELIC_AND_GOLD",
                          "EMERALD_KEY", "SAPPHIRE_KEY", "STOLEN_GOLD"):
                    log.debug("COMBAT_REWARD: выбираем награду #%d (%s)", i, rt)
                    return ChooseAction(i)
            # Все награды обработаны — завершаем эпизод
            floor = getattr(game, "floor", "?")
            hp = f"{game.current_hp}/{game.max_hp}" if game.max_hp else "?/?"
            log.info("РАН ЗАВЕРШЁН | исход=win  этаж=%-2s HP=%s", floor, hp)
            self._state_q.put(("terminal", "win", game))
            return ProceedAction()

        # ── Не бой — передаём мета-агенту ───────────────────────────
        if screen != "NONE":
            self._error_count = 0
            action = self._meta_agent.act(game)
            log.info("Навигация: %-20s → %s", screen, type(action).__name__)
            return action

        # ── В бою: player может быть None на старте ───────────────────
        if player is None:
            return StateAction()

        # Игрок мёртв — ждём GAME_OVER, не кладём в очередь (иначе двойной штраф)
        if player.current_hp <= 0:
            return StateAction()

        # Все монстры мертвы или сбежали — бой завершается, ждём COMBAT_REWARD
        if not any(getattr(m, "current_hp", 0) > 0 and not getattr(m, "is_gone", False)
                   for m in game.monsters if m is not None):
            return StateAction()

        # Игра ещё анимирует предыдущее действие — ждём завершения
        action_phase = str(getattr(game, "action_phase", "")).upper()
        if "EXECUTING" in action_phase:
            return StateAction()

        # Если предыдущие команды отклонялись (игра между ходами), не нагружаем PPO —
        # просто завершаем ход, это первое действие которое разблокируется.
        if self._error_count >= 2:
            log.debug("Повторные ошибки (%d) в бою — EndTurn для разблокировки", self._error_count)
            self._error_count = 0
            return EndTurnAction()

        # ── В бою: ждём действие от основного потока ─────────────────
        self._state_q.put(("state", game))
        try:
            action_int = self._action_q.get(timeout=60)
        except queue.Empty:
            return EndTurnAction()

        self._error_count = 0
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


