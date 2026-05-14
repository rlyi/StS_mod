"""main.py — точка входа для CommunicationMod.

config.properties:
  command=python C:/StS_mod/main.py
  runAtGameStart=true

ВАЖНО: stdout/stdin заняты протоколом CommunicationMod.
Весь вывод — в logs/ai.log и logs/ai_errors.log.

Порядок запуска:
  1. Загружаем агентов (~7 сек, в пределах таймаута CommunicationMod 10 сек)
  2. signal_ready() — к этому моменту Python уже готов к первому сообщению игры
  3. coordinator.run() обрабатывает out_of_game → сразу стартует ран
"""

import os
import sys
import logging

# ── Логирование в файл (stdout занят протоколом игры) ─────────────────
_ROOT = os.path.dirname(os.path.abspath(__file__))
_LOG_DIR = os.path.join(_ROOT, "logs")
os.makedirs(_LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(_LOG_DIR, "ai.log"), encoding="utf-8"),
    ],
)
log = logging.getLogger("StSAI")
sys.stderr = open(os.path.join(_LOG_DIR, "ai_errors.log"), "a", encoding="utf-8")

sys.path.insert(0, _ROOT)


from agents import make_meta_agent as _load_meta_agent


# spirecomm импортируется быстро — можно сразу
from spirecomm.communication.coordinator import Coordinator
from spirecomm.communication.action import StateAction, ProceedAction, EndTurnAction, StartGameAction
from spirecomm.spire.character import PlayerClass


def _screen_str(game) -> str:
    s = str(game.screen_type).upper()
    return s.split(".")[-1] if "." in s else s


class SlayTheSpireAI:
    def __init__(self):
        self._combat_agent = None
        self._meta_agent   = None
        self._last_screen  = ""
        self._turn         = 0

        # ── Шаг 1: загружаем агентов (~7 сек, укладываемся в таймаут 10 сек) ─
        log.info("Загружаем агентов...")
        from agents.combat_agent import CombatAgent
        self._combat_agent = CombatAgent()
        self._meta_agent   = _load_meta_agent()

        # ── Шаг 2: signal_ready() — агенты уже готовы ─────────────────
        self.coordinator = Coordinator()
        self.coordinator.register_state_change_callback(self._handle_game_state)
        self.coordinator.register_out_of_game_callback(self._handle_out_of_game)
        self.coordinator.register_command_error_callback(self._handle_error)
        self.coordinator.signal_ready()
        log.info("signal_ready отправлен. Ожидание игры...")

    # ── Callbacks ────────────────────────────────────────────────────

    def _handle_out_of_game(self):
        if self._meta_agent is not None:
            self._meta_agent.reset_run()
        from config import CHARACTER, SEED
        player_class = PlayerClass[CHARACTER]
        return StartGameAction(player_class, ascension_level=0, seed=SEED)

    def _handle_error(self, error):
        self._error_count = getattr(self, "_error_count", 0) + 1
        log.warning("Ошибка от игры (#%d): %s", self._error_count, error)
        if self._error_count >= 3:
            self._error_count = 0
            return ProceedAction()
        return StateAction()

    def _handle_game_state(self, game):
        screen = _screen_str(game)
        player = game.player  # может быть None на стартовых экранах


        if screen == "NONE":
            if player is None:
                return StateAction()
            if player.current_hp <= 0:
                return StateAction()
            if not any(getattr(m, "current_hp", 0) > 0 and not getattr(m, "is_gone", False)
                       for m in game.monsters if m is not None):
                return StateAction()
            action_phase = str(getattr(game, "action_phase", "")).upper()
            if "EXECUTING" in action_phase:
                return StateAction()
            error_count = getattr(self, "_error_count", 0)
            if error_count >= 2:
                self._error_count = 0
                return EndTurnAction()
            self._turn += 1
            if self._turn == 1:
                log.info("Бой начался | этаж=%s HP=%d/%d",
                         getattr(game, "floor", "?"),
                         player.current_hp, player.max_hp)
            return self._combat_agent.act(game)
        else:
            if screen != self._last_screen:
                hp_str = f"{player.current_hp}/{player.max_hp}" if player else "?/?"
                log.info("Экран: %-20s | этаж=%s HP=%s gold=%s",
                         screen, getattr(game, "floor", "?"),
                         hp_str, getattr(game, "gold", "?"))
                self._last_screen = screen
            self._turn = 0
            return self._meta_agent.act(game)

    def run(self):
        self.coordinator.run()  # блокирует навсегда


if __name__ == "__main__":
    ai = SlayTheSpireAI()
    ai.run()
