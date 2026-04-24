"""main.py — точка входа для CommunicationMod.

config.properties:
  command=python C:/StS_mod/main.py
  runAtGameStart=true

ВАЖНО: stdout/stdin заняты протоколом CommunicationMod.
Весь вывод — в logs/ai.log и logs/ai_errors.log.

Порядок запуска:
  1. Создаём Coordinator и сразу signal_ready() — до любых тяжёлых импортов
  2. Загружаем агентов (SB3, sklearn — могут занять несколько секунд)
  3. Переключаем callback на боевой
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


def _load_meta_agent():
    """Загружает мета-агента согласно META_AGENT в config.py."""
    from config import META_AGENT
    _AGENTS = {
        "random": ("agents.meta_agent",        "RandomMetaAgent"),
        "tree":   ("agents.meta_tree_agent",  "DecisionTreeMetaAgent"),
        "forest": ("agents.meta_forest_agent", "RandomForestMetaAgent"),
        "llm":    ("agents.meta_llm_agent",    "LLMMetaAgent"),
    }
    module_name, class_name = _AGENTS.get(META_AGENT, _AGENTS["random"])
    import importlib
    cls = getattr(importlib.import_module(module_name), class_name)
    log.info("Мета-агент: %s (%s)", class_name, META_AGENT)
    return cls()


# spirecomm импортируется быстро — можно сразу
from spirecomm.communication.coordinator import Coordinator
from spirecomm.communication.action import StateAction, ProceedAction, EndTurnAction


def _screen_str(game) -> str:
    s = str(game.screen_type).upper()
    return s.split(".")[-1] if "." in s else s


class SlayTheSpireAI:
    def __init__(self):
        self._combat_agent = None
        self._meta_agent   = None
        self._agents_ready = False
        self._last_screen  = ""
        self._turn         = 0

        self.coordinator = Coordinator()

        # ── Шаг 1: signal_ready() ДО загрузки тяжёлых модулей ─────────
        # CommunicationMod даёт только 10 секунд. Успеваем до таймаута.
        self.coordinator.register_state_change_callback(self._handle_loading)
        self.coordinator.register_out_of_game_callback(lambda: StateAction())
        self.coordinator.register_command_error_callback(self._handle_error)
        self.coordinator.signal_ready()
        log.info("signal_ready отправлен — загружаем агентов...")

        # ── Шаг 2: загружаем тяжёлые агенты ───────────────────────────
        # Мета-агент выбирается через META_AGENT в config.py.
        from agents.combat_agent import CombatAgent
        self._combat_agent = CombatAgent()
        self._meta_agent   = _load_meta_agent()
        self._agents_ready = True

        # ── Шаг 3: переключаем callback на боевой ─────────────────────
        self.coordinator.register_state_change_callback(self._handle_game_state)
        log.info("Агенты готовы. Ожидание игры...")

    # ── Callbacks ────────────────────────────────────────────────────

    def _handle_loading(self, game):
        """Пока агенты грузятся — просто опрашиваем состояние."""
        return StateAction()

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
