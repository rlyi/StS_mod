"""main.py — точка входа.

Режимы (config.py):
  BENCHMARK_MODE = False → живая игра через CommunicationMod
  BENCHMARK_MODE = True  → бенчмарк на фиксированных сидах (META_AGENT, BENCHMARK_SEEDS)

config.properties (живая игра):
  command=python C:/StS_mod/main.py
  runAtGameStart=true
"""

import os
import sys
import json
import logging
import importlib
from collections import defaultdict

import config as _cfg

_ROOT    = os.path.dirname(os.path.abspath(__file__))
_LOG_DIR = os.path.join(_ROOT, "logs")
os.makedirs(_LOG_DIR, exist_ok=True)

_log_name = "benchmark" if _cfg.BENCHMARK_MODE else "ai"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(_LOG_DIR, f"{_log_name}.log"), encoding="utf-8"),
    ],
)
log = logging.getLogger("StSAI")
sys.stderr = open(os.path.join(_LOG_DIR, f"{_log_name}_errors.log"), "a", encoding="utf-8")

sys.path.insert(0, _ROOT)

from agents import make_meta_agent as _load_meta_agent
from spirecomm.communication.coordinator import Coordinator
from spirecomm.communication.action import (
    StateAction, ProceedAction, EndTurnAction, StartGameAction,
)
from spirecomm.spire.character import PlayerClass


def _screen_str(game) -> str:
    s = str(game.screen_type).upper()
    return s.split(".")[-1] if "." in s else s


# ── Живая игра ────────────────────────────────────────────────────────────────

class SlayTheSpireAI:
    def __init__(self):
        self._last_screen  = ""
        self._turn         = 0

        log.info("Загружаем агентов...")
        from agents.graph_battle_agent import GraphBattleAgent
        self._combat_agent = GraphBattleAgent()
        log.info("Боевой агент: GraphBattleAgent (BFS)")
        self._meta_agent = _load_meta_agent()

        self.coordinator = Coordinator()
        self.coordinator.register_state_change_callback(self._handle_game_state)
        self.coordinator.register_out_of_game_callback(self._handle_out_of_game)
        self.coordinator.register_command_error_callback(self._handle_error)
        self.coordinator.signal_ready()
        log.info("signal_ready отправлен. Ожидание игры...")

    def _handle_out_of_game(self):
        if self._meta_agent is not None:
            self._meta_agent.reset_run()
        if hasattr(self._combat_agent, 'reset_run'):
            self._combat_agent.reset_run()
        player_class = PlayerClass[_cfg.CHARACTER]
        return StartGameAction(player_class, ascension_level=0, seed=_cfg.SEED)

    def _handle_error(self, error):
        self._error_count = getattr(self, "_error_count", 0) + 1
        log.warning("Ошибка от игры (#%d): %s", self._error_count, error)
        if self._error_count >= 3:
            self._error_count = 0
            return ProceedAction()
        return StateAction()

    def _handle_game_state(self, game):
        screen = _screen_str(game)
        player = game.player

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
                         getattr(game, "floor", "?"), player.current_hp, player.max_hp)
            return self._combat_agent.act(game)
        elif (game.in_combat
              and getattr(game, 'current_action', None) in ('DiscardAction', 'ExhaustAction')
              and getattr(self._combat_agent, 'handles_forced_discard', False)):
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
        self.coordinator.run()


# ── Бенчмарк ─────────────────────────────────────────────────────────────────

_AGENT_MAP = {
    "llm":  ("agents.meta_llm_agent",  "LLMMetaAgent"),
    "rule": ("agents.meta_rule_agent", "RuleMetaAgent"),
}


class BenchmarkRunner:
    def __init__(self):
        agent_key = _cfg.META_AGENT
        log.info("Загрузка агента: %s", agent_key)
        mod, cls = _AGENT_MAP[agent_key]
        self._meta_agent  = getattr(importlib.import_module(mod), cls)()
        self._agent_name  = agent_key
        log.info("  %s OK", agent_key)

        from agents.graph_battle_agent import GraphBattleAgent
        self._combat_agent = GraphBattleAgent()
        log.info("  combat=graph OK")

        self._seeds        = _cfg.BENCHMARK_SEEDS
        self._runs_total   = _cfg.BENCHMARK_RUNS_PER_SEED
        self._results_file = _cfg.BENCHMARK_RESULTS_FILE

        self._seed_idx    = 0
        self._run_num     = 0
        self._last_floor  = 0
        self._error_count = 0

        self._results: list[dict] = []
        self._load_and_restore()

        total = len(self._seeds) * self._runs_total
        done  = len(self._results)
        log.info("Бенчмарк: агент=%s | %d сидов × %d забегов = %d итого (%d выполнено)",
                 agent_key, len(self._seeds), self._runs_total, total, done)

        self._coordinator = Coordinator()
        self._coordinator.register_state_change_callback(self._handle_state)
        self._coordinator.register_out_of_game_callback(self._handle_out_of_game)
        self._coordinator.register_command_error_callback(self._handle_error)
        self._coordinator.signal_ready()

    @property
    def _seed(self) -> int:
        return self._seeds[self._seed_idx]

    def _is_finished(self) -> bool:
        return self._seed_idx >= len(self._seeds)

    def _load_and_restore(self):
        if os.path.exists(self._results_file):
            with open(self._results_file, encoding="utf-8") as f:
                self._results = json.load(f)
            log.info("Загружено %d существующих результатов", len(self._results))

        done_counts: dict[int, int] = defaultdict(int)
        for r in self._results:
            done_counts[r["seed"]] += 1

        for si, seed in enumerate(self._seeds):
            n = done_counts[seed]
            if n < self._runs_total:
                self._seed_idx = si
                self._run_num  = n
                log.info("Продолжение с: сид=%d забег=%d/%d", seed, n + 1, self._runs_total)
                return

        self._seed_idx = len(self._seeds)
        log.info("Бенчмарк уже полностью завершён.")

    def _record_and_advance(self):
        self._results.append({"seed": self._seed, "floor": self._last_floor})
        os.makedirs(os.path.dirname(self._results_file), exist_ok=True)
        with open(self._results_file, "w", encoding="utf-8") as f:
            json.dump(self._results, f, indent=2, ensure_ascii=False)

        log.info("забег %d/%d | сид=%-5d | этаж=%d",
                 len(self._results), len(self._seeds) * self._runs_total,
                 self._seed, self._last_floor)

        self._run_num += 1
        if self._run_num >= self._runs_total:
            self._run_num = 0
            self._seed_idx += 1

    def _handle_out_of_game(self):
        if self._is_finished():
            floors = [r["floor"] for r in self._results]
            avg = sum(floors) / len(floors) if floors else 0
            log.info("БЕНЧМАРК ЗАВЕРШЁН | агент=%s | avg_floor=%.1f | забегов=%d",
                     self._agent_name, avg, len(floors))
            sys.exit(0)

        self._meta_agent.reset_run()
        self._last_floor  = 0
        self._error_count = 0
        return StartGameAction(PlayerClass.IRONCLAD, ascension_level=0, seed=self._seed)

    def _handle_state(self, game):
        if self._is_finished():
            return StateAction()

        screen = _screen_str(game)
        player = game.player

        floor = getattr(game, "floor", None)
        if floor:
            self._last_floor = floor

        if screen in ("GAME_OVER", "COMPLETE"):
            self._record_and_advance()
            return ProceedAction()

        if screen != "NONE":
            self._error_count = 0
            return self._meta_agent.act(game)

        if player is None or player.current_hp <= 0:
            return StateAction()
        if not any(getattr(m, "current_hp", 0) > 0 and not getattr(m, "is_gone", False)
                   for m in game.monsters if m is not None):
            return StateAction()
        if "EXECUTING" in str(getattr(game, "action_phase", "")).upper():
            return StateAction()
        if self._error_count >= 2:
            self._error_count = 0
            return EndTurnAction()

        return self._combat_agent.act(game)

    def _handle_error(self, error):
        self._error_count += 1
        log.warning("Ошибка (#%d): %s", self._error_count, error)
        return StateAction()

    def run(self):
        self._coordinator.run()


# ── Точка входа ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if _cfg.BENCHMARK_MODE:
        BenchmarkRunner().run()
    else:
        SlayTheSpireAI().run()
