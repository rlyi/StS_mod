"""benchmark.py — сравнение мета-агентов на фиксированных сидах.

config.properties для запуска:
  command=python C:/StS_mod/benchmark.py

Что делает:
  Для каждого агента (random / tree / forest) × каждого сида × RUNS_PER_SEED забегов:
    - Стартует игру с фиксированным сидом через StartGameAction
    - CombatAgent (PPO) ведёт бои
    - Текущий мета-агент ведёт навигацию
    - На GAME_OVER / COMPLETE записывает результат в data/benchmark_results.json
  После всех забегов печатает итоговую таблицу в лог.

Поддерживает продолжение: если benchmark_results.json уже есть, пропускает
уже сыгранные комбинации (агент × сид × run_num).
"""

import os
import sys
import json
import logging
import datetime
import importlib
from collections import defaultdict

_ROOT = os.path.dirname(os.path.abspath(__file__))
_LOG_DIR = os.path.join(_ROOT, "logs")
os.makedirs(_LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(_LOG_DIR, "benchmark.log"), encoding="utf-8"),
    ],
)
log = logging.getLogger("Benchmark")
sys.stderr = open(os.path.join(_LOG_DIR, "benchmark_errors.log"), "a", encoding="utf-8")

sys.path.insert(0, _ROOT)

from spirecomm.communication.coordinator import Coordinator
from spirecomm.communication.action import (
    StateAction, ProceedAction, StartGameAction, EndTurnAction
)
from spirecomm.spire.character import PlayerClass

# ── Конфигурация (менять перед каждым запуском) ───────────────────────────
AGENT         = "tree"          # "random" | "tree" | "forest"
SEEDS         = [42, 123, 456]    # список сидов
RUNS_PER_SEED = 10                # забегов на каждый сид
RESULTS_FILE  = os.path.join(_ROOT, "data", f"benchmark_{AGENT}.json")
# ─────────────────────────────────────────────────────────────────────────

_AGENT_MAP = {
    "random": ("agents.meta_agent",        "RandomMetaAgent"),
    "tree":   ("agents.meta_tree_agent",   "DecisionTreeMetaAgent"),
    "forest": ("agents.meta_forest_agent", "RandomForestMetaAgent"),
}

BENCHMARK_AGENTS = [AGENT]
BENCHMARK_SEEDS  = SEEDS


def _screen_str(game) -> str:
    s = str(game.screen_type).upper()
    return s.split(".")[-1] if "." in s else s


class BenchmarkRunner:
    def __init__(self):
        log.info("Загрузка агента: %s", AGENT)
        mod, cls = _AGENT_MAP[AGENT]
        self._meta_agents = {AGENT: getattr(importlib.import_module(mod), cls)()}
        log.info("  %s OK", AGENT)

        from agents.combat_agent import CombatAgent
        self._combat_agent = CombatAgent()
        log.info("  CombatAgent OK")

        # Позиция в бенчмарке
        self._agent_idx = 0
        self._seed_idx  = 0
        self._run_num   = 0   # номер текущего забега внутри seed × agent

        # Текущий забег — отслеживаем состояние для записи результата
        self._last_floor  = 0
        self._last_hp     = 0
        self._last_max_hp = 0
        self._error_count = 0

        # Результаты
        self._results: list[dict] = []
        self._load_and_restore()

        total = len(BENCHMARK_AGENTS) * len(BENCHMARK_SEEDS) * RUNS_PER_SEED
        done  = len(self._results)
        log.info("Бенчмарк: %d агентов × %d сидов × %d забегов = %d итого (%d выполнено)",
                 len(BENCHMARK_AGENTS), len(BENCHMARK_SEEDS), RUNS_PER_SEED, total, done)

        self._coordinator = Coordinator()
        self._coordinator.register_state_change_callback(self._handle_state)
        self._coordinator.register_out_of_game_callback(self._handle_out_of_game)
        self._coordinator.register_command_error_callback(self._handle_error)
        self._coordinator.signal_ready()

    # ── Свойства текущей позиции ──────────────────────────────────────────

    @property
    def _agent_name(self) -> str:
        return BENCHMARK_AGENTS[self._agent_idx]

    @property
    def _seed(self) -> int:
        return BENCHMARK_SEEDS[self._seed_idx]

    @property
    def _meta_agent(self):
        return self._meta_agents[self._agent_name]

    def _is_finished(self) -> bool:
        return self._agent_idx >= len(BENCHMARK_AGENTS)

    # ── Загрузка / восстановление позиции ────────────────────────────────

    def _load_and_restore(self):
        if os.path.exists(RESULTS_FILE):
            with open(RESULTS_FILE, encoding="utf-8") as f:
                self._results = json.load(f)
            log.info("Загружено %d существующих результатов", len(self._results))

        done_counts: dict[tuple, int] = defaultdict(int)
        for r in self._results:
            done_counts[(r["agent"], r["seed"])] += 1

        for ai, agent in enumerate(BENCHMARK_AGENTS):
            for si, seed in enumerate(BENCHMARK_SEEDS):
                n = done_counts[(agent, seed)]
                if n < RUNS_PER_SEED:
                    self._agent_idx = ai
                    self._seed_idx  = si
                    self._run_num   = n
                    log.info("Продолжение с: агент=%s сид=%d забег=%d/%d",
                             agent, seed, n + 1, RUNS_PER_SEED)
                    return

        self._agent_idx = len(BENCHMARK_AGENTS)  # всё готово
        log.info("Бенчмарк уже полностью завершён.")

    # ── Запись результата и переход к следующему ──────────────────────────

    def _record_and_advance(self, outcome: str):
        result = {
            "agent":   self._agent_name,
            "seed":    self._seed,
            "run":     self._run_num,
            "outcome": outcome,
            "floor":   self._last_floor,
            "hp":      self._last_hp,
            "max_hp":  self._last_max_hp,
            "ts":      datetime.datetime.now().isoformat(),
        }
        self._results.append(result)
        os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
        with open(RESULTS_FILE, "w", encoding="utf-8") as f:
            json.dump(self._results, f, indent=2, ensure_ascii=False)

        log.info("РЕЗУЛЬТАТ | агент=%-6s сид=%-5d забег=%2d/%d | %-4s этаж=%-2d HP=%d/%d",
                 self._agent_name, self._seed, self._run_num + 1, RUNS_PER_SEED,
                 outcome.upper(), self._last_floor, self._last_hp, self._last_max_hp)

        # Переходим к следующей позиции
        self._run_num += 1
        if self._run_num >= RUNS_PER_SEED:
            self._run_num = 0
            self._seed_idx += 1
            if self._seed_idx >= len(BENCHMARK_SEEDS):
                self._seed_idx = 0
                self._agent_idx += 1
                if not self._is_finished():
                    log.info("Переход к агенту: %s", self._agent_name)

    # ── Callbacks ────────────────────────────────────────────────────────

    def _handle_out_of_game(self):
        if self._is_finished():
            log.info("БЕНЧМАРК ЗАВЕРШЁН! Всего: %d результатов", len(self._results))
            self._print_summary()
            return StateAction()

        log.info("Старт: агент=%-6s сид=%-5d забег=%d/%d",
                 self._agent_name, self._seed, self._run_num + 1, RUNS_PER_SEED)
        self._last_floor  = 0
        self._last_hp     = 0
        self._last_max_hp = 0
        self._error_count = 0
        return StartGameAction(PlayerClass.IRONCLAD, ascension_level=0, seed=self._seed)

    def _handle_state(self, game):
        if self._is_finished():
            return StateAction()

        screen = _screen_str(game)
        player = game.player

        # Обновляем трекинг текущего состояния
        floor = getattr(game, "floor", None)
        if floor:
            self._last_floor = floor
        if player:
            self._last_hp     = player.current_hp
            self._last_max_hp = player.max_hp

        # ── Конец забега ─────────────────────────────────────────────
        if screen == "GAME_OVER":
            victory = getattr(getattr(game, "screen", None), "victory", False)
            self._record_and_advance("win" if victory else "lose")
            return ProceedAction()

        if screen == "COMPLETE":
            self._record_and_advance("win")
            return ProceedAction()

        # ── Навигация — мета-агент ────────────────────────────────────
        if screen != "NONE":
            self._error_count = 0
            return self._meta_agent.act(game)

        # ── Бой — CombatAgent ─────────────────────────────────────────
        if player is None:
            return StateAction()
        if player.current_hp <= 0:
            return StateAction()
        if not any(getattr(m, "current_hp", 0) > 0
                   for m in game.monsters if m is not None):
            return StateAction()

        action_phase = str(getattr(game, "action_phase", "")).upper()
        if "EXECUTING" in action_phase:
            return StateAction()

        if self._error_count >= 2:
            self._error_count = 0
            return EndTurnAction()

        return self._combat_agent.act(game)

    def _handle_error(self, error):
        self._error_count += 1
        log.warning("Ошибка (#%d): %s", self._error_count, error)
        return StateAction()

    # ── Итоговая таблица ──────────────────────────────────────────────────

    def _print_summary(self):
        stats = defaultdict(lambda: {"wins": 0, "total": 0, "floors": [], "hp_pct": []})
        for r in self._results:
            s = stats[r["agent"]]
            s["total"] += 1
            s["floors"].append(r["floor"])
            if r["outcome"] == "win":
                s["wins"] += 1
            if r["max_hp"] > 0:
                s["hp_pct"].append(r["hp"] / r["max_hp"])

        log.info("=" * 60)
        log.info("%-8s | winrate | avg_floor | avg_hp_pct | runs", "agent")
        log.info("-" * 60)
        for agent in BENCHMARK_AGENTS:
            s = stats[agent]
            if s["total"] == 0:
                continue
            winrate   = s["wins"] / s["total"] * 100
            avg_floor = sum(s["floors"]) / len(s["floors"])
            avg_hp    = sum(s["hp_pct"]) / len(s["hp_pct"]) * 100 if s["hp_pct"] else 0
            log.info("%-8s | %5.1f%%  | %9.1f | %10.1f%% | %d",
                     agent, winrate, avg_floor, avg_hp, s["total"])
        log.info("=" * 60)

    def run(self):
        self._coordinator.run()


if __name__ == "__main__":
    BenchmarkRunner().run()
