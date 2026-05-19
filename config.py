import os

# ── Game settings ──────────────────────────────────────────────────────
SEED = None
CHARACTER = "IRONCLAD"

# ── Meta agent selection ───────────────────────────────────────────────
# Варианты: "rule" | "llm"
META_AGENT = "rule"

# ── Target deck (Perfected Strike build) ───────────────────────────────
# Карта → максимум копий, которые стоит набирать.
# Порядок важен: агент берёт карты сверху вниз по приоритету.
TARGET_DECK: dict[str, int] = {
    'perfected strike': 5,
    'battle trance':    2,
    'offering':         1,
    'reaper':           2,
    'twin strike':      2,
    'shockwave':        2,
    'thunderclap':      2,
    'dropkick':         2,
    'pommel strike':    2,
    'shrug it off':     2,
    'impervious':       2,
    'ghostly armor':    1,
    'flame barrier':    1,
    'blind':            1,
    'apotheosis':       1,
    'handofgreed':      1,
    'master of strategy': 1,
    'flash of steel':   1,
    'trip':             1,
    'dark shackles':    1,
    'swift strike':     1,
    'dramatic entrance': 1,
    'finesse':          1,
}

# Карты, которые удаляются в первую очередь (смит / магазин / события).
# Всё остальное вне TARGET_DECK удаляется после них.
REMOVAL_PRIORITY: list[str] = ['defend', 'strike']

# Приоритет реликтов босса (первый = самый желанный).
BOSS_RELIC_PRIORITY: list[str] = [
    'sozu', 'runic dome', "philosopher's stone", 'ectoplasm',
    'velvet choker', 'cursed key', 'fusion hammer', 'snecko eye',
    'mark of pain', 'busted crown', 'coffee dripper', "slaver's collar",
    'runic cube', 'runic pyramid', 'black blood', 'calling bell',
    'empty cage', 'black star', 'sacred bark',
]

# Приоритет апгрейда у костра.
# По умолчанию — порядок карт из TARGET_DECK плюс стартовые карты в конце.
UPGRADE_PRIORITY: list[str] = [
    'apotheosis', 'perfected strike', 'bash', 'shockwave', 'battle trance',
    'offering', 'blind', 'dropkick', 'flame barrier', 'twin strike',
    'pommel strike', 'thunderclap', 'shrug it off', 'impervious',
    'ghostly armor', 'master of strategy', 'flash of steel', 'trip',
    'dark shackles', 'swift strike', 'dramatic entrance', 'finesse',
    'reaper', 'handofgreed',
]

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# ── Benchmark ──────────────────────────────────────────────────────────
# BENCHMARK_MODE = True → запустить бенчмарк вместо живой игры
BENCHMARK_MODE          = False
BENCHMARK_RUNS_PER_SEED = 1
BENCHMARK_RESULTS_FILE  = os.path.join(PROJECT_ROOT, f"benchmark_{META_AGENT}.json")
BENCHMARK_SEEDS         = [
    101, 202, 303, 404, 505, 606, 707, 808, 909, 1010,
    1111, 1212, 1313, 1414, 1515, 1616, 1717, 1818, 1919, 2020,
    2121, 2222, 2323, 2424, 2525, 2626, 2727, 2828, 2929, 3030,
    3131, 3232, 3333, 3434, 3535, 3636, 3737, 3838, 3939, 4040,
    4141, 4242, 4343, 4444, 4545, 4646, 4747, 4848, 4949, 5050,
    5151, 5252, 5353, 5454, 5555, 5656, 5757, 5858, 5959, 6060,
    6161, 6262, 6363, 6464, 6565, 6666, 6767, 6868, 6969, 7070,
    7171, 7272, 7373, 7474, 7575, 7676, 7777, 7878, 7979, 8080,
    8181, 8282, 8383, 8484, 8585, 8686, 8787, 8888, 8989, 9090,
    9191, 9292, 9393, 9494, 9595, 9696, 9797, 9898, 9999, 10000,
]
