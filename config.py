import os

# ── Game settings ──────────────────────────────────────────────────────
SEED = 42
CHARACTER = "IRONCLAD"
MAX_ACT = 1
MAX_FLOOR = 17  # Act 1: floors 1-17

# ── Meta agent selection ───────────────────────────────────────────────
# Варианты: "random" | "tree" | "forest" | "imitation" | "llm"
META_AGENT = "tree"

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR   = os.path.join(PROJECT_ROOT, "models")
DATA_RUNS_DIR = os.path.join(PROJECT_ROOT, "data", "runs")
STS_RUNS_DIR  = r"C:\Users\User\Documents\My Games\SlayTheSpire\runs"

# ── Observation / action space ─────────────────────────────────────────
# 10 (player) + 5*7 (hand: type×4 + dmg + block + cost) + 4*4 (enemies) + 2*2 (potions) = 65
OBS_SIZE    = 65
ACTION_SIZE = 36  # 5 cards × (no target + 4 enemies) + end turn (25) + 2 potions × 5 = 36
POTION_SLOTS = 2  # слотов зелий в наблюдении

# ── Свойства карт: (базовый_урон, базовый_блок) ────────────────────────
# Урон и блок — базовые значения без учёта силы/ловкости
CARD_PROPERTIES = {
    # Стартовые
    "Strike_R":         (6,  0),
    "Defend_R":         (0,  5),
    "Bash":             (8,  0),
    # Обычные атаки
    "Anger":            (6,  0),
    "Body Slam":        (0,  0),   # урон = текущий блок
    "Clash":            (14, 0),
    "Cleave":           (8,  0),
    "Clothesline":      (12, 0),
    "Headbutt":         (9,  0),
    "Heavy Blade":      (14, 0),
    "Iron Wave":        (5,  5),
    "Perfected Strike": (6,  0),
    "Pommel Strike":    (9,  0),
    "Sword Boomerang":  (9,  0),   # 3×3
    "Thunderclap":      (4,  0),
    "Twin Strike":      (10, 0),   # 2×5
    "Wild Strike":      (12, 0),
    # Обычные скиллы
    "Armaments":        (0,  5),
    "Flex":             (0,  0),
    "Havoc":            (0,  0),
    "Shrug It Off":     (0,  8),
    "True Grit":        (0,  7),
    "Warcry":           (0,  0),
    # Необычные атаки
    "Blood for Blood":  (18, 0),
    "Carnage":          (20, 0),
    "Dropkick":         (5,  0),
    "Hemokinesis":      (15, 0),
    "Pummel":           (8,  0),   # 4×2
    "Rampage":          (8,  0),
    "Reckless Charge":  (7,  0),
    "Searing Blow":     (12, 0),
    "Sever Soul":       (16, 0),
    "Uppercut":         (13, 0),
    "Whirlwind":        (5,  0),   # ×energy
    # Необычные скиллы/пауэры
    "Battle Trance":    (0,  0),
    "Bloodletting":     (0,  0),
    "Burning Pact":     (0,  0),
    "Combust":          (0,  0),
    "Dark Embrace":     (0,  0),
    "Disarm":           (0,  0),
    "Dual Wield":       (0,  0),
    "Entrench":         (0,  0),
    "Evolve":           (0,  0),
    "Fire Breathing":   (0,  0),
    "Flame Barrier":    (0,  12),
    "Ghostly Armor":    (0,  10),
    "Infernal Blade":   (0,  0),
    "Inflame":          (0,  0),
    "Intimidate":       (0,  0),
    "Metallicize":      (0,  0),
    "Power Through":    (0,  15),
    "Rage":             (0,  0),
    "Rupture":          (0,  0),
    "Second Wind":      (0,  0),
    "Seeing Red":       (0,  0),
    "Sentinel":         (0,  5),
    "Shockwave":        (0,  0),
    "Spot Weakness":    (0,  0),
    # Редкие атаки
    "Bludgeon":         (32, 0),
    "Choke":            (12, 0),
    "Feed":             (10, 0),
    "Fiend Fire":       (7,  0),   # ×cards
    "Immolate":         (21, 0),
    "Reaper":           (4,  0),   # ×enemies
    # Редкие скиллы/пауэры
    "Barricade":        (0,  0),
    "Berserk":          (0,  0),
    "Brutality":        (0,  0),
    "Champion":         (0,  0),
    "Corruption":       (0,  0),
    "Dark Shackles":    (0,  0),
    "Double Tap":       (0,  0),
    "Exhume":           (0,  0),
    "Impervious":       (0,  30),
    "Juggernaut":       (0,  0),
    "Limit Break":      (0,  0),
    "Offering":         (0,  0),
    # Статусные / проклятия — нет урона/блока
}

# ── Monster intent → integer index ────────────────────────────────────
INTENT_TO_IDX = {
    "ATTACK":        0,
    "ATTACK_DEBUFF": 1,
    "ATTACK_BUFF":   2,
    "ATTACK_DEFEND": 3,
    "BUFF":          4,
    "DEBUFF":        5,
    "DEFEND":        6,
    "DEFEND_BUFF":   7,
    "DEFEND_DEBUFF": 8,
    "ESCAPE":        9,
    "MAGIC":         10,
    "NONE":          11,
    "SLEEP":         12,
    "STUN":          13,
    "UNKNOWN":       14,
}
INTENT_MAX_IDX = 14

# ── Награды ────────────────────────────────────────────────────────────
REWARD_WIN              =  2.0    # все враги убиты
REWARD_KILL_ENEMY       =  0.5    # за каждое убийство (было 0.1)
REWARD_DAMAGE_MULT      =  0.01   # за единицу нанесённого урона (было 0.005)
REWARD_LOSE             = -2.0    # смерть игрока
REWARD_DAMAGE_TAKEN_MULT= -0.03   # за единицу полученного урона (было -0.01)
REWARD_TURN_PENALTY     = -0.01   # штраф за каждый ход
REWARD_ACT1_BOSS        = 10.0    # победа над боссом Акта 1

# Боссы Акта 1 (для определения победы)
ACT1_BOSSES = {"The Champ", "Hexaghost", "Slime Boss"}

# Враги Акта 1 (для справки)
ACT1_ENEMIES = {
    "Jaw Worm", "Cultist", "Red Louse", "Green Louse",
    "Acid Slime (S)", "Acid Slime (M)", "Spike Slime (S)", "Spike Slime (M)",
    "Gremlin Nob", "Lagavulin",
    "Gremlin Fat", "Gremlin Sneaky", "Gremlin Mad", "Gremlin Shield", "Gremlin Wizard",
}
