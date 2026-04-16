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
# 10 (player) + 5*7 (hand: type×4 + dmg + block + cost) + 4*4 (enemies) = 61
OBS_SIZE    = 61
ACTION_SIZE = 26  # 5 cards × (no target + 4 enemies) + end turn (action 25)

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

# ── Card ID → integer index ────────────────────────────────────────────
# Только Ironclad + статусные/проклятые карты
CARD_TO_IDX = {
    # Стартовые
    "Strike_R": 0, "Defend_R": 1, "Bash": 2,
    # Обычные
    "Anger": 3, "Armaments": 4, "Body Slam": 5, "Clash": 6,
    "Cleave": 7, "Clothesline": 8, "Flex": 9, "Havoc": 10,
    "Headbutt": 11, "Heavy Blade": 12, "Iron Wave": 13,
    "Perfected Strike": 14, "Pommel Strike": 15, "Shrug It Off": 16,
    "Sword Boomerang": 17, "Thunderclap": 18, "True Grit": 19,
    "Twin Strike": 20, "Warcry": 21, "Wild Strike": 22,
    # Необычные
    "Battle Trance": 23, "Blood for Blood": 24, "Bloodletting": 25,
    "Burning Pact": 26, "Carnage": 27, "Combust": 28, "Dark Embrace": 29,
    "Disarm": 30, "Dropkick": 31, "Dual Wield": 32, "Entrench": 33,
    "Evolve": 34, "Fire Breathing": 35, "Flame Barrier": 36,
    "Ghostly Armor": 37, "Hemokinesis": 38, "Infernal Blade": 39,
    "Inflame": 40, "Intimidate": 41, "Metallicize": 42, "Power Through": 43,
    "Pummel": 44, "Rage": 45, "Rampage": 46, "Reckless Charge": 47,
    "Rupture": 48, "Searing Blow": 49, "Second Wind": 50, "Seeing Red": 51,
    "Sentinel": 52, "Sever Soul": 53, "Shockwave": 54, "Spot Weakness": 55,
    "Uppercut": 56, "Whirlwind": 57,
    # Редкие
    "Barricade": 58, "Berserk": 59, "Bludgeon": 60, "Brutality": 61,
    "Champion": 62, "Choke": 63, "Corruption": 64, "Dark Shackles": 65,
    "Double Tap": 66, "Exhume": 67, "Feed": 68, "Fiend Fire": 69,
    "Immolate": 70, "Impervious": 71, "Juggernaut": 72, "Limit Break": 73,
    "Offering": 74, "Reaper": 75,
    # Статусные / проклятия
    "Slimed": 76, "Wound": 77, "Dazed": 78, "Burn": 79, "Void": 80,
    "Curse of the Bell": 81, "Normality": 82, "Pain": 83, "Parasite": 84,
    "Pride": 85, "Regret": 86, "Shame": 87, "Decay": 88, "Doubt": 89,
    "Clumsy": 90, "Injury": 91,
    # Неизвестная карта
    "UNKNOWN": 99,
}
CARD_IDX_UNKNOWN = 99

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
