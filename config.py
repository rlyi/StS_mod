import os

# ── Game settings ──────────────────────────────────────────────────────
SEED = 42
CHARACTER = "IRONCLAD"
MAX_ACT = 1
MAX_FLOOR = 17  # Act 1: floors 1-17

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR   = os.path.join(PROJECT_ROOT, "models")
DATA_RUNS_DIR = os.path.join(PROJECT_ROOT, "data", "runs")
STS_RUNS_DIR  = r"C:\Users\User\Documents\My Games\SlayTheSpire\runs"

# ── Observation / action space ─────────────────────────────────────────
# 3 (player) + 5*2 (hand) + 3*3 (enemies) = 22
OBS_SIZE    = 22
ACTION_SIZE = 16  # 5 cards × 3 target groups + end turn

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
REWARD_KILL_ENEMY       =  0.1    # за каждое убийство
REWARD_DAMAGE_MULT      =  0.005  # за единицу нанесённого урона (0.05 за 10)
REWARD_LOSE             = -2.0    # смерть игрока
REWARD_DAMAGE_TAKEN_MULT= -0.01   # за единицу полученного урона
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
