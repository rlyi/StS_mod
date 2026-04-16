"""training/train_meta.py — обучение Decision Tree мета-агентов.

Датасет: data/cluster_the_spire.zip (~280k ранов всего, ~105k Ironclad)
Обучает два классификатора:
  meta_card_dt.pkl     — выбор карты-награды  (бинарный: взять / пропустить)
  meta_campfire_dt.pkl — действие у костра    (бинарный: REST / SMITH)

Запуск (без запущенной игры):
  python training/train_meta.py
"""

import os
import sys
import json
import zipfile
import pickle
import logging

import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODELS_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger("train_meta")

_ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_ZIP  = os.path.join(_ROOT, "data", "cluster_the_spire.zip")

IRONCLAD_START_DECK = ["Strike_R"] * 5 + ["Defend_R"] * 4 + ["Bash"]
IRONCLAD_BASE_HP    = 80   # для нормализации (стартовое HP)

# Имена признаков — порядок должен совпадать с meta_tree_agent.py
CARD_FEATURE_NAMES = [
    "hp_pct", "floor_norm", "gold_norm",
    "deck_size_norm", "attack_ratio", "skill_ratio", "power_ratio",
    "card_is_attack", "card_is_skill", "card_is_power",
]
CAMPFIRE_FEATURE_NAMES = [
    "hp_pct", "floor_norm", "gold_norm",
    "deck_size_norm", "attack_ratio", "skill_ratio", "power_ratio",
]

# Для событий — добавляем one-hot кодирование событий (топ-20 частых)
EVENT_FEATURE_NAMES = [
    "hp_pct", "floor_norm", "gold_norm",
    "deck_size_norm", "attack_ratio", "skill_ratio", "power_ratio",
]  # + one-hot для событий будет добавлено динамически

# Для реликтов босса — как с картами
BOSS_RELIC_FEATURE_NAMES = [
    "hp_pct", "floor_norm", "gold_norm",
    "deck_size_norm", "attack_ratio", "skill_ratio", "power_ratio",
]

# ── Классификация карт ────────────────────────────────────────────────
# Синхронизировано с agents/meta_tree_agent.py

_ATTACKS = {
    "Strike_R", "Bash", "Anger", "Body Slam", "Clash", "Cleave", "Clothesline",
    "Headbutt", "Heavy Blade", "Iron Wave", "Perfected Strike", "Pommel Strike",
    "Sword Boomerang", "Thunderclap", "Twin Strike", "Wild Strike",
    "Blood for Blood", "Carnage", "Dropkick", "Hemokinesis", "Pummel", "Rampage",
    "Reckless Charge", "Searing Blow", "Sever Soul", "Uppercut", "Whirlwind",
    "Bludgeon", "Choke", "Feed", "Fiend Fire", "Immolate", "Reaper",
}
_POWERS = {
    "Combust", "Dark Embrace", "Evolve", "Fire Breathing", "Inflame", "Metallicize",
    "Rage", "Rupture", "Barricade", "Berserk", "Brutality", "Champion", "Corruption",
    "Juggernaut", "Limit Break",
}


def _card_type(card_id: str) -> str:
    base = card_id.split("+")[0].strip()
    if base in _ATTACKS: return "ATTACK"
    if base in _POWERS:  return "POWER"
    return "SKILL"


def _deck_features(deck: list) -> list:
    """4 признака деки: [size_norm, attack_ratio, skill_ratio, power_ratio]."""
    n = max(len(deck), 1)
    attacks = sum(1 for c in deck if _card_type(c) == "ATTACK")
    powers  = sum(1 for c in deck if _card_type(c) == "POWER")
    skills  = n - attacks - powers
    return [min(n / 30.0, 1.0), attacks / n, skills / n, powers / n]


def _card_type_features(card_id: str) -> list:
    """3 признака карты: [is_attack, is_skill, is_power]."""
    ct = _card_type(card_id)
    return [
        1.0 if ct == "ATTACK" else 0.0,
        1.0 if ct == "SKILL"  else 0.0,
        1.0 if ct == "POWER"  else 0.0,
    ]


# ── Парсинг одного рана ───────────────────────────────────────────────

def parse_run(data: dict):
    """Извлечь обучающие примеры из одного рана.

    Возвращает (card_samples, campfire_samples, event_samples, boss_relic_samples) где каждый элемент —
    (np.ndarray признаков, int метка), или None если ран не подходит.
    """
    char = str(data.get("character_chosen", "")).upper()
    if "IRONCLAD" not in char:
        return None

    card_samples       = []
    campfire_samples   = []
    event_samples      = []
    boss_relic_samples = []

    hp_arr   = data.get("current_hp_per_floor", [])
    gold_arr = data.get("gold_per_floor", [])

    def _hp(floor: int) -> float:
        idx = floor - 1
        v = hp_arr[idx] if 0 <= idx < len(hp_arr) else IRONCLAD_BASE_HP
        return min(float(v) / IRONCLAD_BASE_HP, 2.0)

    def _gold(floor: int) -> float:
        idx = floor - 1
        v = gold_arr[idx] if 0 <= idx < len(gold_arr) else 0
        return min(float(v) / 999.0, 1.0)

    # Восстанавливаем деку по ходу рана (по порядку выборов карт)
    deck = list(IRONCLAD_START_DECK)

    for choice in sorted(data.get("card_choices", []), key=lambda c: c.get("floor", 0)):
        floor     = int(choice.get("floor", 1))
        picked    = choice.get("picked") or "SKIP"
        not_picked = choice.get("not_picked") or []

        state = [_hp(floor), min(floor / 17.0, 1.0), _gold(floor)] + _deck_features(deck)
        # state: 3 + 4 = 7 признаков

        all_offered = ([] if picked == "SKIP" else [picked]) + list(not_picked)
        for card_id in all_offered:
            if not card_id:
                continue
            base  = card_id.split("+")[0].strip()
            label = 1 if (card_id == picked and picked != "SKIP") else 0
            card_samples.append((state + _card_type_features(base), label))

        if picked and picked != "SKIP":
            deck.append(picked.split("+")[0].strip())

    for choice in data.get("campfire_choices", []):
        floor = int(choice.get("floor", 1))
        key   = str(choice.get("key", "REST")).upper()
        if key not in ("REST", "SMITH", "FORGE"):
            continue
        state = [_hp(floor), min(floor / 17.0, 1.0), _gold(floor)] + _deck_features(deck)
        label = 0 if key == "REST" else 1
        campfire_samples.append((state, label))

    # ── События ─────────────────────────────────────────────────────────
    for choice in data.get("event_choices", []):
        floor = int(choice.get("floor", 1))
        event_name = choice.get("event_name", "UNKNOWN")
        player_choice = choice.get("player_choice", "")
        # Метка: 1 если выбор "хороший" (нет урона, есть награда), 0 если плохой
        damage_taken = choice.get("damage_taken", 0)
        gold_gain = choice.get("gold_gain", 0)
        relics_obtained = choice.get("relics_obtained", [])
        cards_obtained = choice.get("cards_obtained", [])
        # "Хороший" выбор: нет урона И (есть золото ИЛИ реликт ИЛИ карта)
        is_good = (damage_taken == 0) and (gold_gain > 0 or relics_obtained or cards_obtained)
        label = 1 if is_good else 0
        state = [_hp(floor), min(floor / 17.0, 1.0), _gold(floor)] + _deck_features(deck)
        event_samples.append((state + [hash(event_name) % 100 / 100.0], label))  # простой hash encoding

    # ── Реликты босса ──────────────────────────────────────────────────
    for relic_choice in data.get("boss_relics", []):
        picked = relic_choice.get("picked")
        not_picked = relic_choice.get("not_picked", [])
        # Используем floor босса (примерно 17 или 34)
        floor = 17  # Act 1 boss по умолчанию
        if len(deck) > 20:  # примерная эвристика для Act 2
            floor = 34
        state_base = [_hp(floor), min(floor / 17.0, 1.0), _gold(floor)] + _deck_features(deck)
        all_offered = ([picked] if picked else []) + list(not_picked)
        for relic_id in all_offered:
            if not relic_id:
                continue
            label = 1 if (relic_id == picked) else 0
            # Для реликтов используем hash как простой признак
            boss_relic_samples.append((state_base + [hash(relic_id) % 100 / 100.0], label))

    return card_samples, campfire_samples, event_samples, boss_relic_samples


# ── Загрузка датасета ─────────────────────────────────────────────────

def load_dataset():
    log.info("Датасет: %s", DATASET_ZIP)
    card_X, card_y               = [], []
    campfire_X, campfire_y       = [], []
    event_X, event_y             = [], []
    boss_relic_X, boss_relic_y   = [], []
    processed = skipped = 0

    with zipfile.ZipFile(DATASET_ZIP) as zf:
        names = [n for n in zf.namelist() if n.endswith(".json")]
        log.info("JSON файлов: %d", len(names))

        for i, name in enumerate(names):
            if i % 25_000 == 0 and i > 0:
                log.info("  %d / %d  карт=%d  кост=%d  событ=%d  реликт=%d",
                         i, len(names), len(card_X), len(campfire_X), len(event_X), len(boss_relic_X))
            try:
                with zf.open(name) as f:
                    data = json.load(f)
            except Exception:
                skipped += 1
                continue

            if not isinstance(data, dict):
                skipped += 1
                continue

            result = parse_run(data)
            if result is None:
                skipped += 1
                continue

            cs, fs, es, brs = result
            for feats, label in cs:
                card_X.append(feats)
                card_y.append(label)
            for feats, label in fs:
                campfire_X.append(feats)
                campfire_y.append(label)
            for feats, label in es:
                event_X.append(feats)
                event_y.append(label)
            for feats, label in brs:
                boss_relic_X.append(feats)
                boss_relic_y.append(label)
            processed += 1

    log.info("Ранов обработано: %d, пропущено: %d", processed, skipped)
    log.info("Примеров: карт=%d  кост=%d  событ=%d  реликт=%d",
             len(card_X), len(campfire_X), len(event_X), len(boss_relic_X))
    return (
        (np.array(card_X, dtype=np.float32), np.array(card_y)),
        (np.array(campfire_X, dtype=np.float32), np.array(campfire_y)),
        (np.array(event_X, dtype=np.float32), np.array(event_y)),
        (np.array(boss_relic_X, dtype=np.float32), np.array(boss_relic_y)),
    )


# ── Обучение ─────────────────────────────────────────────────────────

def train_model(X, y, name: str, feature_names: list, max_depth: int = 6):
    if len(X) < 50:
        log.warning("Слишком мало данных для %s (%d примеров) — пропуск", name, len(X))
        return None

    unique = np.unique(y)
    if len(unique) < 2:
        log.warning("%s: только один класс (%s) — пропуск", name, unique)
        return None

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=50,
        class_weight="balanced",
        random_state=42,
    )
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)

    print(f"\n{'='*50}")
    print(f"  {name}  (train={len(X_tr)}  test={len(X_te)})")
    print('='*50)
    print(classification_report(y_te, y_pred))
    print(export_text(model, feature_names=feature_names))

    return model


def save_model(model, filename: str):
    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, filename)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    log.info("Сохранено: %s", path)
    return path


# ── Точка входа ───────────────────────────────────────────────────────

def main():
    (card_X, card_y), (campfire_X, campfire_y), (event_X, event_y), (boss_relic_X, boss_relic_y) = load_dataset()

    model = train_model(card_X, card_y, "CardChoice (взять=1 / пропустить=0)",
                        CARD_FEATURE_NAMES)
    if model:
        save_model(model, "meta_card_dt.pkl")

    model = train_model(campfire_X, campfire_y, "CampfireChoice (REST=0 / SMITH=1)",
                        CAMPFIRE_FEATURE_NAMES)
    if model:
        save_model(model, "meta_campfire_dt.pkl")

    model = train_model(event_X, event_y, "EventChoice (хороший=1 / плохой=0)",
                        EVENT_FEATURE_NAMES + ["event_hash"])
    if model:
        save_model(model, "meta_event_dt.pkl")

    model = train_model(boss_relic_X, boss_relic_y, "BossRelicChoice (взять=1 / пропустить=0)",
                        BOSS_RELIC_FEATURE_NAMES + ["relic_hash"])
    if model:
        save_model(model, "meta_boss_relic_dt.pkl")


if __name__ == "__main__":
    main()
