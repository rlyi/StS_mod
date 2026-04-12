"""train_meta.py — обучение Decision Tree на истории запусков StS.

Парсит JSON-файлы из папки runs/ (стандартные save-файлы игры),
обучает DecisionTreeClassifier предсказывать победу/поражение,
сохраняет модель в models/meta_tree.pkl и рисует дерево в PNG.

Запуск (не требует запущенной игры):
  python training/train_meta.py

Данные:
  C:\\Users\\User\\Documents\\My Games\\SlayTheSpire\\runs\\*.json
  (или data/runs/*.json — положите свои файлы туда)
"""

import os
import sys
import json
import glob
import pickle
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from config import STS_RUNS_DIR, DATA_RUNS_DIR, MODELS_DIR, CHARACTER

FEATURE_NAMES = [
    "hp_percent", "floor_norm", "act_norm", "gold_norm", "deck_size_norm",
    "attack_ratio", "defense_ratio", "curse_ratio", "relic_count_norm",
    "has_boss_relic",
]


# ── Парсинг ───────────────────────────────────────────────────────────

def parse_run_file(path: str):
    """Вернуть (features, label) из run_summary JSON, или None если файл нельзя использовать."""
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None

    # Фильтр: только Ironclad, только Акт 1
    char = str(data.get("character_chosen", "")).upper()
    if char != CHARACTER and char != "IRONCLAD":
        return None

    floor = int(data.get("floor_reached", 0))
    victory = bool(data.get("victory", False))

    # Метка: 1 = победа над боссом Акта 1, 0 = поражение
    label = 1 if victory else 0

    # Признаки финального состояния
    master_deck = data.get("master_deck", [])
    deck_size   = len(master_deck)

    # Тип карт — в master_deck либо строки, либо объекты
    attack  = 0
    defense = 0
    curse   = 0
    for card in master_deck:
        card_id = card if isinstance(card, str) else card.get("id", "")
        cl = _classify_card(card_id)
        if   cl == "ATTACK":  attack  += 1
        elif cl == "SKILL":   defense += 1
        elif cl == "CURSE":   curse   += 1

    relics_raw = data.get("relics", [])
    relic_count = len(relics_raw)
    has_boss = int(_has_boss_relic(relics_raw))

    gold = int(data.get("gold", 0))

    # hp_percent при смерти / финале (если нет данных — предполагаем 0 при смерти)
    hp_pct = float(data.get("hp_percent", 0.0 if not victory else 1.0))

    features = np.array([
        hp_pct,
        min(floor / 57.0, 1.0),
        min(data.get("act", 1) / 3.0, 1.0),
        min(gold   / 999.0,  1.0),
        min(deck_size / 30.0, 1.0),
        min(attack  / max(deck_size, 1), 1.0),
        min(defense / max(deck_size, 1), 1.0),
        min(curse   / max(deck_size, 1), 1.0),
        min(relic_count / 20.0, 1.0),
        float(has_boss),
    ], dtype=np.float32)

    return features, label


def load_dataset():
    """Загрузить все файлы из обеих папок с ранами."""
    X, y = [], []
    paths = (
        glob.glob(os.path.join(STS_RUNS_DIR, "*.json")) +
        glob.glob(os.path.join(DATA_RUNS_DIR, "*.json"))
    )
    for path in paths:
        result = parse_run_file(path)
        if result:
            feat, label = result
            X.append(feat)
            y.append(label)

    print(f"[train_meta] Загружено файлов: {len(paths)}, использовано: {len(X)}")
    return np.array(X), np.array(y)


# ── Обучение ──────────────────────────────────────────────────────────

def train(X: np.ndarray, y: np.ndarray) -> DecisionTreeClassifier:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = DecisionTreeClassifier(
        max_depth=5,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("\n[train_meta] Результаты на тесте:")
    print(classification_report(y_test, y_pred, target_names=["defeat", "victory"]))

    print("\n[train_meta] Правила дерева:")
    print(export_text(model, feature_names=FEATURE_NAMES))

    return model


def save_model(model: DecisionTreeClassifier):
    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, "meta_tree.pkl")
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"[train_meta] Модель сохранена: {path}")
    return path


def save_tree_image(model: DecisionTreeClassifier):
    """Сохранить визуализацию дерева в PNG (требует matplotlib + sklearn)."""
    try:
        import matplotlib.pyplot as plt
        from sklearn.tree import plot_tree

        fig, ax = plt.subplots(figsize=(20, 10))
        plot_tree(
            model,
            feature_names=FEATURE_NAMES,
            class_names=["defeat", "victory"],
            filled=True,
            ax=ax,
        )
        img_path = os.path.join(MODELS_DIR, "meta_tree.png")
        plt.savefig(img_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[train_meta] Дерево сохранено: {img_path}")
    except ImportError:
        print("[train_meta] matplotlib не установлен — PNG не создан")


def main():
    X, y = load_dataset()

    if len(X) < 10:
        print(
            "[train_meta] Недостаточно данных для обучения.\n"
            f"  Положите run_summary JSON файлы в:\n"
            f"  {STS_RUNS_DIR}\n  или\n  {DATA_RUNS_DIR}"
        )
        return

    model = train(X, y)
    save_model(model)
    save_tree_image(model)


if __name__ == "__main__":
    main()


# ── Вспомогательные ───────────────────────────────────────────────────

_ATTACK_KEYWORDS  = {"Strike", "Bash", "Cleave", "Anger", "Clash", "Clothesline",
                     "Carnage", "Headbutt", "Heavy", "Iron Wave", "Pommel",
                     "Sword", "Thunderclap", "Twin Strike", "Wild Strike",
                     "Blood", "Dropkick", "Hemokinesis", "Pummel", "Rampage",
                     "Searing", "Sever", "Uppercut", "Whirlwind", "Bludgeon",
                     "Fiend Fire", "Immolate", "Feed", "Reaper", "Choke"}

_CURSE_KEYWORDS   = {"Curse", "Wound", "Dazed", "Burn", "Void", "Shame",
                     "Regret", "Normality", "Pain", "Parasite", "Doubt",
                     "Decay", "Clumsy", "Injury", "Pride", "Slimed"}

_BOSS_RELICS_SET  = {
    "Burning Blood", "Ring of the Snake", "Cracked Core", "Pure Water",
    "Astrolabe", "Black Star", "Calling Bell", "Coffee Dripper",
    "Cursed Key", "Ectoplasm", "Empty Cage", "Fusion Hammer",
    "Pandora's Box", "Philosopher's Stone", "Runic Dome", "Runic Pyramid",
    "Sacred Bark", "Slaver's Collar", "Snecko Eye", "Sozu",
    "Tiny House", "Velvet Choker", "Violet Lotus",
}


def _classify_card(card_id: str) -> str:
    for kw in _ATTACK_KEYWORDS:
        if kw.lower() in card_id.lower():
            return "ATTACK"
    for kw in _CURSE_KEYWORDS:
        if kw.lower() in card_id.lower():
            return "CURSE"
    return "SKILL"


def _has_boss_relic(relics) -> bool:
    for r in relics:
        name = r if isinstance(r, str) else r.get("id", "")
        if name in _BOSS_RELICS_SET:
            return True
    return False
