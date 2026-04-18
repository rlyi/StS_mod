"""training/train_meta_slaythedata.py — обучение DT из SlayTheData.

Читает JSON файлы напрямую, фильтрует Ironclad A0-A5 (не 2018),
обучает 5 моделей без сохранения промежуточных данных.

Признаки:
  card     — [hp_pct, floor/17, gold_norm, is_attack, is_skill, is_power, is_upgraded, rarity]  (8)
  campfire — [hp_pct, floor/17, gold_norm, is_pre_boss, deck_size_proxy]                         (5)
  path     — [hp_pct, floor/17, gold_norm]                                                        (3)
  event    — [hp_pct, floor/17, gold_norm, event_hash]                                            (4)
  shop     — [hp_pct, floor/17, gold_norm]                                                        (3)
"""
import os
import sys
import json
from pathlib import Path
import pickle

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODELS_DIR
from agents.meta_tree_agent import card_features

def log(msg):
    print(msg, flush=True)

DATA_DIR = r'c:\StS_mod\data\slay-the-data.-7z\SlayTheData'

MAX_CARD_SAMPLES  = 200_000
MAX_EVENT_SAMPLES = 100_000

# Накопители
X_cards,    y_cards,    w_cards    = [], [], []
X_campfire, y_campfire, w_campfire = [], [], []
X_path,     y_path,     w_path     = [], [], []
X_event,    y_event,    w_event    = [], [], []
X_shop,     y_shop,     w_shop     = [], [], []

VICTORY_WEIGHT = 3.0

stats = {'files_processed': 0, 'total_runs': 0, 'ironclad_filtered': 0, 'victories': 0}

# Слова, указывающие на "пропустить" событие
_SKIP_WORDS = {'leave', 'skip', 'refuse', 'ignore', 'walk away', 'turn it down', 'i must be going'}

def _is_skip(choice_text: str) -> bool:
    lower = choice_text.lower()
    return any(kw in lower for kw in _SKIP_WORDS)


def parse_run(run):
    if 'event' in run:
        r = run['event']
    else:
        r = run
    if r.get('character_chosen') != 'IRONCLAD':
        return None
    if r.get('ascension_level', 0) > 5:
        return None
    if r.get('is_daily') or r.get('is_trial'):
        return None
    local_time = r.get('local_time', '20190101')
    if local_time.startswith('2018'):
        return None
    return r


def extract_ctx(r, floor):
    """Базовый контекст: [hp_pct, floor/17, gold_norm]."""
    hp_per_floor  = r.get('current_hp_per_floor', [])
    max_hp_floors = r.get('max_hp_per_floor', [])
    max_hp        = max_hp_floors[0] if max_hp_floors else 80
    gold_per_floor = r.get('gold_per_floor', [])

    hp   = hp_per_floor[floor - 1]  if floor - 1 < len(hp_per_floor)   else max_hp
    gold = gold_per_floor[floor - 1] if floor - 1 < len(gold_per_floor) else 0

    return [
        hp / max_hp if max_hp > 0 else 1.0,
        min(floor / 17.0, 1.0),
        min(gold / 500.0, 1.0),
    ]


def process_run(r):
    global stats

    is_victory = r.get('victory', False)
    if is_victory:
        stats['victories'] += 1
    w = VICTORY_WEIGHT if is_victory else 1.0

    # Размер итоговой колоды как прокси для deck_size во время рана
    deck_size_proxy = min(len(r.get('master_deck', [])) / 30.0, 1.0)

    # ── 1. CARD CHOICES ──────────────────────────────────────────────────
    # Одна строка на каждую предложенную карту.
    # Label: 1 = взяли эту карту, 0 = нет.
    for cc in r.get('card_choices', []):
        if len(X_cards) >= MAX_CARD_SAMPLES:
            break
        floor   = int(cc.get('floor', 1))
        picked  = cc.get('picked', '')
        not_picked = cc.get('not_picked', [])
        ctx = extract_ctx(r, floor)

        for card_id in not_picked:
            if len(X_cards) >= MAX_CARD_SAMPLES:
                break
            X_cards.append(ctx + card_features(card_id))
            y_cards.append(0)
            w_cards.append(w)

        if picked and picked != 'SKIP':
            X_cards.append(ctx + card_features(picked))
            y_cards.append(1)
            w_cards.append(w)

    # ── 2. CAMPFIRE ──────────────────────────────────────────────────────
    # Фичи: [hp_pct, floor/17, gold_norm, is_pre_boss, deck_size_proxy]
    # Label: 0 = REST, 1 = SMITH
    for cf in r.get('campfire_choices', []):
        key   = cf.get('key', '')
        floor = int(cf.get('floor', 1))
        if key not in ('REST', 'SMITH'):
            continue
        ctx = extract_ctx(r, floor)
        is_pre_boss = 1.0 if floor >= 14 else 0.0
        X_campfire.append(ctx + [is_pre_boss, deck_size_proxy])
        y_campfire.append(0 if key == 'REST' else 1)
        w_campfire.append(w)

    # ── 3. PATH ──────────────────────────────────────────────────────────
    # Фичи: [hp_pct, floor/17, gold_norm]
    # Label: 0=M, 1=?, 2=E, 3=R, 4=$, 5=T
    node_map = {'M': 0, '?': 1, 'E': 2, 'R': 3, '$': 4, 'T': 5}
    for floor_idx, node_type in enumerate(r.get('path_per_floor', [])):
        if node_type in node_map:
            floor = floor_idx + 1
            X_path.append(extract_ctx(r, floor))
            y_path.append(node_map[node_type])
            w_path.append(w)

    # ── 4. EVENT ─────────────────────────────────────────────────────────
    # Фичи: [hp_pct, floor/17, gold_norm, event_hash]
    # Label: 0 = игрок пропустил/ушёл, 1 = игрок взаимодействовал
    for ev in r.get('event_choices', []):
        if len(X_event) >= MAX_EVENT_SAMPLES:
            break
        floor      = int(ev.get('floor', 1))
        choice     = str(ev.get('player_choice', ''))
        event_name = str(ev.get('event_name', ''))
        ctx        = extract_ctx(r, floor)
        event_hash = hash(event_name) % 997 / 997.0
        X_event.append(ctx + [event_hash])
        y_event.append(0 if _is_skip(choice) else 1)
        w_event.append(w)

    # ── 5. SHOP ──────────────────────────────────────────────────────────
    # Label: 0 = карта, 1 = реликвия
    items_purchased = r.get('items_purchased', [])
    purchase_floors = r.get('item_purchase_floors', [])
    for idx, item in enumerate(items_purchased):
        floor    = int(purchase_floors[idx]) if idx < len(purchase_floors) else 1
        is_relic = any(c.isupper() for c in item[1:]) if len(item) > 1 else False
        X_shop.append(extract_ctx(r, floor))
        y_shop.append(1 if is_relic else 0)
        w_shop.append(w)


# ── Обработка файлов ──────────────────────────────────────────────────────
json_files = list(Path(DATA_DIR).glob('*.json'))
log(f"Найдено файлов: {len(json_files)}")

for i, filepath in enumerate(json_files):
    if i % 100 == 0:
        log(f"Обработано: {i}/{len(json_files)} | "
            f"cards={len(X_cards)} event={len(X_event)} runs={stats['ironclad_filtered']}")

    try:
        with open(filepath, encoding='utf-8') as f:
            data = json.load(f)
        if not isinstance(data, list):
            continue

        stats['total_runs'] += len(data)
        for run_wrap in data:
            r = parse_run(run_wrap)
            if r:
                stats['ironclad_filtered'] += 1
                process_run(r)

        stats['files_processed'] += 1

        if len(X_cards) >= MAX_CARD_SAMPLES and len(X_event) >= MAX_EVENT_SAMPLES:
            log("Достигнуты оба лимита (card + event) — остановка.")
            break

    except Exception as e:
        print(f"Ошибка в {filepath}: {e}")
        continue

log("=" * 60)
log(f"Файлов: {stats['files_processed']}  ранов: {stats['ironclad_filtered']}  побед: {stats['victories']}")

# ── Обучение моделей ──────────────────────────────────────────────────────
models_to_train = [
    ('card',     X_cards,    y_cards,    w_cards),
    ('campfire', X_campfire, y_campfire, w_campfire),
    ('path',     X_path,     y_path,     w_path),
    ('event',    X_event,    y_event,    w_event),
    ('shop',     X_shop,     y_shop,     w_shop),
]

for name, X, y, w in models_to_train:
    if len(X) < 100:
        log(f"{name}: недостаточно данных ({len(X)} записей) — пропуск")
        continue

    X_arr = np.array(X, dtype=np.float32)
    y_arr = np.array(y)
    w_arr = np.array(w, dtype=np.float32)

    X_train, X_test, y_train, y_test, w_train, _ = train_test_split(
        X_arr, y_arr, w_arr, test_size=0.2, random_state=42
    )

    clf = DecisionTreeClassifier(max_depth=10, min_samples_leaf=50, random_state=42)
    clf.fit(X_train, y_train, sample_weight=w_train)

    acc = clf.score(X_test, y_test)
    log(f"{name}: {len(X)} записей (pobedy x{VICTORY_WEIGHT}), accuracy={acc:.3f}")

    path = os.path.join(MODELS_DIR, f'meta_{name}_slaythedata_dt.pkl')
    with open(path, 'wb') as f:
        pickle.dump(clf, f)
    log(f"  -> {path}")

log("=" * 60)
log("ГОТОВО!")
