import os
import pickle
import random
import logging
import numpy as np

from agents.base_agent import BaseMetaAgent
from config import MODELS_DIR

CARD_MODEL_PATH     = os.path.join(MODELS_DIR, "meta_card_slaythedata_dt.pkl")
CAMPFIRE_MODEL_PATH = os.path.join(MODELS_DIR, "meta_campfire_slaythedata_dt.pkl")
PATH_MODEL_PATH     = os.path.join(MODELS_DIR, "meta_path_slaythedata_dt.pkl")
EVENT_MODEL_PATH    = os.path.join(MODELS_DIR, "meta_event_slaythedata_dt.pkl")
SHOP_MODEL_PATH     = os.path.join(MODELS_DIR, "meta_shop_slaythedata_dt.pkl")

_SKIP_WORDS = {'leave', 'skip', 'refuse', 'ignore', 'walk away', 'turn it down', 'i must be going'}

# Карты в порядке приоритета для апгрейда (SMITH) и дублирования (Duplicator)
_UPGRADE_PRIORITY = [
    'Bash', 'Whirlwind', 'Reaper', 'Feed', 'Immolate', 'Fiend Fire',
    'Limit Break', 'Barricade', 'Corruption', 'Berserk',
    'Dropkick', 'Shrug It Off', 'Sentinel', 'Second Wind',
]
# Карты которые нужно удалять первыми (for_purge)
_REMOVE_PRIORITY = [
    'AscendersBane', 'Clumsy', 'Decay', 'Doubt', 'Injury',
    'Normalcy', 'Pain', 'Parasite', 'Pride', 'Regret', 'Shame', 'Writhe',
    'Strike_R', 'Defend_R',
]


def _event_choice(event_name: str, hp_pct: float, gold: int, n_options: int) -> int | None:
    """Жёсткие правила для событий Акт 1 Ironclad.

    Стратегия: стабильность — избегать проклятий, потери max HP, ловушек.
    Возвращает индекс выбора или None если событие неизвестно.
    """
    n = event_name.lower()

    if 'big fish' in n:
        return 0                          # +Max HP — безопасно и хорошо

    if 'cleric' in n:
        if gold >= 50:   return 1         # Удалить карту — лучший вариант
        if gold >= 35 and hp_pct < 0.7:
                         return 0         # Лечение если мало HP
        return n_options - 1              # Уйти

    if 'duplicator' in n:
        return 0                          # Всегда дублируем карту

    if 'forgotten altar' in n:
        return n_options - 1              # Уйти — не терять max HP

    if 'golden idol' in n:
        return n_options - 1              # Leave — ловушка может дать проклятие

    if 'hypnotizing' in n or ('mushroom' in n and 'colored' in n):
        # 0=Curse-Injury, 1=Take 20 dmg, 2=Lose 6 max HP
        if hp_pct > 0.35: return 1        # Smash — предсказуемый урон
        return 2                          # Hide — лучше потерять max HP чем умереть

    if 'knowing skull' in n:
        return n_options - 1              # Уйти — не тратить HP

    if 'masked bandits' in n:
        return 1                          # Бой — не терять всё золото

    if 'mysterious sphere' in n:
        return n_options - 1              # Уйти — тяжёлый бой в Акт 1

    if 'note for yourself' in n:
        return 0                          # Взять карту

    if 'old beggar' in n:
        return 0                          # Удалить карту — очень ценно

    if 'old chest' in n:
        return 0                          # Открыть сундук

    if 'scrap ooze' in n:
        return n_options - 1              # Уйти — непредсказуемый урон

    if 'shining light' in n:
        if hp_pct > 0.5: return 0         # Апгрейд 2 карт стоит урона
        return n_options - 1              # Уйти если мало HP

    if 'shrine' in n and 'blood' in n:
        return 0                          # Обычно даёт HP

    if 'winding halls' in n:
        return n_options - 1              # Уйти — оба варианта плохие

    if 'world of goop' in n:
        return n_options - 1              # Уйти — не менять HP на золото

    if 'dead adventurer' in n:
        return n_options - 1              # Уйти — лишний бой не нужен

    if 'liars game' in n:
        return 0                          # Первый вариант

    if 'wheel of change' in n:
        return 0                          # Нет выбора

    return None                           # Неизвестное событие — используем модель

# ── Классификация карт ─────────────────────────────────────────────────
# Используется и при обучении (train_meta.py), и при инференсе.

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
_RARES = {
    "Bludgeon", "Feed", "Fiend Fire", "Immolate", "Reaper",
    "Impervious", "Offering",
    "Barricade", "Berserk", "Brutality", "Champion", "Corruption", "Juggernaut", "Limit Break",
}
_UNCOMMONS = {
    "Blood for Blood", "Carnage", "Dropkick", "Hemokinesis", "Pummel", "Rampage",
    "Reckless Charge", "Searing Blow", "Sever Soul", "Uppercut", "Whirlwind",
    "Battle Trance", "Bloodletting", "Burning Pact", "Disarm", "Dual Wield", "Entrench",
    "Exhume", "Flame Barrier", "Infernal Blade", "Intimidate", "Power Through",
    "Seeing Red", "Second Wind", "Sentinel", "Shockwave", "Spot Weakness",
    "Combust", "Dark Embrace", "Evolve", "Fire Breathing", "Inflame", "Metallicize",
    "Rage", "Rupture",
}


def card_type(card_id: str) -> str:
    """Классифицирует карту Ironclad: ATTACK / POWER / SKILL."""
    base = card_id.split("+")[0].strip()
    if base in _ATTACKS: return "ATTACK"
    if base in _POWERS:  return "POWER"
    return "SKILL"


def _card_feats_from_obj(card, ctx: list) -> np.ndarray:
    """Фичи для инференса из объекта Card spirecomm (использует реальные type/rarity)."""
    from spirecomm.spire.card import CardType, CardRarity
    ct = card.type
    is_upgraded = 1.0 if card.upgrades > 0 else 0.0
    rarity = card.rarity
    rarity_norm = 1.0 if rarity == CardRarity.RARE else (0.5 if rarity == CardRarity.UNCOMMON else 0.0)
    feats = [
        1.0 if ct == CardType.ATTACK else 0.0,
        1.0 if ct == CardType.SKILL  else 0.0,
        1.0 if ct == CardType.POWER  else 0.0,
        is_upgraded,
        rarity_norm,
    ]
    return np.array(ctx + feats, dtype=np.float32).reshape(1, -1)


def card_features(card_id: str) -> list:
    """5 признаков карты: [is_attack, is_skill, is_power, is_upgraded, rarity_norm].

    Используется и при обучении (train_meta_slaythedata.py), и при инференсе.
    Итоговый вектор для card модели: [hp_pct, floor_norm, gold_norm] + card_features = 8 признаков.
    """
    base = card_id.split("+")[0].strip()
    ct = card_type(base)
    is_upgraded = 1.0 if "+" in card_id else 0.0
    if base in _RARES:
        rarity = 1.0
    elif base in _UNCOMMONS:
        rarity = 0.5
    else:
        rarity = 0.0
    return [
        1.0 if ct == "ATTACK" else 0.0,
        1.0 if ct == "SKILL"  else 0.0,
        1.0 if ct == "POWER"  else 0.0,
        is_upgraded,
        rarity,
    ]


# ── Агент ─────────────────────────────────────────────────────────────

class DecisionTreeMetaAgent(BaseMetaAgent):
    """Decision Tree мета-агент, обученный на датасете SlayTheData.

    Модели (train_meta_slaythedata.py):
      meta_card_slaythedata_dt.pkl     — выбор карты-награды    (8 признаков)
      meta_campfire_slaythedata_dt.pkl — REST vs SMITH у костра (5 признаков)
      meta_path_slaythedata_dt.pkl     — тип узла на карте      (3 признака)
      meta_event_slaythedata_dt.pkl    — engage vs skip событие (4 признака)
      meta_shop_slaythedata_dt.pkl     — карта vs реликвия      (3 признака)
    """

    def __init__(self):
        self.card_model     = self._load(CARD_MODEL_PATH,     "card")
        self.campfire_model = self._load(CAMPFIRE_MODEL_PATH, "campfire")
        self.path_model     = self._load(PATH_MODEL_PATH,     "path")
        self.event_model    = self._load(EVENT_MODEL_PATH,    "event")
        self.shop_model     = self._load(SHOP_MODEL_PATH,     "shop")

    def _load(self, path: str, name: str):
        log = logging.getLogger("DecisionTreeMetaAgent")
        if os.path.exists(path):
            with open(path, "rb") as f:
                model = pickle.load(f)
            log.info("%s модель загружена: %s", name, path)
            return model
        log.warning("%s модель не найдена: %s", name, path)
        return None

    # ── Стратегические решения ────────────────────────────────────────

    def choose_card(self, game) -> int:
        cards = getattr(getattr(game, "screen", None), "cards", [])
        if not cards:
            return -1

        if self.card_model is None:
            return random.randrange(len(cards))

        hp_pct    = game.current_hp / max(game.max_hp, 1) if game.max_hp else 1.0
        floor     = getattr(game, "floor", 1)
        gold_norm = min(getattr(game, "gold", 0) / 500.0, 1.0)
        ctx = [hp_pct, min(floor / 17.0, 1.0), gold_norm]
        log = logging.getLogger("DecisionTreeMetaAgent")

        best_idx  = -1
        best_prob = -1.0

        for i, card in enumerate(cards[:3]):
            feats = _card_feats_from_obj(card, ctx)
            try:
                prob = float(self.card_model.predict_proba(feats)[0][1])
            except Exception as e:
                log.warning("card predict_proba error: %s", e)
                prob = 0.0
            log.debug("  card[%d] %s prob=%.3f", i, getattr(card, "card_id", "?"), prob)
            if prob > best_prob:
                best_prob = prob
                best_idx  = i

        # Пропускаем только если все карты хуже 0.05 (курсы, трэш)
        if best_prob < 0.05:
            log.info("choose_card: все карты < 0.05 — SKIP")
            return -1
        return best_idx

    def choose_path(self, game) -> int:
        nodes = getattr(getattr(game, "screen", None), "next_nodes", [])
        if not nodes:
            return 0

        hp_pct    = game.current_hp / max(game.max_hp, 1) if game.max_hp else 1.0
        floor     = getattr(game, "floor", 1)
        gold_norm = min(getattr(game, "gold", 0) / 500.0, 1.0)
        feats = np.array([hp_pct, min(floor / 17.0, 1.0), gold_norm], dtype=np.float32).reshape(1, -1)

        node_type_map = {0: "M", 1: "?", 2: "E", 3: "R", 4: "$", 5: "T"}
        if self.path_model is not None:
            try:
                preferred = node_type_map.get(int(self.path_model.predict(feats)[0]), "M")
                matching = [i for i, node in enumerate(nodes)
                            if str(getattr(node, "symbol", "")).upper() == preferred]
                if matching:
                    return random.choice(matching)
            except Exception:
                pass

        return random.randrange(len(nodes))

    def choose_shop(self, game) -> int:
        """Выбор в магазине. Возвращает индекс товара или -1 = выйти.

        Индексы: [карты | реликвии | зелья | purge]
        Специальный возврат: -2 = purge (удалить карту).

        Приоритет: purge > зелье > карта > выйти.
        Реликвии в Акт 1 обычно недоступны по цене — пропускаем.
        """
        s       = getattr(game, "screen", None)
        gold    = getattr(game, "gold", 0)
        cards   = getattr(s, "cards",   [])
        relics  = getattr(s, "relics",  [])
        potions = getattr(s, "potions", [])

        # 1. Зелье — если доступно и недорогое (< 100 золота)
        # TODO: purge (удаление карты) — высший приоритет, но нужно верифицировать индекс действия
        potion_offset = len(cards) + len(relics)
        for i, potion in enumerate(potions):
            price = getattr(potion, "price", 9999)
            if price <= gold and price < 100:
                return potion_offset + i

        # 3. Карта — если DT рекомендует и цена подъёмная
        if cards and self.shop_model is not None:
            hp_pct    = game.current_hp / max(game.max_hp, 1) if game.max_hp else 1.0
            floor     = getattr(game, "floor", 1)
            gold_norm = min(gold / 500.0, 1.0)
            feats = np.array(
                [hp_pct, min(floor / 17.0, 1.0), gold_norm], dtype=np.float32
            ).reshape(1, -1)
            try:
                pred = int(self.shop_model.predict(feats)[0])
                if pred == 0:
                    for i, card in enumerate(cards):
                        if getattr(card, "price", 9999) <= gold:
                            return i
            except Exception:
                pass

        return -1  # выйти

    def choose_rest(self, game) -> str:
        hp_pct = game.current_hp / max(game.max_hp, 1) if game.max_hp else 1.0
        floor  = getattr(game, "floor", 1)
        gold_norm = min(getattr(game, "gold", 0) / 500.0, 1.0)
        deck      = getattr(game, "deck", [])
        deck_size_proxy = min(len(deck) / 30.0, 1.0)
        is_pre_boss     = 1.0 if floor >= 14 else 0.0

        if self.campfire_model is None:
            return "REST" if hp_pct < 0.6 else "SMITH"

        feats = np.array(
            [hp_pct, min(floor / 17.0, 1.0), gold_norm, is_pre_boss, deck_size_proxy],
            dtype=np.float32
        ).reshape(1, -1)
        try:
            pred = int(self.campfire_model.predict(feats)[0])
        except Exception:
            return "REST"
        return "REST" if pred == 0 else "SMITH"

    def choose_event(self, game) -> int:
        s       = getattr(game, "screen", None)
        options = getattr(s, "options", [])
        if not options:
            return 0

        event_name = getattr(s, "event_name", "")
        hp_pct     = game.current_hp / max(game.max_hp, 1) if game.max_hp else 1.0
        gold       = getattr(game, "gold", 0)

        # Жёсткие правила для известных событий
        idx = _event_choice(event_name, hp_pct, gold, len(options))
        if idx is not None:
            return max(0, min(idx, len(options) - 1))

        # Fallback: DT модель для неизвестных событий
        if self.event_model is None:
            return 0

        gold_norm  = min(gold / 500.0, 1.0)
        floor      = getattr(game, "floor", 1)
        event_hash = hash(event_name) % 997 / 997.0
        feats = np.array(
            [hp_pct, min(floor / 17.0, 1.0), gold_norm, event_hash],
            dtype=np.float32
        ).reshape(1, -1)
        try:
            pred = int(self.event_model.predict(feats)[0])
        except Exception:
            return 0
        if pred == 0:
            return len(options) - 1  # skip — последняя опция
        return 0                      # engage — первая опция

    def choose_hand(self, game) -> int:
        """Выбор карты из руки (HAND_SELECT).

        Armaments: апгрейд лучшей карты в руке.
        Dual Wield: дублирование лучшей атаки/пауэра в руке.
        В обоих случаях логика одинакова — берём лучшую карту.
        """
        s     = getattr(game, "screen", None)
        cards = getattr(s, "cards", [])
        if not cards:
            return 0

        def cid(c):
            return getattr(c, "card_id", str(c)).split("+")[0].strip()

        # Приоритет из _UPGRADE_PRIORITY — подходит и для апгрейда и для дублирования
        for priority_id in _UPGRADE_PRIORITY:
            for i, c in enumerate(cards):
                if cid(c) == priority_id:
                    return i

        # Fallback: редкая > необычная > обычная атака
        for i, c in enumerate(cards):
            if cid(c) in _RARES:
                return i
        for i, c in enumerate(cards):
            if cid(c) in _UNCOMMONS and card_type(cid(c)) == "ATTACK":
                return i
        for i, c in enumerate(cards):
            if card_type(cid(c)) == "ATTACK":
                return i
        return 0

    def choose_boss_relic(self, game) -> int:
        relics = getattr(getattr(game, "screen", None), "relics", [])
        return 0 if not relics else random.randrange(len(relics))

    def choose_grid(self, game, for_upgrade: bool = False) -> int:
        """Эвристика выбора карты в GRID.

        Контекст определяется флагами экрана:
          for_upgrade → SMITH: выбираем лучшую для апгрейда
          for_purge   → удаление: выбираем худшую карту
          иначе       → дублирование (Duplicator): выбираем лучшую
        """
        s     = getattr(game, "screen", None)
        cards = getattr(s, "cards", [])
        if not cards:
            return 0

        is_upgrade = for_upgrade or getattr(s, "for_upgrade", False)
        is_purge   = getattr(s, "for_purge", False)

        def cid(c):
            return getattr(c, "card_id", str(c)).split("+")[0].strip()

        if is_purge:
            # Удаляем худшую карту: проклятия → базовые Strike/Defend
            for remove_id in _REMOVE_PRIORITY:
                for i, c in enumerate(cards):
                    if cid(c) == remove_id:
                        return i
            # Fallback: удаляем карту с наименьшей редкостью
            for i, c in enumerate(cards):
                if card_type(cid(c)) == "SKILL" and cid(c) not in _UNCOMMONS and cid(c) not in _RARES:
                    return i
            return 0

        else:
            # Апгрейд или дублирование — выбираем лучшую карту
            for priority_id in _UPGRADE_PRIORITY:
                for i, c in enumerate(cards):
                    is_already_upgraded = "+" in getattr(c, "card_id", "")
                    if cid(c) == priority_id and (not is_upgrade or not is_already_upgraded):
                        return i
            # Fallback: любая не-апгрейднутая атака
            for i, c in enumerate(cards):
                if card_type(cid(c)) == "ATTACK" and "+" not in getattr(c, "card_id", ""):
                    return i
            return 0
