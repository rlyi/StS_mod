import os
import pickle
import random
import logging
import numpy as np
from spirecomm.communication.action import ProceedAction, ChooseAction, StateAction

from agents.base_agent import BaseMetaAgent
from config import MODELS_DIR


# ── Признаки для Decision Tree ────────────────────────────────────────

def game_to_features(game) -> np.ndarray:
    """Преобразует состояние игры в вектор признаков (10 float32).

    Используется Decision Tree и для логирования.
    """
    player = game.player
    deck   = getattr(game, "deck",   [])
    relics = getattr(game, "relics", [])

    hp_percent   = player.current_hp / max(player.max_hp, 1)
    floor_number = getattr(game, "floor", 0)
    act_number   = getattr(game, "act",   1)
    gold         = getattr(game, "gold",  0)
    deck_size    = len(deck)

    attack_cards  = 0
    defense_cards = 0
    curse_cards   = 0
    for card in deck:
        t = _card_type_str(card)
        if   t == "ATTACK": attack_cards  += 1
        elif t == "SKILL":  defense_cards += 1
        elif t in ("CURSE", "STATUS"): curse_cards += 1

    relic_count   = len(relics)
    has_boss_relic = int(any(_is_boss_relic(r) for r in relics))

    return np.array([
        hp_percent,
        floor_number / 57.0,
        act_number   / 3.0,
        min(gold     / 999.0, 1.0),
        min(deck_size / 30.0, 1.0),
        min(attack_cards  / 20.0, 1.0),
        min(defense_cards / 20.0, 1.0),
        min(curse_cards   / 10.0, 1.0),
        min(relic_count   / 20.0, 1.0),
        float(has_boss_relic),
    ], dtype=np.float32)


# ── Агент ─────────────────────────────────────────────────────────────

class MetaAgent(BaseMetaAgent):
    """Высокоуровневый агент для карты / магазина / событий.

    Этап 4: использует if/else правила.
    Этап 5: если есть models/meta_tree.pkl, применяет Decision Tree.
    """

    def __init__(self):
        self.tree = None
        model_path = os.path.join(MODELS_DIR, "meta_tree.pkl")
        if os.path.exists(model_path):
            with open(model_path, "rb") as f:
                self.tree = pickle.load(f)
            logging.getLogger("MetaAgent").info("Decision Tree загружен из %s", model_path)
        else:
            logging.getLogger("MetaAgent").info("Tree не найден — используются if/else правила")

    # ── Реализация абстрактных методов ────────────────────────────────

    def choose_path(self, game) -> int:
        """Предпочитаем элиты, иначе первый вариант."""
        nodes = getattr(getattr(game, "screen", None), "next_nodes", None) or []
        for i, node in enumerate(nodes):
            if str(getattr(node, "symbol", "")).upper() == "E":
                return i
        return 0

    def choose_card(self, game) -> int:
        """Берём первую карту, если колода не переполнена."""
        cards = getattr(getattr(game, "screen", None), "cards", [])
        if not cards:
            return -1

        hp_pct    = game.player.current_hp / max(game.player.max_hp, 1)
        deck_size = len(getattr(game, "deck", []))

        # Пропустить если колода уже большая и HP в норме
        if deck_size >= 20 and hp_pct > 0.5:
            return -1

        # Decision Tree: можно предсказать, какая карта лучше по признакам
        if self.tree is not None:
            features = game_to_features(game).reshape(1, -1)
            predicted = int(self.tree.predict(features)[0])
            # predicted = 1 → победная колода, берём; 0 → не берём
            if predicted == 0 and hp_pct > 0.4:
                return -1

        return 0

    def choose_shop(self, game) -> int:
        return -1  # покупки пока не делаем

    def choose_event(self, game) -> int:
        return 0  # всегда первый вариант

    # ── Главный диспетчер ─────────────────────────────────────────────

    def act(self, game):
        screen = _screen_str(game)
        s = getattr(game, "screen", None)

        if screen == "MAP":
            nodes = getattr(s, "next_nodes", [])
            return ChooseAction(random.randrange(len(nodes)) if nodes else 0)

        elif screen == "CARD_REWARD":
            cards = getattr(s, "cards", [])
            if not cards:
                return ProceedAction()
            # 80% берём случайную карту, 20% пропускаем
            if random.random() < 0.8:
                return ChooseAction(random.randrange(len(cards)))
            return ProceedAction()

        elif screen == "COMBAT_REWARD":
            return ProceedAction()

        elif screen in ("SHOP_SCREEN", "SHOP_ROOM"):
            return ProceedAction()

        elif screen == "REST":
            options = getattr(s, "rest_options", [])
            if not options:
                return ProceedAction()
            return ChooseAction(random.randrange(len(options)))

        elif screen in ("CHEST", "OPEN_CHEST"):
            return ProceedAction()

        elif screen == "EVENT":
            options = getattr(s, "options", [])
            return ChooseAction(random.randrange(len(options)) if options else 0)

        elif screen in ("GRID", "HAND_SELECT"):
            if s and getattr(s, "confirm_up", False):
                return ProceedAction()
            cards = getattr(s, "cards", [])
            return ChooseAction(random.randrange(len(cards)) if cards else 0)

        elif screen == "BOSS_REWARD":
            relics = getattr(s, "relics", [])
            return ChooseAction(random.randrange(len(relics)) if relics else 0)

        elif screen in ("GAME_OVER", "COMPLETE"):
            return ProceedAction()

        else:
            logging.getLogger("MetaAgent").warning("Неизвестный экран: '%s' — ProceedAction", screen)
            return ProceedAction()

    # ── Вспомогательные ───────────────────────────────────────────────

    def _handle_rest(self, game) -> "Action":
        options  = getattr(getattr(game, "screen", None), "rest_options", [])
        if not options:
            return ProceedAction()  # Отдых завершён — жмём Proceed
        hp_pct   = game.player.current_hp / max(game.player.max_hp, 1)
        opt_strs = [str(o).upper() for o in options]

        if hp_pct < 0.5:
            # Отдохнуть (восстановить HP)
            for i, o in enumerate(opt_strs):
                if o in ("REST", "SLEEP"):
                    return ChooseAction(i)
        else:
            # Улучшить карту (Smith)
            for i, o in enumerate(opt_strs):
                if o == "SMITH":
                    return ChooseAction(i)

        return ChooseAction(0)


# ── Утилиты ───────────────────────────────────────────────────────────

_BOSS_RELICS = {
    "Burning Blood", "Ring of the Snake", "Cracked Core", "Pure Water",
    "Astrolabe", "Black Star", "Calling Bell", "Coffee Dripper",
    "Cursed Key", "Ectoplasm", "Empty Cage", "Fusion Hammer",
    "Pandora's Box", "Philosopher's Stone", "Runic Dome", "Runic Pyramid",
    "Sacred Bark", "Slaver's Collar", "Snecko Eye", "Sozu",
    "Tiny House", "Velvet Choker", "Violet Lotus",
}


def _screen_str(game) -> str:
    s = str(game.screen_type).upper()
    return s.split(".")[-1] if "." in s else s


def _card_type_str(card) -> str:
    s = str(getattr(card, "type", "")).upper()
    return s.split(".")[-1] if "." in s else s


def _is_boss_relic(relic) -> bool:
    return getattr(relic, "name", "") in _BOSS_RELICS
