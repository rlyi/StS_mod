import random
import logging
from abc import ABC, abstractmethod

from spirecomm.communication.action import Action, ProceedAction, ChooseAction

# CARD_REWARD: skip требует команду "skip", а не "proceed"
_SkipAction = Action(command="skip")


class BaseMetaAgent(ABC):
    """Абстрактный базовый класс мета-агента.

    Смена агента — одна строка в config.py (META_AGENT = "tree" / "random" / ...).

    Подкласс реализует только стратегические методы:
      choose_card   — какую карту взять в награду
      choose_path   — какой путь на карте выбрать
      choose_rest   — костёр: REST или SMITH
      choose_event  — какой вариант события выбрать

    Вся механика экранов (HAND_SELECT, GRID, BOSS_REWARD и т.д.) реализована
    здесь в act() и одинакова для всех агентов.
    """

    @abstractmethod
    def choose_card(self, game) -> int:
        """Выбрать карту из наград. Возвращает индекс (0-2) или -1 = пропустить."""

    @abstractmethod
    def choose_path(self, game) -> int:
        """Выбрать следующий узел на карте. Возвращает индекс узла."""

    @abstractmethod
    def choose_rest(self, game) -> str:
        """Выбрать действие у костра. Возвращает 'REST' или 'SMITH'."""

    @abstractmethod
    def choose_event(self, game) -> int:
        """Выбрать вариант события. Возвращает индекс."""

    @abstractmethod
    def choose_boss_relic(self, game) -> int:
        """Выбрать реликт босса. Возвращает индекс."""

    def choose_shop(self, game) -> int:
        """Выбрать покупку в магазине. -1 = выйти (по умолчанию пропускаем)."""
        return -1

    def choose_grid(self, game, for_upgrade: bool = False) -> int:
        """Выбрать карту в GRID (апгрейд или удаление). По умолчанию рандом."""
        cards = getattr(getattr(game, "screen", None), "cards", [])
        return random.randrange(len(cards)) if cards else 0

    def choose_hand(self, game) -> int:
        """Выбрать карту из руки (HAND_SELECT). По умолчанию рандом."""
        cards = getattr(getattr(game, "screen", None), "cards", [])
        return random.randrange(len(cards)) if cards else 0

    def act(self, game):
        """Диспетчер экранов. Вызывает choose_* для стратегических решений."""
        screen = _screen_type(game)
        s = getattr(game, "screen", None)
        log = logging.getLogger("META")
        floor = getattr(game, "floor", "?")
        player = game.player
        hp_str = f"{game.current_hp}/{game.max_hp}" if game.max_hp else "?/?"

        if screen == "MAP":
            nodes = getattr(s, "next_nodes", [])
            
            if not nodes:
                # На этаже босса (16, 33, 50) next_nodes может быть пустым
                # если игрок уже стоит на клетке перед боссом.
                # Пробуем ChooseAction(0) - игра сама выберет единственный путь
                log.warning(f"[MAP] floor={floor}, next_nodes пустой — пробуем ChooseAction(0)")
                return ChooseAction(0)
            
            idx = self.choose_path(game)
            idx = max(0, min(idx, len(nodes) - 1))
            symbols = [str(getattr(n, "symbol", "?")) for n in nodes]
            chosen_symbol = symbols[idx] if idx < len(symbols) else "INVALID"
            log.info("[MAP    ] этаж=%-2s HP=%-7s варианты=%s → выбран=%s (idx=%d)",
                     floor, hp_str, symbols, chosen_symbol, idx)
            return ChooseAction(idx)

        elif screen == "CARD_REWARD":
            cards = getattr(s, "cards", [])
            card_names = [getattr(c, "name", str(c)) for c in cards]
            idx = self.choose_card(game)
            if idx < 0:
                log.info("[CARD   ] этаж=%-2s HP=%-7s варианты=%s → SKIP",
                         floor, hp_str, card_names)
                return _SkipAction
            else:
                picked = card_names[idx] if idx < len(card_names) else f"#{idx}"
                log.info("[CARD   ] этаж=%-2s HP=%-7s варианты=%s → '%s'",
                         floor, hp_str, card_names, picked)
                return ChooseAction(idx)

        elif screen == "REST":
            options = getattr(s, "rest_options", [])
            if not options:
                return ProceedAction()
            choice = self.choose_rest(game).upper()
            opt_strs = [str(o).upper() for o in options]
            log.info("[REST   ] этаж=%-2s HP=%-7s варианты=%s → %s",
                     floor, hp_str, opt_strs, choice)
            for i, o in enumerate(opt_strs):
                if choice in o or o in choice:
                    return ChooseAction(i)
            return ChooseAction(0)

        elif screen == "EVENT":
            options = getattr(s, "options", [])
            if not options:
                return ProceedAction()
            # Фильтруем заблокированные опции — spirecomm их считает, игра нет
            # Игра нумерует только доступные опции, поэтому нужен маппинг индексов
            available = [i for i, o in enumerate(options)
                         if not getattr(o, "locked", False)
                         and "[locked]" not in getattr(o, "text", str(o)).lower()]
            if not available:
                return ProceedAction()
            opt_names = [getattr(o, "text", str(o))[:40] for o in options]
            idx = self.choose_event(game)
            idx = max(0, min(idx, len(options) - 1))
            # Если выбрали заблокированную — берём последнюю доступную (обычно Leave)
            if idx not in available:
                idx = available[-1]
            # Игра считает только незаблокированные опции — маппим в их позицию
            game_idx = available.index(idx)
            log.info("[EVENT  ] этаж=%-2s HP=%-7s варианты=%s → #%d '%s'",
                     floor, hp_str, opt_names, game_idx, opt_names[idx])
            return ChooseAction(game_idx)

        elif screen == "BOSS_REWARD":
            relics = getattr(s, "relics", [])
            relic_names = [getattr(r, "name", str(r)) for r in relics]
            idx = self.choose_boss_relic(game)
            idx = max(0, min(idx, len(relics) - 1)) if relics else 0
            picked = relic_names[idx] if idx < len(relic_names) else f"#{idx}"
            log.info("[BOSS   ] этаж=%-2s HP=%-7s реликты=%s → '%s'",
                     floor, hp_str, relic_names, picked)
            return ChooseAction(idx)

        elif screen == "COMBAT_REWARD":
            return ProceedAction()

        elif screen in ("SHOP_SCREEN", "SHOP_ROOM"):
            idx = self.choose_shop(game)
            if idx < 0:
                return ProceedAction()  # выйти из магазина
            cards   = getattr(s, "cards",   [])
            relics  = getattr(s, "relics",  [])
            potions = getattr(s, "potions", [])
            all_items = cards + relics + potions
            if idx >= len(all_items):
                return ProceedAction()
            item_name = getattr(all_items[idx], "name", str(all_items[idx]))
            log.info("[SHOP   ] этаж=%-2s HP=%-7s золота=%s → покупаем '%s' (idx=%d)",
                     floor, hp_str, getattr(game, "gold", "?"), item_name, idx)
            return ChooseAction(idx)

        elif screen in ("CHEST", "OPEN_CHEST"):
            return ProceedAction()

        elif screen == "HAND_SELECT":
            num_cards = getattr(s, "num_cards", 1)
            selected  = getattr(s, "selected_cards", [])
            cards     = getattr(s, "cards", [])
            if len(selected) >= num_cards or not cards:
                return ProceedAction()
            return ChooseAction(self.choose_hand(game))

        elif screen == "GRID":
            if s and getattr(s, "confirm_up", False):
                return ProceedAction()
            cards = getattr(s, "cards", [])
            if not cards:
                return ProceedAction()
            # Определяем контекст: апгрейд (for_upgrade) если экран для смита
            for_upgrade = getattr(s, "for_upgrade", False)
            return ChooseAction(self.choose_grid(game, for_upgrade=for_upgrade))

        elif screen in ("GAME_OVER", "COMPLETE"):
            return ProceedAction()

        else:
            logging.getLogger(self.__class__.__name__).warning(
                "Неизвестный экран: '%s' — ProceedAction", screen)
            return ProceedAction()


def _screen_type(game) -> str:
    s = str(game.screen_type).upper()
    return s.split(".")[-1] if "." in s else s
