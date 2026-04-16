import random
from agents.base_agent import BaseMetaAgent


class RandomMetaAgent(BaseMetaAgent):
    """Случайный мета-агент (baseline).

    Используется как нижняя граница при сравнении агентов.
    """

    def choose_card(self, game) -> int:
        cards = getattr(getattr(game, "screen", None), "cards", [])
        if not cards:
            return -1
        # 80% берём случайную карту, 20% пропускаем
        if random.random() < 0.8:
            return random.randrange(len(cards))
        return -1

    def choose_path(self, game) -> int:
        nodes = getattr(getattr(game, "screen", None), "next_nodes", [])
        return random.randrange(len(nodes)) if nodes else 0

    def choose_rest(self, game) -> str:
        return random.choice(["REST", "SMITH"])

    def choose_event(self, game) -> int:
        options = getattr(getattr(game, "screen", None), "options", [])
        return random.randrange(len(options)) if options else 0

    def choose_boss_relic(self, game) -> int:
        relics = getattr(getattr(game, "screen", None), "relics", [])
        return random.randrange(len(relics)) if relics else 0
