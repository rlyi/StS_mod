from enum import Enum

from engine.enums import CardId


class MemoryItem(Enum):
    ATTACKS_THIS_TURN = 'attacks_this_turn'
    CARDS_THIS_TURN = 'cards_this_turn'
    CLAWS_THIS_BATTLE = 'claws_this_battle'
    FROST_THIS_BATTLE = 'frost_this_battle'
    KILLED_WITH_LESSON_LEARNED = 'killed_with_lesson_learned'
    LAST_KNOWN_TURN = 'last_known_turn'
    LIGHTNING_THIS_BATTLE = 'lightning_this_battle'
    MANTRA_THIS_BATTLE = 'mantra_this_battle'
    NECRONOMICON_READY = 'necronomicon_ready'
    ORANGE_PELLETS_ATTACK = 'orange_pellets_attack'
    ORANGE_PELLETS_POWER = 'orange_pellets_power'
    ORANGE_PELLETS_SKILL = 'orange_pellets_skill'
    PANACHE_COUNTER = 'panache_counter'
    PANACHE_DAMAGE = 'panache_damage'
    RECYCLE = 'recycle'
    SAVE_INTERNAL_MANTRA = 'save_internal_mantra'
    STANCE = 'stance'
    TYPE_LAST_PLAYED = 'type_last_played'


class ResetSchedule(Enum):
    GAME = 'game'
    BATTLE = 'battle'
    TURN = 'turn'


class StanceType(Enum):
    NO_STANCE = 'no_stance'
    CALM = 'calm'
    WRATH = 'wrath'
    DIVINITY = 'divinity'


class TheBotsMemoryBook:
    def __init__(self, memory_general: dict = None,
                 memory_by_card: dict = None):
        self.memory_general = {} if memory_general is None else memory_general
        self.memory_by_card = {} if memory_by_card is None else memory_by_card

    def set_new_game_state(self):
        for card_id in [
            CardId.GENETIC_ALGORITHM,
            CardId.GLASS_KNIFE,
            CardId.PERSEVERANCE,
            CardId.RAMPAGE,
            CardId.RITUAL_DAGGER,
            CardId.STEAM_BARRIER,
            CardId.WINDMILL_STRIKE,
        ]:
            self.initialize_memory_by_card(card_id)

        self.memory_general[MemoryItem.KILLED_WITH_LESSON_LEARNED] = 0

        self.set_new_battle_state()
        self.set_new_turn_state()

    def set_new_battle_state(self):
        self.memory_general[MemoryItem.CLAWS_THIS_BATTLE] = 0
        self.memory_general[MemoryItem.FROST_THIS_BATTLE] = 0
        self.memory_general[MemoryItem.LIGHTNING_THIS_BATTLE] = 0
        self.memory_general[MemoryItem.MANTRA_THIS_BATTLE] = 0
        self.memory_general[MemoryItem.PANACHE_DAMAGE] = 0
        self.memory_general[MemoryItem.SAVE_INTERNAL_MANTRA] = 0
        self.memory_general[MemoryItem.STANCE] = StanceType.NO_STANCE
        self.memory_general[MemoryItem.TYPE_LAST_PLAYED] = 0

        for card_id, schedule_dict in self.memory_by_card.items():
            for reset_schedule in schedule_dict.keys():
                if reset_schedule == ResetSchedule.BATTLE:
                    self.initialize_memory_by_card(card_id)

        self.set_new_turn_state()

    def set_new_turn_state(self):
        self.memory_general[MemoryItem.ATTACKS_THIS_TURN] = 0
        self.memory_general[MemoryItem.CARDS_THIS_TURN] = 0
        self.memory_general[MemoryItem.LAST_KNOWN_TURN] = 0
        self.memory_general[MemoryItem.NECRONOMICON_READY] = 1
        self.memory_general[MemoryItem.ORANGE_PELLETS_ATTACK] = 0
        self.memory_general[MemoryItem.ORANGE_PELLETS_SKILL] = 0
        self.memory_general[MemoryItem.ORANGE_PELLETS_POWER] = 0
        self.memory_general[MemoryItem.PANACHE_COUNTER] = 5
        self.memory_general[MemoryItem.RECYCLE] = 0

    def initialize_memory_by_card(self, card_id: CardId):
        reset_schedule = None
        if card_id == CardId.GENETIC_ALGORITHM:
            reset_schedule = ResetSchedule.GAME
        elif card_id == CardId.GLASS_KNIFE:
            reset_schedule = ResetSchedule.BATTLE
        elif card_id == CardId.PERSEVERANCE:
            reset_schedule = ResetSchedule.BATTLE
        elif card_id == CardId.RAMPAGE:
            reset_schedule = ResetSchedule.BATTLE
        elif card_id == CardId.RITUAL_DAGGER:
            reset_schedule = ResetSchedule.GAME
        elif card_id == CardId.STEAM_BARRIER:
            reset_schedule = ResetSchedule.BATTLE
        elif card_id == CardId.WINDMILL_STRIKE:
            reset_schedule = ResetSchedule.BATTLE
        else:
            return
        self.memory_by_card[card_id] = {reset_schedule: {"": 0}}

    @staticmethod
    def new_default(last_known_turn: int = 0):
        book = TheBotsMemoryBook()
        book.set_new_game_state()
        book.memory_general[MemoryItem.LAST_KNOWN_TURN] = last_known_turn
        return book
