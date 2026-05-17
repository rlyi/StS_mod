import math
from typing import Callable, List, Optional

from engine.battle_state import BattleState
from engine.enums import CardId, PowerId, RelicId, CardType
from engine.entities import DEBUFFS
from engine.memory import MemoryItem, ResetSchedule, StanceType
from engine.helper import pickle_deepcopy


def get_power_count(powers: dict, desired_powers: list) -> int:
    return sum(powers[p] for p in powers.keys() if p in desired_powers)


class ComparatorAssessmentConfig:
    def __init__(self, powers_we_like: List[PowerId], powers_we_like_less: List[PowerId],
                 powers_we_dislike: List[PowerId], powers_we_love: List[PowerId] = None,
                 cards_that_exit_wrath: List[CardId] = None):
        self.powers_we_love: List[PowerId] = [] if powers_we_love is None else powers_we_love
        self.powers_we_like: List[PowerId] = powers_we_like
        self.powers_we_like_less: List[PowerId] = powers_we_like_less
        self.powers_we_dislike: List[PowerId] = powers_we_dislike
        self.cards_that_exit_wrath: List[CardId] = [] if cards_that_exit_wrath is None else cards_that_exit_wrath


CA = None  # forward ref, defined below


class ComparatorAssessment:
    def __init__(self, state: BattleState, original: BattleState, config: ComparatorAssessmentConfig):
        self.state: BattleState = state
        self.original: BattleState = original
        self.cached_values: dict = {}
        self.config: ComparatorAssessmentConfig = config

    def __get_value(self, name: str, load_function):
        if self.cached_values.get(name) is None:
            self.cached_values[name] = load_function()
        return self.cached_values.get(name)

    def battle_won(self) -> bool:
        alive_monsters = False
        unawakened_present = False
        for mon in self.state.monsters:
            if mon.current_hp > 0:
                alive_monsters = True
            if mon.powers.get(PowerId.UNAWAKENED, 0):
                unawakened_present = True
        return self.__get_value('bw', lambda: not alive_monsters and not unawakened_present)

    def battle_lost(self) -> bool:
        return self.__get_value('bl', lambda: self.state.player.current_hp <= 0)

    def incoming_damage(self) -> int:
        return self.__get_value('id', lambda: self.original.player.current_hp - self.state.player.current_hp)

    def dead_monsters(self) -> int:
        return self.__get_value('dm', lambda: len([True for monster in self.state.monsters if monster.current_hp <= 0]))

    def dead_edge_monsters(self) -> int:
        return self.__get_value('dem', lambda:
        0 if self.battle_won() else self.state.monsters[0].current_hp <= 0 or self.state.monsters[2].current_hp <= 0)

    def monsters_vulnerable_hp(self) -> List[int]:
        return self.__get_value('mvhp',
                                lambda: [monster.current_hp - min(monster.powers.get(PowerId.VULNERABLE, 0) * 5, 3) for
                                         monster in self.state.monsters if monster.current_hp > 0] or [0])

    def lowest_health_monster(self) -> int:
        return self.__get_value('lhm', lambda: 0 if self.battle_won() else min(self.monsters_vulnerable_hp()))

    def lowest_true_health_monster(self) -> int:
        return self.__get_value('lowest_true_health_monster', lambda:
        0 if self.battle_won() else min([m.current_hp for m in self.state.monsters]))

    def lowest_health_edge_monster(self) -> int:
        return self.__get_value('lowest_health_edge_monster', lambda:
        0 if self.battle_won() else min(self.monsters_vulnerable_hp()[0], self.monsters_vulnerable_hp()[-1]))

    def total_monster_health(self) -> int:
        return self.__get_value('tmh', lambda:
        0 if self.battle_won() else sum(
            self.monsters_vulnerable_hp()) - self.state.total_random_damage_dealt - self.state.total_random_poison_added)

    def total_monster_health_percent(self) -> float:
        return self.__get_value('total_monster_health_percent', lambda: 0 if self.battle_won()
        else float(sum([m.current_hp for m in self.state.monsters])) / float(
            sum([m.max_hp for m in self.state.monsters])))

    def draw_free_early(self) -> int:
        return self.__get_value('dfe', lambda: self.state.draw_free_early)

    def draw_free(self) -> int:
        return self.__get_value('df', lambda: self.state.draw_free + self.state.draw_free_early)

    def draw_pay_early(self) -> int:
        return self.__get_value('dpe', lambda: self.state.draw_pay_early)

    def draw_pay(self) -> int:
        return self.__get_value('dp', lambda: self.state.draw_pay + self.state.draw_pay_early)

    def energy(self) -> int:
        return self.__get_value('e', lambda: self.state.player.energy)

    def intangible(self) -> int:
        return self.__get_value('i', lambda: self.state.player.powers.get(PowerId.INTANGIBLE_PLAYER, 0))

    def enemy_vulnerable(self) -> int:
        return self.__get_value('ev',
                                lambda: min(max([m.powers.get(PowerId.VULNERABLE, 0) for m in self.state.monsters]), 4))

    def enemy_weak(self) -> int:
        return self.__get_value('ew',
                                lambda: min(max([m.powers.get(PowerId.WEAKENED, 0) for m in self.state.monsters]), 4))

    def enemy_talking_to_hand(self) -> int:
        return self.__get_value('eh',
                                lambda: min(max([m.powers.get(PowerId.BLOCK_RETURN, 0) for m in self.state.monsters]),
                                            10))

    def player_max_hp(self) -> int:
        return self.__get_value('pmhp', lambda: self.state.player.max_hp)

    def pen_nib_counter(self) -> int:
        return self.__get_value('penc', lambda: self.state.player.relics.get(RelicId.PEN_NIB, -1))

    def nunchaku_counter(self) -> int:
        return self.__get_value('nunc', lambda: self.state.player.relics.get(RelicId.NUNCHAKU, -1))

    def ink_bottle_counter(self) -> int:
        return self.__get_value('ink_bottle_counter', lambda: self.state.player.relics.get(RelicId.INK_BOTTLE, -1))

    def player_powers_great(self) -> int:
        return self.__get_value('player_powers_great',
                                lambda: get_power_count(self.state.player.powers, self.config.powers_we_love))

    def player_powers_good(self) -> int:
        return self.__get_value('player_powers_good',
                                lambda: get_power_count(self.state.player.powers, self.config.powers_we_like))

    def player_powers_less_good(self) -> int:
        return self.__get_value('powers_less_good',
                                lambda: get_power_count(self.state.player.powers, self.config.powers_we_like_less))

    def player_powers_bad(self) -> int:
        return self.__get_value('player_powers_bad',
                                lambda: get_power_count(self.state.player.powers, self.config.powers_we_dislike))

    def bad_cards_exhausted(self) -> int:
        return self.__get_value('bad_cards_exhausted', lambda: len(
            [True for c in self.state.exhaust_pile if c.type == CardType.CURSE or c.type == CardType.STATUS]))

    def ethereal_saved_for_later(self) -> int:
        return self.__get_value('ethereal_saved_for_later',
                                lambda: len([True for c in self.state.discard_pile if c.ethereal
                                             and c.type != CardType.CURSE
                                             and c.type != CardType.STATUS]))

    def awkward_shivs(self) -> int:
        return self.__get_value('awkward_shivs',
                                lambda: len([True for c in self.state.hand if c.id == CardId.SHIV]) + len(
                                    [True for c in self.state.discard_pile if c.id == CardId.SHIV]))

    def enemy_artifacts(self) -> int:
        return self.__get_value('enemy_artifacts',
                                lambda: sum([m.powers.get(PowerId.ARTIFACT, 0) for m in self.state.monsters]))

    def barricaded_block(self) -> int:
        return self.__get_value('barricaded_block', lambda: sum(
            [m.block for m in self.state.monsters if m.powers.get(PowerId.BARRICADE, 0) != 0]))

    def nob_adjusted_scaling_damage(self) -> int:
        return self.__get_value('nob_adjusted_incoming_damage', self.__nob_adjusted_incoming_damage)

    def __nob_adjusted_incoming_damage(self) -> int:
        anger_strength_up = sum([m.powers.get(PowerId.STRENGTH, 0) for m in self.state.monsters if
                                 m.powers.get(PowerId.ANGER_NOB, 0)])
        gremlin_nob_hp = sum([m.current_hp for m in self.state.monsters if m.powers.get(PowerId.ANGER_NOB, 0)])
        return self.original.player.current_hp - self.state.player.current_hp + (
                int(gremlin_nob_hp / 15) * anger_strength_up)

    def orb_slot_count(self) -> int:
        return self.__get_value('orb_slots', lambda: self.state.orb_slots)

    def channeled_orb_count(self) -> int:
        return self.__get_value('orb_count', lambda: len(self.state.orbs))

    def player_bias(self) -> int:
        return self.__get_value('player_bias', lambda: self.state.player.powers.get(PowerId.BIAS, 0))

    def repair_count(self) -> int:
        missing_hp = self.state.player.max_hp - self.state.player.current_hp
        if missing_hp >= 1:
            return self.__get_value('repair_count', lambda: self.state.player.powers.get(PowerId.REPAIR, 0))
        return 0

    def cards_left_in_hand(self) -> int:
        return self.__get_value('cards_left_in_hand', lambda: len(self.state.hand))

    def power_up_ritual_dagger(self) -> int:
        return self.__get_value('power_up_ritual_dagger',
                                lambda: sum(
                                    self.state.memory_by_card[CardId.RITUAL_DAGGER][ResetSchedule.GAME].values()))

    def power_up_genetic_algorithm(self) -> int:
        return self.__get_value('power_up_genetic_algorithm',
                                lambda: sum(
                                    self.state.memory_by_card[CardId.GENETIC_ALGORITHM][ResetSchedule.GAME].values()))

    def power_down_steam_barrier(self) -> int:
        return self.__get_value('power_down_steam_barrier',
                                lambda: sum(
                                    self.state.memory_by_card[CardId.STEAM_BARRIER][ResetSchedule.BATTLE].values()))

    def powered_up_claws(self) -> int:
        return self.__get_value('powered_up_claws',
                                lambda: self.state.memory_general[MemoryItem.CLAWS_THIS_BATTLE])

    def enemy_plated_armor(self) -> int:
        return self.__get_value('enemy_plated_armor', lambda: sum(
            [m.powers.get(PowerId.PLATED_ARMOR, 0) for m in self.state.monsters if
             m.powers.get(PowerId.PLATED_ARMOR, 0) != 0]))

    def stance_is_calm(self) -> int:
        return self.__get_value('stance_is_calm',
                                lambda: 1 if self.state.memory_general[MemoryItem.STANCE] == StanceType.CALM else 0)

    def stance_is_not_wrath(self) -> int:
        exit_plan = False
        for c in self.state.hand:
            if c.id in self.config.cards_that_exit_wrath:
                exit_plan = True
        return self.__get_value('stance_is_not_wrath',
                                lambda: 1 if self.state.memory_general[
                                                 MemoryItem.STANCE] == StanceType.WRATH and not exit_plan else 0)

    def played_blasphemy(self) -> int:
        return self.__get_value('we_played_blasphemy_without_permission',
                                lambda: 1 if self.state.player.powers.get(PowerId.BLASPHEMER, 0) else 0)

    def most_kills_with_lesson_learned(self) -> int:
        return self.__get_value('most_kills_with_lesson_learned',
                                lambda: self.state.memory_general[MemoryItem.KILLED_WITH_LESSON_LEARNED])

    def count_tranquility(self) -> int:
        return self.__get_value('tinh',
                                lambda: len([True for c in self.state.hand if (c.id == CardId.TRANQUILITY)]))

    def count_crescendo(self) -> int:
        return self.__get_value('cinh',
                                lambda: len([True for c in self.state.hand if (c.id == CardId.CRESCENDO)]))

    def block_for_next_turn(self) -> int:
        return self.__get_value('bfnt', lambda: self.state.saved_block_for_next_turn)

    def count_expensive_cheapening_retain_cards(self) -> int:
        return self.__get_value('cdst',
                                lambda: len(
                                    [True for c in self.state.hand if (c.id == CardId.SANDS_OF_TIME and c.cost > 0)]))

    def inconvenient_time_warp_count(self) -> int:
        bad_time_warp = False
        for m in self.state.monsters:
            if PowerId.TIME_WARP in m.powers:
                if m.powers[PowerId.TIME_WARP] == 10 or m.powers[PowerId.TIME_WARP] == 11:
                    bad_time_warp = True
        return self.__get_value('nitwc', lambda: bad_time_warp)

    def revive_option_count(self) -> int:
        from engine.enums import PotionId, RelicId as R
        fairy_count = 0
        lizard_count = 0
        if PotionId.FAIRY_IN_A_BOTTLE in self.state.potions:
            for p in self.state.potions:
                if p == PotionId.FAIRY_IN_A_BOTTLE:
                    fairy_count += 1
        if self.state.relics.get(RelicId.LIZARD_TAIL):
            if self.state.relics[RelicId.LIZARD_TAIL] != -2:
                lizard_count += 1
        return self.__get_value('roc', lambda: fairy_count + lizard_count)


# type alias
CA = ComparatorAssessment

# ── comparison functions ──────────────────────────────────────────────────────

def battle_not_lost(best: CA, challenger: CA) -> Optional[bool]:
    return None if best.battle_lost() == challenger.battle_lost() else not challenger.battle_lost()


def battle_is_won(best: CA, challenger: CA) -> Optional[bool]:
    return None if best.battle_won() == challenger.battle_won() else challenger.battle_won()


def most_optimal_winning_battle(best: CA, challenger: CA) -> Optional[bool]:
    if not best.battle_won() or not challenger.battle_won():
        return None
    if best.player_max_hp() != challenger.player_max_hp():
        return challenger.player_max_hp() > best.player_max_hp()
    if best.most_kills_with_lesson_learned() != challenger.most_kills_with_lesson_learned():
        return challenger.most_kills_with_lesson_learned() > best.most_kills_with_lesson_learned()
    if best.power_up_ritual_dagger() != challenger.power_up_ritual_dagger():
        return challenger.power_up_ritual_dagger() > best.power_up_ritual_dagger()
    if best.power_up_genetic_algorithm() != challenger.power_up_genetic_algorithm():
        return challenger.power_up_genetic_algorithm() > best.power_up_genetic_algorithm()
    if best.incoming_damage() != challenger.incoming_damage():
        return challenger.incoming_damage() < best.incoming_damage()
    if best.repair_count() != challenger.repair_count():
        return challenger.repair_count() > best.repair_count()
    if best.pen_nib_counter() != challenger.pen_nib_counter():
        return challenger.pen_nib_counter() > best.pen_nib_counter()
    if best.nunchaku_counter() != challenger.nunchaku_counter():
        return challenger.nunchaku_counter() > best.nunchaku_counter()
    if best.ink_bottle_counter() != challenger.ink_bottle_counter():
        return challenger.ink_bottle_counter() > best.ink_bottle_counter()
    if best.cards_left_in_hand() != challenger.cards_left_in_hand():
        return challenger.cards_left_in_hand() > best.cards_left_in_hand()
    if best.energy() != challenger.energy():
        return challenger.energy() > best.energy()
    return False


def most_free_early_draw(best: CA, challenger: CA) -> Optional[bool]:
    return None if best.draw_free_early() == challenger.draw_free_early() \
        else challenger.draw_free_early() > best.draw_free_early()


def most_free_draw(best: CA, challenger: CA) -> Optional[bool]:
    return None if best.draw_free() == challenger.draw_free() \
        else challenger.draw_free() > best.draw_free()


def most_lasting_intangible(best: CA, challenger: CA) -> Optional[bool]:
    return None if max(1, best.intangible()) == max(1, challenger.intangible()) \
        else challenger.intangible() > best.intangible()


def least_incoming_damage_over_1(best: CA, challenger: CA) -> Optional[bool]:
    return None if max(2, best.incoming_damage()) == max(2, challenger.incoming_damage()) \
        else challenger.incoming_damage() < best.incoming_damage()


def most_dead_monsters(best: CA, challenger: CA) -> Optional[bool]:
    return None if best.dead_monsters() == challenger.dead_monsters() \
        else challenger.dead_monsters() > best.dead_monsters()


def most_dead_edge_monsters(best: CA, challenger: CA) -> Optional[bool]:
    return None if best.dead_edge_monsters() == challenger.dead_edge_monsters() \
        else challenger.dead_edge_monsters() > best.dead_edge_monsters()


def most_enemy_vulnerable(best: CA, challenger: CA) -> Optional[bool]:
    return None if max(1, best.enemy_vulnerable()) == max(1, challenger.enemy_vulnerable()) \
        else challenger.enemy_vulnerable() > best.enemy_vulnerable()


def most_enemy_weak(best: CA, challenger: CA) -> Optional[bool]:
    return None if max(1, best.enemy_weak()) == max(1, challenger.enemy_weak()) \
        else challenger.enemy_weak() > best.enemy_weak()


def most_enemy_talking_to_hand(best: CA, challenger: CA) -> Optional[bool]:
    return None if best.enemy_talking_to_hand() == challenger.enemy_talking_to_hand() \
        else challenger.enemy_talking_to_hand() > best.enemy_talking_to_hand()


def lowest_health_monster(best: CA, challenger: CA) -> Optional[bool]:
    return None if best.lowest_health_monster() == challenger.lowest_health_monster() \
        else challenger.lowest_health_monster() < best.lowest_health_monster()


def highest_health_monster(best: CA, challenger: CA) -> Optional[bool]:
    return None if best.lowest_true_health_monster() == challenger.lowest_true_health_monster() \
        else challenger.lowest_true_health_monster() > best.lowest_true_health_monster()


def lowest_health_edge_monster(best: CA, challenger: CA) -> Optional[bool]:
    return None if best.lowest_health_edge_monster() == challenger.lowest_health_edge_monster() \
        else challenger.lowest_health_edge_monster() < best.lowest_health_edge_monster()


def lowest_total_monster_health(best: CA, challenger: CA) -> Optional[bool]:
    return None if best.total_monster_health() == challenger.total_monster_health() \
        else challenger.total_monster_health() < best.total_monster_health()


def lowest_barricaded_block(best: CA, challenger: CA) -> Optional[bool]:
    return None if best.barricaded_block() == challenger.barricaded_block() \
        else challenger.barricaded_block() < best.barricaded_block()


def most_draw_pay_early(best: CA, challenger: CA) -> Optional[bool]:
    return None if best.draw_pay_early() == challenger.draw_pay_early() \
        else challenger.draw_pay_early() > best.draw_pay_early()


def most_draw_pay(best: CA, challenger: CA) -> Optional[bool]:
    return None if best.draw_pay() == challenger.draw_pay() else challenger.draw_pay() > best.draw_pay()


def most_good_player_powers(best: CA, challenger: CA) -> Optional[bool]:
    return None if best.player_powers_good() == challenger.player_powers_good() \
        else challenger.player_powers_good() > best.player_powers_good()


def most_great_player_powers(best: CA, challenger: CA) -> Optional[bool]:
    return None if best.player_powers_great() == challenger.player_powers_great() \
        else challenger.player_powers_great() > best.player_powers_great()


def most_less_good_player_powers(best: CA, challenger: CA) -> Optional[bool]:
    return None if best.player_powers_less_good() == challenger.player_powers_less_good() \
        else challenger.player_powers_less_good() > best.player_powers_less_good()


def least_bad_player_powers(best: CA, challenger: CA) -> Optional[bool]:
    return None if best.player_powers_bad() == challenger.player_powers_bad() \
        else challenger.player_powers_bad() < best.player_powers_bad()


def most_bad_cards_exhausted(best: CA, challenger: CA) -> Optional[bool]:
    return None if best.bad_cards_exhausted() == challenger.bad_cards_exhausted() \
        else challenger.bad_cards_exhausted() > best.bad_cards_exhausted()


def most_energy(best: CA, challenger: CA) -> Optional[bool]:
    return None if best.energy() == challenger.energy() else challenger.energy() > best.energy()


def least_incoming_damage(best: CA, challenger: CA) -> Optional[bool]:
    return None if best.incoming_damage() == challenger.incoming_damage() \
        else challenger.incoming_damage() < best.incoming_damage()


def most_ethereal_cards_saved_for_later(best: CA, challenger: CA) -> Optional[bool]:
    return None if best.ethereal_saved_for_later() == challenger.ethereal_saved_for_later() \
        else challenger.ethereal_saved_for_later() > best.ethereal_saved_for_later()


def least_awkward_shivs(best: CA, challenger: CA) -> Optional[bool]:
    return None if best.awkward_shivs() == challenger.awkward_shivs() \
        else challenger.awkward_shivs() < best.awkward_shivs()


def least_enemy_artifacts(best: CA, challenger: CA) -> Optional[bool]:
    return None if best.enemy_artifacts() == challenger.enemy_artifacts() \
        else challenger.enemy_artifacts() < best.enemy_artifacts()


def least_nob_adjusted_scaling_damage(best: CA, challenger: CA) -> Optional[bool]:
    return None if best.nob_adjusted_scaling_damage() == challenger.nob_adjusted_scaling_damage() \
        else challenger.nob_adjusted_scaling_damage() < best.nob_adjusted_scaling_damage()


def most_orb_slots(best: CA, challenger: CA) -> Optional[bool]:
    return None if best.orb_slot_count() == challenger.orb_slot_count() \
        else challenger.orb_slot_count() > best.orb_slot_count()


def most_channeled_orbs(best: CA, challenger: CA) -> Optional[bool]:
    return None if best.channeled_orb_count() == challenger.channeled_orb_count() \
        else challenger.channeled_orb_count() > best.channeled_orb_count()


def most_cards_left_in_hand(best: CA, challenger: CA) -> Optional[bool]:
    return None if best.cards_left_in_hand() == challenger.cards_left_in_hand() \
        else challenger.cards_left_in_hand() > best.cards_left_in_hand()


def most_powered_up_ritual_dagger(best: CA, challenger: CA) -> Optional[bool]:
    return None if best.power_up_ritual_dagger() == challenger.power_up_ritual_dagger() \
        else challenger.power_up_ritual_dagger() > best.power_up_ritual_dagger()


def most_powered_up_genetic_algorithm(best: CA, challenger: CA) -> Optional[bool]:
    return None if best.power_up_genetic_algorithm() == challenger.power_up_genetic_algorithm() \
        else challenger.power_up_genetic_algorithm() > best.power_up_genetic_algorithm()


def least_powered_down_steam_barrier(best: CA, challenger: CA) -> Optional[bool]:
    return None if best.power_down_steam_barrier() == challenger.power_down_steam_barrier() \
        else challenger.power_down_steam_barrier() < best.power_down_steam_barrier()


def most_powered_up_claws(best: CA, challenger: CA) -> Optional[bool]:
    return None if best.powered_up_claws() == challenger.powered_up_claws() \
        else challenger.powered_up_claws() > best.powered_up_claws()


def lowest_enemy_plated_armor(best: CA, challenger: CA) -> Optional[bool]:
    return None if best.enemy_plated_armor() == challenger.enemy_plated_armor() \
        else challenger.enemy_plated_armor() < best.enemy_plated_armor()


def stance_is_not_wrath(best: CA, challenger: CA) -> Optional[bool]:
    return None if best.stance_is_not_wrath() == challenger.stance_is_not_wrath() \
        else challenger.stance_is_not_wrath() < best.stance_is_not_wrath()


def stance_is_calm(best: CA, challenger: CA) -> Optional[bool]:
    return None if best.stance_is_calm() == challenger.stance_is_calm() \
        else challenger.stance_is_calm() > best.stance_is_calm()


def no_blasphemy(best: CA, challenger: CA) -> Optional[bool]:
    return None if best.played_blasphemy() == challenger.played_blasphemy() \
        else challenger.played_blasphemy() < best.played_blasphemy()


def killed_with_lesson_learned(best: CA, challenger: CA) -> Optional[bool]:
    return None if best.most_kills_with_lesson_learned() == challenger.most_kills_with_lesson_learned() \
        else challenger.most_kills_with_lesson_learned() > best.most_kills_with_lesson_learned()


def most_tranquility(best: CA, challenger: CA) -> Optional[bool]:
    return None if best.count_tranquility() == challenger.count_tranquility() \
        else challenger.count_tranquility() > best.count_tranquility()


def most_crescendo(best: CA, challenger: CA) -> Optional[bool]:
    return None if best.count_crescendo() == challenger.count_crescendo() \
        else challenger.count_crescendo() > best.count_crescendo()


def most_block_saved_for_next_turn(best: CA, challenger: CA) -> Optional[bool]:
    return None if best.block_for_next_turn() == challenger.block_for_next_turn() \
        else challenger.block_for_next_turn() > best.block_for_next_turn()


def kept_expensive_decreasing_cost_retain_cards(best: CA, challenger: CA) -> Optional[bool]:
    return None if best.count_expensive_cheapening_retain_cards() == challenger.count_expensive_cheapening_retain_cards() \
        else challenger.count_expensive_cheapening_retain_cards() > best.count_expensive_cheapening_retain_cards()


def avoid_inconvenient_time_warp(best: CA, challenger: CA) -> Optional[bool]:
    return None if best.inconvenient_time_warp_count() == challenger.inconvenient_time_warp_count() \
        else challenger.inconvenient_time_warp_count() < best.inconvenient_time_warp_count()


def preserve_revive_options(best: CA, challenger: CA) -> Optional[bool]:
    return None if best.revive_option_count() == challenger.revive_option_count() \
        else challenger.revive_option_count() > best.revive_option_count()


# ── comparator classes ────────────────────────────────────────────────────────

Comparison = Callable[[CA, CA], Optional[bool]]

powers_we_like: List[PowerId] = [
    PowerId.ACCURACY, PowerId.AFTER_IMAGE, PowerId.BATTLE_HYMN, PowerId.BARRICADE,
    PowerId.BERSERK, PowerId.BLUR, PowerId.BUFFER, PowerId.COLLECT, PowerId.CORRUPTION,
    PowerId.DARK_EMBRACE, PowerId.DEMON_FORM, PowerId.DEVA, PowerId.DEVOTION,
    PowerId.ECHO_FORM, PowerId.ELECTRO, PowerId.ENVENOM, PowerId.ESTABLISHMENT,
    PowerId.EVOLVE, PowerId.FAKE_ALPHA_BETA, PowerId.FEEL_NO_PAIN, PowerId.FIRE_BREATHING,
    PowerId.FOCUS, PowerId.FORESIGHT, PowerId.HEATSINK, PowerId.INFINITE_BLADES,
    PowerId.INTANGIBLE_PLAYER, PowerId.JUGGERNAUT, PowerId.LIKE_WATER, PowerId.LOOP,
    PowerId.MACHINE_LEARNING, PowerId.MANTRA_INTERNAL, PowerId.MASTER_REALITY,
    PowerId.MAYHEM, PowerId.MENTAL_FORTRESS, PowerId.METALLICIZE, PowerId.NIRVANA,
    PowerId.NOXIOUS_FUMES, PowerId.OMEGA, PowerId.PANACHE_INTERNAL, PowerId.PHANTASMAL,
    PowerId.PLATED_ARMOR, PowerId.REPAIR, PowerId.RUSHDOWN, PowerId.SADISTIC,
    PowerId.SIMMERING_RAGE, PowerId.STUDY, PowerId.THORNS, PowerId.THOUSAND_CUTS,
    PowerId.TOOLS_OF_THE_TRADE,
]

powers_we_like_less: List[PowerId] = [
    PowerId.ARTIFACT, PowerId.DEXTERITY, PowerId.ENERGIZED,
    PowerId.FREE_ATTACK_POWER, PowerId.STRENGTH, PowerId.VIGOR,
]

cards_that_exit_wrath: List[CardId] = [
    CardId.EMPTY_BODY, CardId.EMPTY_FIST, CardId.EMPTY_MIND,
    CardId.FEAR_NO_EVIL, CardId.INNER_PEACE, CardId.TRANQUILITY, CardId.VIGILANCE,
]

powers_we_dislike: List[PowerId] = DEBUFFS.copy()

default_comparisons: List[Comparison] = [
    battle_not_lost,
    battle_is_won,
    preserve_revive_options,
    most_optimal_winning_battle,
    no_blasphemy,
    most_free_early_draw,
    most_free_draw,
    most_lasting_intangible,
    least_incoming_damage_over_1,
    most_great_player_powers,
    most_dead_monsters,
    most_tranquility,
    most_enemy_talking_to_hand,
    most_enemy_vulnerable,
    most_enemy_weak,
    least_awkward_shivs,
    killed_with_lesson_learned,
    most_powered_up_ritual_dagger,
    kept_expensive_decreasing_cost_retain_cards,
    lowest_health_monster,
    lowest_total_monster_health,
    lowest_barricaded_block,
    lowest_enemy_plated_armor,
    most_orb_slots,
    most_channeled_orbs,
    most_draw_pay_early,
    most_draw_pay,
    most_good_player_powers,
    least_bad_player_powers,
    most_less_good_player_powers,
    least_enemy_artifacts,
    most_bad_cards_exhausted,
    most_powered_up_genetic_algorithm,
    most_cards_left_in_hand,
    least_incoming_damage,
    most_ethereal_cards_saved_for_later,
    most_powered_up_claws,
    stance_is_not_wrath,
    stance_is_calm,
    least_powered_down_steam_barrier,
    most_block_saved_for_next_turn,
    most_energy,
]


def _add_after(comparisons: List[Comparison], to_add: Comparison, after: Comparison):
    comparisons.insert(comparisons.index(after) + 1, to_add)


def _move_after(comparisons: List[Comparison], to_move: Comparison, after: Comparison):
    comparisons.remove(to_move)
    _add_after(comparisons, to_move, after)


class CommonGeneralComparator:
    def __init__(self, comparisons: List[Comparison] = None, assessment_config: ComparatorAssessmentConfig = None):
        self.comparisons: List[Comparison] = default_comparisons if comparisons is None else comparisons
        self.assessment_config: ComparatorAssessmentConfig = \
            ComparatorAssessmentConfig(powers_we_like, powers_we_like_less, powers_we_dislike,
                                       cards_that_exit_wrath=cards_that_exit_wrath) \
                if assessment_config is None else assessment_config

    def does_challenger_defeat_the_best(self, best_state: BattleState, challenger_state: BattleState,
                                        original: BattleState) -> bool:
        best = CA(best_state, original, self.assessment_config)
        challenger = CA(challenger_state, original, self.assessment_config)
        for c in self.comparisons:
            v = c(best, challenger)
            if v is not None:
                return v
        return False


def _make_big_fight_comparisons():
    c = default_comparisons.copy()
    _move_after(c, most_good_player_powers, most_dead_monsters)
    _move_after(c, least_enemy_artifacts, most_enemy_vulnerable)
    _add_after(c, avoid_inconvenient_time_warp, least_incoming_damage_over_1)
    _move_after(c, most_powered_up_ritual_dagger, most_powered_up_genetic_algorithm)
    _move_after(c, killed_with_lesson_learned, most_powered_up_genetic_algorithm)
    _add_after(c, most_crescendo, most_tranquility)
    return c


class BigFightComparator(CommonGeneralComparator):
    def __init__(self):
        super().__init__(_make_big_fight_comparisons())


def _make_gremlin_nob_comparisons():
    c = default_comparisons.copy()
    c.remove(most_free_early_draw)
    c.remove(most_free_draw)
    c.remove(least_incoming_damage_over_1)
    _add_after(c, least_nob_adjusted_scaling_damage, most_lasting_intangible)
    return c


class GremlinNobComparator(CommonGeneralComparator):
    def __init__(self):
        super().__init__(_make_gremlin_nob_comparisons())


def _make_three_sentries_comparisons():
    c = default_comparisons.copy()
    c.remove(lowest_health_monster)
    _add_after(c, lowest_health_edge_monster, most_lasting_intangible)
    _add_after(c, most_dead_edge_monsters, most_lasting_intangible)
    return c


class ThreeSentriesComparator(CommonGeneralComparator):
    def __init__(self):
        super().__init__(_make_three_sentries_comparisons())


def _make_three_sentries_turn1_comparisons():
    c = default_comparisons.copy()
    c.remove(lowest_health_monster)
    _add_after(c, lowest_health_edge_monster, most_lasting_intangible)
    _move_after(c, most_dead_monsters, most_lasting_intangible)
    return c


class ThreeSentriesTurn1Comparator(CommonGeneralComparator):
    def __init__(self):
        super().__init__(_make_three_sentries_turn1_comparisons())


def _make_transient_comparisons():
    c = default_comparisons.copy()
    c.remove(lowest_health_monster)
    c.remove(lowest_total_monster_health)
    return c


class TransientComparator(CommonGeneralComparator):
    def __init__(self):
        super().__init__(_make_transient_comparisons())


def _make_waiting_lagavulin_comparisons():
    c = default_comparisons.copy()
    c.remove(lowest_health_monster)
    c.remove(lowest_total_monster_health)
    _add_after(c, highest_health_monster, most_optimal_winning_battle)
    _move_after(c, most_good_player_powers, highest_health_monster)
    return c


class WaitingLagavulinComparator(CommonGeneralComparator):
    def __init__(self):
        super().__init__(_make_waiting_lagavulin_comparisons())
