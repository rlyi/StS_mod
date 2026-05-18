"""Card effects + all custom hooks for BattleState simulation."""
import math
from enum import Enum
from typing import List

from engine.card_cost import Cost
from engine.enums import CardId, PowerId, RelicId, CardType
from engine.memory import MemoryItem, StanceType


class TargetType(Enum):
    NONE = 'none'
    SELF = 'self'
    MONSTER = 'monster'
    ALL_MONSTERS = 'all_monsters'
    RANDOM = 'random'


def get_x_trigger_amount(player) -> int:
    amount = player.energy
    if player.relics.get(RelicId.CHEMICAL_X):
        amount += 2
    return amount


class CardEffects:
    def __init__(
            self,
            damage: int = 0,
            hits: int = 0,
            blockable: bool = True,
            block: int = 0,
            block_times: int = 1,
            target: TargetType = TargetType.SELF,
            applies_powers: dict = None,
            energy_gain: int = 0,
            draw: int = 0,
            pre_hooks: list = None,
            post_hooks: list = None,
            post_others_discarded_hooks: list = None,
            post_self_discarded_hooks: list = None,
            end_turn_hooks: list = None,
            heal: int = 0,
            amount_to_discard: int = 0,
            amount_to_exhaust: int = 0,
            spawn_cards_in_hand=None,
            spawn_cards_in_draw=None,
            spawn_cards_in_discard=None,
            retains: bool = False,
            sets_stance: StanceType = None,
            amount_to_scry: int = 0,
            channel_orbs: list = None,
    ):
        self.damage = damage
        self.hits = hits
        self.blockable = blockable
        self.block = block
        self.block_times = block_times
        self.target = target
        self.applies_powers = {} if applies_powers is None else applies_powers
        self.energy_gain = energy_gain
        self.draw = draw
        self.pre_hooks = [] if pre_hooks is None else pre_hooks
        self.post_hooks = [] if post_hooks is None else post_hooks
        self.post_others_discarded_hooks = [] if post_others_discarded_hooks is None else post_others_discarded_hooks
        self.post_self_discarded_hooks = [] if post_self_discarded_hooks is None else post_self_discarded_hooks
        self.end_turn_hooks = [] if end_turn_hooks is None else end_turn_hooks
        self.heal = heal
        self.amount_to_discard = amount_to_discard
        self.amount_to_exhaust = amount_to_exhaust
        self.spawn_cards_in_hand = spawn_cards_in_hand
        self.spawn_cards_in_draw = spawn_cards_in_draw
        self.spawn_cards_in_discard = spawn_cards_in_discard
        self.retains = retains
        self.sets_stance = sets_stance
        self.amount_to_scry = amount_to_scry
        self.channel_orbs = [] if channel_orbs is None else channel_orbs
        self.hp_damage = 0


# ── Custom hooks ─────────────────────────────────────────────────────────────

def dropkick_post_hook(state, effect, card, target_index=-1):
    if target_index > -1 and state.monsters[target_index].powers.get(PowerId.VULNERABLE):
        state.player.energy += 1
        state.draw_cards(1)


def entrench_post_hook(state, effect, card, target_index=-1):
    state.add_player_block(state.player.block)


def feed_post_hook(state, effect, card, target_index=-1):
    amount = 3 if not card.upgrade else 4
    alive = len([True for m in state.monsters if m.current_hp > 0])
    life_link = state.monsters[target_index].powers.get(PowerId.LIFE_LINK) and alive > 0
    if state.monsters[target_index].current_hp <= 0 and \
            not state.monsters[target_index].powers.get(PowerId.MINION) and not life_link:
        state.player.max_hp += amount
        state.player.current_hp += amount


def fiend_fire_pre_hook(state, effect, card, target_index=-1):
    effect.hits = len(state.hand) - 1


def fiend_fire_post_hook(state, effect, card, target_index=-1):
    for _ in range(len(state.hand)):
        state.exhaust_card(state.hand[0])


def limit_break_post_hook(state, effect, card, target_index=-1):
    if state.player.powers.get(PowerId.STRENGTH):
        state.player.powers[PowerId.STRENGTH] *= 2


def spot_weakness_post_hook(state, effect, card, target_index=-1):
    amount = 3 if not card.upgrade else 4
    if state.monsters[target_index].hits and state.monsters[target_index].damage != -1:
        state.player.add_powers({PowerId.STRENGTH: amount}, state.player.relics, state.player.powers)


def apotheosis_post_hook(state, effect, card, target_index=-1):
    for pile in [state.hand, state.draw_pile, state.discard_pile, state.exhaust_pile]:
        for c in pile:
            if c.id != CardId.BURN:
                c.upgrade += 1


def heel_hook_post_hook(state, effect, card, target_index=-1):
    if target_index > -1 and state.monsters[target_index].powers.get(PowerId.WEAKENED):
        state.player.energy += 1
        state.draw_cards(1)


def storm_of_steel_post_hook(state, effect, card, target_index=-1):
    from engine.cards import get_card
    amount = len(state.hand)
    for _ in range(amount):
        state.discard_card(state.hand[0])
    state.spawn_in_hand(get_card(CardId.SHIV), amount)


def storm_of_steel_upgraded_post_hook(state, effect, card, target_index=-1):
    from engine.cards import get_card
    amount = len(state.hand)
    for _ in range(amount):
        state.discard_card(state.hand[0])
    state.spawn_in_hand(get_card(CardId.SHIV, upgrade=1), amount)


def eviscerate_post_others_discarded_hook(card):
    card.cost = max(0, card.cost - 1)


def sneaky_strike_post_hook(state, effect, card, target_index=-1):
    if state.cards_discarded_this_turn:
        state.player.energy += 2


def unload_post_hook(state, effect, card, target_index=-1):
    for idx in reversed(range(len(state.hand))):
        if state.hand[idx].type != CardType.ATTACK:
            state.discard_card(state.hand[idx])


def tactician_post_self_discarded_hook(state, effect, card, target_index=-1):
    state.player.energy += 1


def tactician_upgraded_post_self_discarded_hook(state, effect, card, target_index=-1):
    state.player.energy += 2


def reflex_post_self_discarded_hook(state, effect, card, target_index=-1):
    state.draw_cards(2)


def reflex_upgraded_post_self_discarded_hook(state, effect, card, target_index=-1):
    state.draw_cards(3)


def bane_pre_hook(state, effect, card, target_index=-1):
    if target_index > -1 and state.monsters[target_index].powers.get(PowerId.POISON):
        effect.hits = 2


def bullet_time_post_hook(state, effect, card, target_index=-1):
    for c in state.hand:
        if c.cost != Cost.unplayable:
            c.cost = 0


def catalyst_post_hook(state, effect, card, target_index=-1):
    if target_index > -1 and state.monsters[target_index].powers.get(PowerId.POISON):
        base = state.monsters[target_index].powers[PowerId.POISON]
        state.monsters[target_index].add_powers({PowerId.POISON: base}, state.player.relics, state.player.powers)


def catalyst_upgraded_post_hook(state, effect, card, target_index=-1):
    if target_index > -1 and state.monsters[target_index].powers.get(PowerId.POISON):
        base = state.monsters[target_index].powers[PowerId.POISON]
        state.monsters[target_index].add_powers({PowerId.POISON: base * 2}, state.player.relics, state.player.powers)


def bouncing_flask_post_hook(state, effect, card, target_index=-1):
    hits = 3 if not card.upgrade else 4
    state.add_random_poison(3, hits)


def deep_breath_pre_hook(state, effect, card, target_index=-1):
    state.draw_pile.extend(state.discard_pile)
    state.discard_pile.clear()


def enlightenment_post_hook(state, effect, card, target_index=-1):
    for c in state.hand:
        if c.cost >= 2:
            c.cost = 1


def impatience_post_hook(state, effect, card, target_index=-1):
    attacks = len([True for c in state.hand if c.type == CardType.ATTACK])
    draw = 2 if not card.upgrade else 3
    if attacks == 0:
        state.draw_cards(draw)


def stack_pre_hook(state, effect, card, target_index=-1):
    basic = 0 if not card.upgrade else 3
    effect.block = len(state.discard_pile) + basic


def mind_blast_pre_hook(state, effect, card, target_index=-1):
    effect.damage = len(state.draw_pile)


def auto_shields_pre_hook(state, effect, card, target_index=-1):
    block = 11 if not card.upgrade else 15
    if state.player.block == 0:
        effect.block = block


def aggregate_post_hook(state, effect, card, target_index=-1):
    state.player.energy += math.floor(len(state.draw_pile) / 4)


def aggregate_upgraded_post_hook(state, effect, card, target_index=-1):
    state.player.energy += math.floor(len(state.draw_pile) / 3)


def double_energy_post_hook(state, effect, card, target_index=-1):
    state.player.energy *= 2


def sunder_post_hook(state, effect, card, target_index=-1):
    if state.monsters[target_index].current_hp <= 0:
        state.player.energy += 3


def melter_pre_hook(state, effect, card, target_index=-1):
    state.monsters[target_index].block = 0


def calculated_gamble_post_hook(state, effect, card, target_index=-1):
    amount = len(state.hand)
    for _ in range(amount):
        state.discard_card(state.hand[0])
    state.draw_cards(amount)


def all_for_one_post_hook(state, effect, card, target_index=-1):
    hand_space = 10 - len(state.hand)
    retrieval, rest = [], []
    for c in state.discard_pile:
        if c.cost == 0 and len(retrieval) < hand_space:
            retrieval.append(c)
        else:
            rest.append(c)
    state.hand.extend(retrieval)
    state.discard_pile = rest


def decay_end_turn_hook(state, effect, card, target_index=-1):
    state.player.inflict_damage(state.player, 2, 1, vulnerable_modifier=1, is_attack=False)


def doubt_end_turn_hook(state, effect, card, target_index=-1):
    state.player.add_powers({PowerId.WEAKENED: 1}, state.player.relics, state.player.powers)


def shame_end_turn_hook(state, effect, card, target_index=-1):
    state.player.add_powers({PowerId.FRAIL: 1}, state.player.relics, state.player.powers)


def burn_end_turn_hook(state, effect, card, target_index=-1):
    state.player.inflict_damage(state.player, 2, 1, vulnerable_modifier=1, is_attack=False)


def burn_upgraded_end_turn_hook(state, effect, card, target_index=-1):
    state.player.inflict_damage(state.player, 4, 1, vulnerable_modifier=1, is_attack=False)


def regret_end_turn_hook(state, effect, card, target_index=-1):
    state.player.inflict_damage(state.player, len(state.hand), 1,
                                blockable=False, vulnerable_modifier=1, is_attack=False)


def bowling_bash_pre_hook(state, effect, card, target_index=-1):
    effect.hits = len([True for m in state.monsters if m.current_hp > 0])


def conclude_post_hook(state, effect, card, target_index=-1):
    state.end_turn()


def sever_soul_post_hook(state, effect, card, target_index=-1):
    exhaust, keep = [], []
    for c in state.hand:
        (keep if c.type == CardType.ATTACK else exhaust).append(c)
    for c in exhaust:
        state.exhaust_card(c, handle_remove=False)
    state.hand = keep


def second_wind_post_hook(state, effect, card, target_index=-1):
    block = 5 if not card.upgrade else 7
    exhaust, keep = [], []
    for c in state.hand:
        (keep if c.type == CardType.ATTACK else exhaust).append(c)
    times = len(exhaust)
    for c in exhaust:
        state.exhaust_card(c, handle_remove=False)
    for _ in range(times):
        state.add_player_block(block + state.player.powers.get(PowerId.DEXTERITY, 0))
    state.hand = keep


def ritual_dagger_pre_hook(state, effect, card, target_index=-1):
    effect.damage = 15 + state.get_memory_by_card(card.id, card.uuid)


def ritual_dagger_post_hook(state, effect, card, target_index=-1):
    alive = len([True for m in state.monsters if m.current_hp > 0])
    life_link = state.monsters[target_index].powers.get(PowerId.LIFE_LINK) and alive > 0
    if state.monsters[target_index].current_hp <= 0 and \
            not state.monsters[target_index].powers.get(PowerId.MINION) and not life_link:
        state.add_memory_by_card(card.id, card.uuid, 3 if not card.upgrade else 5)


def finisher_pre_hook(state, effect, card, target_index=-1):
    effect.hits = state.get_memory_value(MemoryItem.ATTACKS_THIS_TURN)


def claw_pre_hook(state, effect, card, target_index=-1):
    base = 3 if not card.upgrade else 5
    effect.damage = base + 2 * state.get_memory_value(MemoryItem.CLAWS_THIS_BATTLE)


def claw_post_hook(state, effect, card, target_index=-1):
    state.add_memory_value(MemoryItem.CLAWS_THIS_BATTLE, 1)


def genetic_algorithm_pre_hook(state, effect, card, target_index=-1):
    effect.block = 1 + state.get_memory_by_card(card.id, card.uuid)


def genetic_algorithm_post_hook(state, effect, card, target_index=-1):
    state.add_memory_by_card(card.id, card.uuid, 2 if not card.upgrade else 3)


def steam_barrier_pre_hook(state, effect, card, target_index=-1):
    base = 6 if not card.upgrade else 8
    effect.block = base - state.get_memory_by_card(card.id, card.uuid)


def steam_barrier_post_hook(state, effect, card, target_index=-1):
    state.add_memory_by_card(card.id, card.uuid, 1)


def glass_knife_pre_hook(state, effect, card, target_index=-1):
    base = 8 if not card.upgrade else 12
    effect.damage = base - state.get_memory_by_card(card.id, card.uuid)


def glass_knife_post_hook(state, effect, card, target_index=-1):
    state.add_memory_by_card(card.id, card.uuid, 2)


def streamline_post_hook(state, effect, card, target_index=-1):
    card.cost = max(0, card.cost - 1)


def ftl_pre_hook(state, effect, card, target_index=-1):
    threshold = 3 if not card.upgrade else 4
    if state.get_memory_value(MemoryItem.CARDS_THIS_TURN) < threshold:
        state.draw_cards(1)


def rampage_pre_hook(state, effect, card, target_index=-1):
    effect.damage = 8 + state.get_memory_by_card(card.id, card.uuid)


def rampage_post_hook(state, effect, card, target_index=-1):
    state.add_memory_by_card(card.id, card.uuid, 5 if not card.upgrade else 8)


def blizzard_pre_hook(state, effect, card, target_index=-1):
    effect.damage = 2 * state.get_memory_value(MemoryItem.FROST_THIS_BATTLE)


def thunder_strike_pre_hook(state, effect, card, target_index=-1):
    cracked = 0 if RelicId.CRACKED_CORE not in state.relics else 1
    effect.hits = state.get_memory_value(MemoryItem.LIGHTNING_THIS_BATTLE) + cracked


def judgement_post_hook(state, effect, card, target_index=-1):
    threshold = 30 if not card.upgrade else 40
    if state.monsters[target_index].current_hp <= threshold:
        state.monsters[target_index].current_hp = 0


def crush_joints_pre_hook(state, effect, card, target_index=-1):
    amount = 1 if not card.upgrade else 2
    if state.get_memory_value(MemoryItem.TYPE_LAST_PLAYED) is CardType.SKILL:
        effect.applies_powers.update({PowerId.VULNERABLE: amount})


def sash_whip_pre_hook(state, effect, card, target_index=-1):
    amount = 1 if not card.upgrade else 2
    if state.get_memory_value(MemoryItem.TYPE_LAST_PLAYED) is CardType.ATTACK:
        effect.applies_powers.update({PowerId.WEAKENED: amount})


def follow_up_pre_hook(state, effect, card, target_index=-1):
    if state.get_memory_value(MemoryItem.TYPE_LAST_PLAYED) is CardType.ATTACK:
        state.player.energy += 1


def sanctity_pre_hook(state, effect, card, target_index=-1):
    if state.get_memory_value(MemoryItem.TYPE_LAST_PLAYED) is CardType.SKILL:
        effect.draw = 2


def tantrum_post_hook(state, effect, card, target_index=-1):
    from engine.cards import get_card
    c = get_card(CardId.TANTRUM)
    c.upgrade = card.upgrade
    state.draw_pile.append(c)


def inner_peace_post_hook(state, effect, card, target_index=-1):
    if state.get_stance() == StanceType.CALM:
        state.draw_cards(3 if not card.upgrade else 4)
    else:
        effect.sets_stance = StanceType.CALM


def indignation_post_hook(state, effect, card, target_index=-1):
    if state.get_stance() == StanceType.WRATH:
        amount = 3 if not card.upgrade else 5
        for m in state.monsters:
            m.add_powers({PowerId.VULNERABLE: amount}, state.relics, state.player.powers)
    else:
        effect.sets_stance = StanceType.WRATH


def fear_no_evil_post_hook(state, effect, card, target_index=-1):
    if state.monsters[target_index].hits and state.monsters[target_index].damage != -1:
        state.change_stance(StanceType.CALM)


def halt_pre_hook(state, effect, card, target_index=-1):
    if state.get_stance() is StanceType.WRATH:
        effect.block += 9 if not card.upgrade else 14


def perseverance_pre_hook(state, effect, card, target_index=-1):
    base = 5 if not card.upgrade else 7
    effect.block = base + state.get_memory_by_card(card.id, card.uuid)


def spirit_shield_pre_hook(state, effect, card, target_index=-1):
    mult = 3 if not card.upgrade else 4
    effect.block = (len(state.hand) - 1) * mult


def windmill_strike_pre_hook(state, effect, card, target_index=-1):
    base = 7 if not card.upgrade else 10
    effect.damage = base + state.get_memory_by_card(card.id, card.uuid)


def pressure_points_post_hook(state, effect, card, target_index=-1):
    for m in state.monsters:
        if PowerId.MARK in m.powers:
            m.inflict_damage(m, m.powers.get(PowerId.MARK, 0), 1,
                             blockable=False, vulnerable_modifier=1, is_attack=False)


def brilliance_pre_hook(state, effect, card, target_index=-1):
    base = 12 if not card.upgrade else 16
    effect.damage = base + state.get_memory_value(MemoryItem.MANTRA_THIS_BATTLE)


def lesson_learned_post_hook(state, effect, card, target_index=-1):
    alive = len([True for m in state.monsters if m.current_hp > 0])
    life_link = state.monsters[target_index].powers.get(PowerId.LIFE_LINK) and alive > 0
    if state.monsters[target_index].current_hp <= 0 and \
            not state.monsters[target_index].powers.get(PowerId.MINION) and not life_link:
        state.add_memory_value(MemoryItem.KILLED_WITH_LESSON_LEARNED, 1)


def panache_post_hook(state, effect, card, target_index=-1):
    power_amount = 10 if not card.upgrade else 14
    state.add_memory_value(MemoryItem.PANACHE_DAMAGE, power_amount)


def recycle_post_hook(state, effect, card, target_index=-1):
    state.add_memory_value(MemoryItem.RECYCLE, 1)


def reboot_post_hook(state, effect, card, target_index=-1):
    amount_to_draw = 4 if not card.upgrade else 6
    amount = len(state.hand)
    for _ in range(amount):
        state.discard_card(state.hand[0])
    state.discard_pile.extend(state.draw_pile)
    state.draw_pile.clear()
    state.draw_cards(amount_to_draw)


# ── get_card_effects ──────────────────────────────────────────────────────────

def get_card_effects(card, player, draw_pile: list, discard_pile: list, hand: list) -> List[CardEffects]:
    from engine.cards import get_card as _gc

    if card.id in (CardId.STRIKE_R, CardId.STRIKE_G, CardId.STRIKE_B, CardId.STRIKE_P):
        return [CardEffects(damage=6 if not card.upgrade else 9, hits=1, target=TargetType.MONSTER)]
    if card.id in (CardId.DEFEND_R, CardId.DEFEND_G, CardId.DEFEND_B, CardId.DEFEND_P):
        return [CardEffects(block=5 if not card.upgrade else 8, target=TargetType.SELF)]
    if card.id == CardId.BASH:
        return [CardEffects(damage=8 if not card.upgrade else 10, hits=1, target=TargetType.MONSTER,
                            applies_powers={PowerId.VULNERABLE: 2 if not card.upgrade else 3})]
    if card.id == CardId.ANGER:
        return [CardEffects(damage=6 if not card.upgrade else 8, hits=1, target=TargetType.MONSTER)]
    if card.id == CardId.CLEAVE:
        return [CardEffects(damage=8 if not card.upgrade else 11, hits=1, target=TargetType.ALL_MONSTERS)]
    if card.id == CardId.CLOTHESLINE:
        return [CardEffects(damage=12 if not card.upgrade else 14, hits=1, target=TargetType.MONSTER,
                            applies_powers={PowerId.WEAKENED: 2 if not card.upgrade else 3})]
    if card.id == CardId.HEAVY_BLADE:
        str_bonus = player.powers.get(PowerId.STRENGTH, 0)
        damage = 12 + str_bonus * (2 if not card.upgrade else 4)
        return [CardEffects(damage=damage, hits=1, target=TargetType.MONSTER)]
    if card.id == CardId.IRON_WAVE:
        amount = 5 if not card.upgrade else 7
        return [CardEffects(damage=amount, hits=1, block=amount, target=TargetType.MONSTER)]
    if card.id == CardId.PERFECTED_STRIKE:
        strike_count = len([1 for c in discard_pile + draw_pile + hand if "strike" in c.id.value])
        damage = 6 + strike_count * (2 if not card.upgrade else 3)
        return [CardEffects(damage=damage, hits=1, target=TargetType.MONSTER)]
    if card.id == CardId.POMMEL_STRIKE:
        return [CardEffects(damage=9 if not card.upgrade else 10, hits=1, draw=1, target=TargetType.MONSTER)]
    if card.id == CardId.SHRUG_IT_OFF:
        return [CardEffects(block=8 if not card.upgrade else 11, draw=1, target=TargetType.SELF)]
    if card.id == CardId.WHIRLWIND:
        base = 5 if not card.upgrade else 8
        return [CardEffects(damage=base, hits=get_x_trigger_amount(player), target=TargetType.ALL_MONSTERS)]
    if card.id == CardId.THUNDERCLAP:
        return [CardEffects(damage=4 if not card.upgrade else 6, hits=1, target=TargetType.ALL_MONSTERS,
                            applies_powers={PowerId.VULNERABLE: 1})]
    if card.id == CardId.TWIN_STRIKE:
        return [CardEffects(damage=5 if not card.upgrade else 7, hits=2, target=TargetType.MONSTER)]
    if card.id == CardId.BLOOD_FOR_BLOOD:
        return [CardEffects(damage=18 if not card.upgrade else 22, hits=1, target=TargetType.MONSTER)]
    if card.id == CardId.BLOODLETTING:
        return [CardEffects(energy_gain=2 if not card.upgrade else 3, damage=3, hits=1,
                            blockable=False, target=TargetType.SELF)]
    if card.id == CardId.CARNAGE:
        return [CardEffects(damage=20 if not card.upgrade else 28, hits=1, target=TargetType.MONSTER)]
    if card.id == CardId.UPPERCUT:
        powers = {PowerId.WEAKENED: 1, PowerId.VULNERABLE: 1} if not card.upgrade \
            else {PowerId.WEAKENED: 2, PowerId.VULNERABLE: 2}
        return [CardEffects(damage=13, hits=1, target=TargetType.MONSTER, applies_powers=powers)]
    if card.id == CardId.DISARM:
        return [CardEffects(target=TargetType.MONSTER,
                            applies_powers={PowerId.STRENGTH: -2 if not card.upgrade else -3})]
    if card.id == CardId.DROPKICK:
        return [CardEffects(damage=5 if not card.upgrade else 8, hits=1, target=TargetType.MONSTER,
                            post_hooks=[dropkick_post_hook])]
    if card.id == CardId.ENTRENCH:
        return [CardEffects(target=TargetType.SELF, post_hooks=[entrench_post_hook])]
    if card.id == CardId.FLAME_BARRIER:
        return [CardEffects(target=TargetType.SELF, block=12 if not card.upgrade else 16,
                            applies_powers={PowerId.FLAME_BARRIER: 4 if not card.upgrade else 6})]
    if card.id == CardId.GHOSTLY_ARMOR:
        return [CardEffects(target=TargetType.SELF, block=10 if not card.upgrade else 13)]
    if card.id == CardId.HEMOKINESIS:
        return [CardEffects(damage=15 if not card.upgrade else 20, hits=1, target=TargetType.MONSTER),
                CardEffects(damage=2, hits=1, blockable=False, target=TargetType.SELF)]
    if card.id == CardId.INFLAME:
        return [CardEffects(target=TargetType.SELF, applies_powers={PowerId.STRENGTH: 2 if not card.upgrade else 3})]
    if card.id == CardId.FIRE_BREATHING:
        return [CardEffects(target=TargetType.SELF,
                            applies_powers={PowerId.FIRE_BREATHING: 6 if not card.upgrade else 10})]
    if card.id == CardId.EVOLVE:
        return [CardEffects(target=TargetType.SELF, applies_powers={PowerId.EVOLVE: 1 if not card.upgrade else 2})]
    if card.id == CardId.DEMON_FORM:
        return [CardEffects(target=TargetType.SELF, applies_powers={PowerId.DEMON_FORM: 2 if not card.upgrade else 3})]
    if card.id == CardId.BERSERK:
        return [CardEffects(target=TargetType.SELF,
                            applies_powers={PowerId.VULNERABLE: 2 if not card.upgrade else 1, PowerId.BERSERK: 1})]
    if card.id == CardId.INTIMIDATE:
        return [CardEffects(target=TargetType.ALL_MONSTERS,
                            applies_powers={PowerId.WEAKENED: 1 if not card.upgrade else 2})]
    if card.id == CardId.PUMMEL:
        return [CardEffects(damage=2, hits=4 if not card.upgrade else 5, target=TargetType.MONSTER)]
    if card.id == CardId.SEEING_RED:
        return [CardEffects(energy_gain=2, target=TargetType.SELF)]
    if card.id == CardId.SHOCKWAVE:
        amount = 3 if not card.upgrade else 5
        return [CardEffects(target=TargetType.ALL_MONSTERS,
                            applies_powers={PowerId.WEAKENED: amount, PowerId.VULNERABLE: amount})]
    if card.id == CardId.BLUDGEON:
        return [CardEffects(target=TargetType.MONSTER, damage=32 if not card.upgrade else 42, hits=1)]
    if card.id == CardId.FEED:
        return [CardEffects(target=TargetType.MONSTER, damage=10 if not card.upgrade else 12, hits=1,
                            post_hooks=[feed_post_hook])]
    if card.id == CardId.FIEND_FIRE:
        return [CardEffects(target=TargetType.MONSTER, damage=7 if not card.upgrade else 10, hits=1,
                            pre_hooks=[fiend_fire_pre_hook], post_hooks=[fiend_fire_post_hook])]
    if card.id == CardId.IMMOLATE:
        return [CardEffects(target=TargetType.ALL_MONSTERS, damage=21 if not card.upgrade else 28, hits=1,
                            spawn_cards_in_discard=(_gc(CardId.BURN), 1))]
    if card.id == CardId.IMPERVIOUS:
        return [CardEffects(target=TargetType.SELF, block=30 if not card.upgrade else 40)]
    if card.id == CardId.LIMIT_BREAK:
        return [CardEffects(target=TargetType.SELF, post_hooks=[limit_break_post_hook])]
    if card.id == CardId.OFFERING:
        return [CardEffects(target=TargetType.SELF, damage=6, hits=1, blockable=False,
                            draw=3 if not card.upgrade else 5, energy_gain=2)]
    if card.id == CardId.JAX:
        return [CardEffects(target=TargetType.SELF, damage=3, hits=1, blockable=False,
                            applies_powers={PowerId.STRENGTH: 2 if not card.upgrade else 3})]
    if card.id == CardId.BODY_SLAM:
        return [CardEffects(target=TargetType.MONSTER, damage=player.block, hits=1)]
    if card.id == CardId.CLASH:
        return [CardEffects(target=TargetType.MONSTER, damage=14 if not card.upgrade else 18, hits=1)]
    if card.id == CardId.FLEX:
        return [CardEffects(target=TargetType.SELF, applies_powers={PowerId.STRENGTH: 2 if not card.upgrade else 4})]
    if card.id == CardId.WILD_STRIKE:
        return [CardEffects(target=TargetType.MONSTER, damage=12 if not card.upgrade else 18, hits=1,
                            spawn_cards_in_draw=(_gc(CardId.WOUND), 1))]
    if card.id == CardId.BATTLE_TRANCE:
        return [CardEffects(target=TargetType.SELF, draw=3 if not card.upgrade else 5),
                CardEffects(target=TargetType.SELF, applies_powers={PowerId.NO_DRAW: 1})]
    if card.id == CardId.RAGE:
        return [CardEffects(target=TargetType.SELF, applies_powers={PowerId.RAGE: 3 if not card.upgrade else 5})]
    if card.id == CardId.RAMPAGE:
        return [CardEffects(target=TargetType.MONSTER, damage=8, hits=1,
                            pre_hooks=[rampage_pre_hook], post_hooks=[rampage_post_hook])]
    if card.id == CardId.SWORD_BOOMERANG:
        hits = 3 if not card.upgrade else 4
        return [CardEffects(target=TargetType.RANDOM, hits=hits, damage=3)]
    if card.id == CardId.JUGGERNAUT:
        return [CardEffects(target=TargetType.SELF, applies_powers={PowerId.JUGGERNAUT: 5 if not card.upgrade else 7})]
    if card.id == CardId.METALLICIZE:
        return [CardEffects(target=TargetType.SELF, applies_powers={PowerId.METALLICIZE: 3 if not card.upgrade else 4})]
    if card.id == CardId.RECKLESS_CHARGE:
        return [CardEffects(target=TargetType.MONSTER, damage=7 if not card.upgrade else 10, hits=1,
                            spawn_cards_in_draw=(_gc(CardId.DAZED), 1))]
    if card.id == CardId.POWER_THROUGH:
        return [CardEffects(block=15 if not card.upgrade else 20, target=TargetType.SELF,
                            spawn_cards_in_hand=(_gc(CardId.WOUND), 2))]
    if card.id == CardId.SPOT_WEAKNESS:
        return [CardEffects(target=TargetType.MONSTER, post_hooks=[spot_weakness_post_hook])]
    if card.id == CardId.REAPER:
        return [CardEffects(target=TargetType.ALL_MONSTERS, damage=4 if not card.upgrade else 5, hits=1)]
    if card.id == CardId.SEVER_SOUL:
        return [CardEffects(damage=16 if not card.upgrade else 22, hits=1, target=TargetType.MONSTER,
                            post_hooks=[sever_soul_post_hook])]
    if card.id == CardId.SECOND_WIND:
        return [CardEffects(target=TargetType.SELF, post_hooks=[second_wind_post_hook])]
    if card.id == CardId.BARRICADE:
        return [CardEffects(target=TargetType.SELF, applies_powers={PowerId.BARRICADE: 1})]
    if card.id == CardId.BURNING_PACT:
        return [CardEffects(target=TargetType.SELF, draw=2 if not card.upgrade else 3, amount_to_exhaust=1)]
    if card.id == CardId.DOUBLE_TAP:
        return [CardEffects(target=TargetType.SELF,
                            applies_powers={PowerId.DOUBLE_TAP: 1 if not card.upgrade else 2})]
    if card.id == CardId.CHOKE:
        return [CardEffects(damage=12, hits=1, target=TargetType.MONSTER,
                            applies_powers={PowerId.CHOKED: 3 if not card.upgrade else 5})]

    # ── Colorless / multi-class ──────────────────────────────────────────
    if card.id == CardId.WOUND:
        return [CardEffects(target=TargetType.NONE)]
    if card.id == CardId.VOID:
        return [CardEffects(target=TargetType.NONE)]
    if card.id == CardId.BURN:
        hook = burn_end_turn_hook if not card.upgrade else burn_upgraded_end_turn_hook
        return [CardEffects(target=TargetType.NONE, end_turn_hooks=[hook])]
    if card.id == CardId.SLIMED:
        return [CardEffects(target=TargetType.NONE)]
    if card.id == CardId.DECAY:
        return [CardEffects(target=TargetType.NONE, end_turn_hooks=[decay_end_turn_hook])]
    if card.id == CardId.REGRET:
        return [CardEffects(target=TargetType.NONE, end_turn_hooks=[regret_end_turn_hook])]
    if card.id == CardId.SHAME:
        return [CardEffects(target=TargetType.NONE, end_turn_hooks=[shame_end_turn_hook])]
    if card.id == CardId.DOUBT:
        return [CardEffects(target=TargetType.NONE, end_turn_hooks=[doubt_end_turn_hook])]
    if card.id in (CardId.CURSE_OF_THE_BELL, CardId.NECRONOMICURSE, CardId.PARASITE,
                   CardId.INJURY, CardId.CLUMSY, CardId.ASCENDERS_BANE, CardId.NORMALITY):
        return [CardEffects(target=TargetType.NONE)]
    if card.id == CardId.BANDAGE_UP:
        return [CardEffects(target=TargetType.SELF, heal=4 if not card.upgrade else 6)]
    if card.id == CardId.BITE:
        return [CardEffects(damage=7 if not card.upgrade else 8, hits=1, target=TargetType.MONSTER),
                CardEffects(heal=2 if not card.upgrade else 3, target=TargetType.SELF)]
    if card.id == CardId.DARK_SHACKLES:
        return [CardEffects(target=TargetType.MONSTER,
                            applies_powers={PowerId.STRENGTH: -9 if not card.upgrade else -15})]
    if card.id == CardId.FLASH_OF_STEEL:
        return [CardEffects(target=TargetType.MONSTER, damage=3 if not card.upgrade else 6, hits=1, draw=1)]
    if card.id == CardId.SWIFT_STRIKE:
        return [CardEffects(target=TargetType.MONSTER, damage=7 if not card.upgrade else 10, hits=1, draw=1)]
    if card.id == CardId.TRIP:
        return [CardEffects(target=TargetType.MONSTER if not card.upgrade else TargetType.ALL_MONSTERS,
                            applies_powers={PowerId.VULNERABLE: 2})]
    if card.id == CardId.BLIND:
        return [CardEffects(target=TargetType.MONSTER if not card.upgrade else TargetType.ALL_MONSTERS,
                            applies_powers={PowerId.WEAKENED: 2})]
    if card.id == CardId.APOTHEOSIS:
        return [CardEffects(target=TargetType.SELF, post_hooks=[apotheosis_post_hook])]
    if card.id == CardId.HAND_OF_GREED:
        return [CardEffects(target=TargetType.MONSTER, damage=20 if not card.upgrade else 25, hits=1)]
    if card.id == CardId.MASTER_OF_STRATEGY:
        return [CardEffects(target=TargetType.SELF, draw=3 if not card.upgrade else 4)]
    if card.id == CardId.APPARITION:
        return [CardEffects(target=TargetType.SELF, applies_powers={PowerId.INTANGIBLE_PLAYER: 1})]
    if card.id == CardId.DEEP_BREATH:
        return [CardEffects(target=TargetType.SELF, draw=1 if not card.upgrade else 2,
                            pre_hooks=[deep_breath_pre_hook])]
    if card.id == CardId.RITUAL_DAGGER:
        return [CardEffects(hits=1, target=TargetType.MONSTER,
                            pre_hooks=[ritual_dagger_pre_hook], post_hooks=[ritual_dagger_post_hook])]
    if card.id == CardId.ENLIGHTENMENT:
        return [CardEffects(target=TargetType.SELF, post_hooks=[enlightenment_post_hook])]
    if card.id == CardId.IMPATIENCE:
        return [CardEffects(target=TargetType.SELF, post_hooks=[impatience_post_hook])]
    if card.id == CardId.MAYHEM:
        return [CardEffects(target=TargetType.SELF, applies_powers={PowerId.MAYHEM: 1})]
    if card.id == CardId.PANACHE:
        return [CardEffects(target=TargetType.SELF, applies_powers={PowerId.PANACHE_INTERNAL: 1},
                            post_hooks=[panache_post_hook])]
    if card.id == CardId.SADISTIC_NATURE:
        return [CardEffects(target=TargetType.SELF, applies_powers={PowerId.SADISTIC: 5 if not card.upgrade else 7})]

    # Watcher / other cross-class spawns
    if card.id == CardId.SENTINEL:
        return [CardEffects(block=5 if not card.upgrade else 8, target=TargetType.SELF)]
    if card.id == CardId.FEEL_NO_PAIN:
        return [CardEffects(target=TargetType.SELF, applies_powers={PowerId.FEEL_NO_PAIN: 3 if not card.upgrade else 4})]
    if card.id == CardId.DARK_EMBRACE:
        return [CardEffects(target=TargetType.SELF, applies_powers={PowerId.DARK_EMBRACE: 1})]
    if card.id == CardId.CORRUPTION:
        return [CardEffects(target=TargetType.SELF, applies_powers={PowerId.CORRUPTION: 1})]
    if card.id == CardId.SMITE:
        return [CardEffects(damage=12 if not card.upgrade else 16, hits=1, target=TargetType.MONSTER, retains=True)]
    if card.id == CardId.INSIGHT:
        return [CardEffects(draw=2 if not card.upgrade else 3, retains=True)]
    if card.id == CardId.SAFETY:
        return [CardEffects(target=TargetType.SELF, block=12 if not card.upgrade else 16, retains=True)]
    if card.id == CardId.THROUGH_VIOLENCE:
        return [CardEffects(damage=20 if not card.upgrade else 30, hits=1, target=TargetType.MONSTER, retains=True)]
    if card.id == CardId.OMEGA:
        return [CardEffects(target=TargetType.SELF, applies_powers={PowerId.OMEGA: 50 if not card.upgrade else 60})]
    if card.id == CardId.TANTRUM:
        return [CardEffects(target=TargetType.MONSTER, damage=3, hits=3 if not card.upgrade else 4,
                            sets_stance=StanceType.WRATH, post_hooks=[tantrum_post_hook])]
    if card.id == CardId.FLURRY_OF_BLOWS:
        return [CardEffects(target=TargetType.MONSTER, damage=4 if not card.upgrade else 6, hits=1)]
    if card.id == CardId.WEAVE:
        return [CardEffects(target=TargetType.MONSTER, damage=4 if not card.upgrade else 6, hits=1)]
    if card.id == CardId.FINESSE:
        return [CardEffects(block=2 if not card.upgrade else 4, draw=1, target=TargetType.SELF)]
    if card.id == CardId.DRAMATIC_ENTRANCE:
        return [CardEffects(damage=8 if not card.upgrade else 12, hits=1, target=TargetType.ALL_MONSTERS)]

    return [CardEffects()]
