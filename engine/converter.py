import logging

from engine.battle_state import BattleState
from engine.cards import Card
from engine.enums import CardId, CardType, OrbId, PowerId, RelicId, PotionId, Cost
from engine.entities import Player, Monster
from engine.memory import MemoryItem, TheBotsMemoryBook, ResetSchedule

log = logging.getLogger(__name__)

_CARD_IDS = {v.value for v in CardId}
_RELIC_IDS = {v.value for v in RelicId}
_POWER_IDS = {v.value for v in PowerId}
_POTION_IDS = {v.value for v in PotionId}
_ORB_IDS = {v.value for v in OrbId}


def _card_type(card) -> CardType:
    try:
        from spirecomm.spire.card import CardType as ST
        return {
            ST.ATTACK: CardType.ATTACK,
            ST.SKILL: CardType.SKILL,
            ST.POWER: CardType.POWER,
            ST.STATUS: CardType.STATUS,
            ST.CURSE: CardType.CURSE,
        }.get(card.type, CardType.SKILL)
    except Exception:
        return CardType.SKILL


def make_card(c) -> Card:
    raw = getattr(c, 'card_id', str(c)).replace('+', '').strip().lower()
    if raw not in _CARD_IDS:
        log.debug("Unknown card: %s", raw)
        return Card(CardId.FAKE, 0, Cost.unplayable, False, CardType.FAKE)
    return Card(
        card_id=CardId(raw),
        upgrade=getattr(c, 'upgrades', 0),
        cost=getattr(c, 'cost', 0),
        needs_target=getattr(c, 'has_target', False),
        card_type=_card_type(c),
        ethereal=getattr(c, 'ethereal', False),
        exhausts=getattr(c, 'exhausts', False),
        uuid=getattr(c, 'uuid', ''),
    )


def make_powers(powers) -> dict:
    out = {}
    for p in (powers or []):
        pid = getattr(p, 'power_id', '').lower()
        if pid in _POWER_IDS:
            out[PowerId(pid)] = int(getattr(p, 'amount', 0))
        if pid == 'echo form':
            misc = getattr(p, 'misc', 0)
            if misc:
                out[PowerId.INTERNAL_ECHO_FORM_READY] = misc
    return out


def make_relics(relics) -> dict:
    out = {}
    for r in (relics or []):
        rid = getattr(r, 'relic_id', '').lower()
        if rid in _RELIC_IDS:
            out[RelicId(rid)] = int(getattr(r, 'counter', -1))
        else:
            log.debug("Unknown relic: %s", rid)
    return out


def make_potions(potions) -> list:
    out = []
    for p in (potions or []):
        pid = getattr(p, 'potion_id', '').lower()
        if pid in _POTION_IDS:
            out.append(PotionId(pid))
    return out


def game_to_battle_state(game, memory: TheBotsMemoryBook) -> BattleState:
    player = game.player
    relics = make_relics(getattr(game, 'relics', []))
    potions = make_potions(getattr(game, 'potions', []))

    bs_player = Player(
        is_player=True,
        current_hp=player.current_hp,
        max_hp=player.max_hp,
        block=player.block,
        powers=make_powers(getattr(player, 'powers', [])),
        energy=player.energy,
        relics=relics,
        potions=potions,
    )

    monsters = []
    for m in (game.monsters or []):
        if m is None:
            continue
        base_dmg = getattr(m, 'move_base_damage', None)
        if base_dmg is None or base_dmg < 0:
            base_dmg = getattr(m, 'move_adjusted_damage', 0) or 0
        monsters.append(Monster(
            is_player=False,
            current_hp=m.current_hp,
            max_hp=m.max_hp,
            block=getattr(m, 'block', 0),
            powers=make_powers(getattr(m, 'powers', [])),
            damage=max(base_dmg, 0),
            hits=getattr(m, 'move_hits', 0) or 0,
            is_gone=getattr(m, 'is_gone', False),
            name=getattr(m, 'name', ''),
        ))

    memory_by_card = {
        k: {rk: dict(rv) for rk, rv in v.items()}
        for k, v in memory.memory_by_card.items()
    }

    # Forced discard/exhaust state (DiscardAction or ExhaustAction screen during combat)
    current_action = getattr(game, 'current_action', None)
    screen = getattr(game, 'screen', None)
    must_discard = False
    amount_to_discard = 0
    amount_to_exhaust = 0
    if current_action == 'DiscardAction' and screen is not None:
        amount_to_discard = getattr(screen, 'num_cards', 0) or 0
        must_discard = not getattr(screen, 'can_pick_zero', True)
    elif current_action == 'ExhaustAction' and screen is not None:
        amount_to_exhaust = getattr(screen, 'num_cards', 0) or 0

    return BattleState(
        player=bs_player,
        hand=[make_card(c) for c in getattr(game, 'hand', [])],
        discard_pile=[make_card(c) for c in getattr(game, 'discard_pile', [])],
        exhaust_pile=[make_card(c) for c in getattr(game, 'exhaust_pile', [])],
        draw_pile=[make_card(c) for c in getattr(game, 'draw_pile', [])],
        monsters=monsters,
        relics=relics,
        potions=potions,
        must_discard=must_discard,
        amount_to_discard=amount_to_discard,
        amount_to_exhaust=amount_to_exhaust,
        cards_discarded_this_turn=getattr(game, 'cards_discarded_this_turn', 0),
        memory_general=memory.memory_general.copy(),
        memory_by_card=memory_by_card,
    )


def battlestate_deepcopy(state: BattleState) -> BattleState:
    relics = dict(state.relics)
    potions = list(state.potions)

    player = Player(
        is_player=True,
        current_hp=state.player.current_hp,
        max_hp=state.player.max_hp,
        powers=dict(state.player.powers),
        block=state.player.block,
        energy=state.player.energy,
        relics=relics,
        potions=potions,
    )

    monsters = [
        Monster(
            is_player=False,
            current_hp=m.current_hp,
            max_hp=m.max_hp,
            block=m.block,
            powers=dict(m.powers),
            damage=m.damage,
            hits=m.hits,
            is_gone=m.is_gone,
            name=m.name,
        )
        for m in state.monsters
    ]

    hand = [Card(c.id, c.upgrade, c.cost, c.needs_target, c.type, c.ethereal, c.exhausts, c.uuid)
            for c in state.hand]
    draw_pile = [Card(c.id, c.upgrade, c.cost, c.needs_target, c.type, c.ethereal, c.exhausts, c.uuid)
                 for c in state.draw_pile]
    discard_pile = [Card(c.id, c.upgrade, c.cost, c.needs_target, c.type, c.ethereal, c.exhausts, c.uuid)
                    for c in state.discard_pile]
    exhaust_pile = [Card(c.id, c.upgrade, c.cost, c.needs_target, c.type, c.ethereal, c.exhausts, c.uuid)
                    for c in state.exhaust_pile]

    memory_by_card = {}
    for key, value in state.memory_by_card.items():
        for reset_key, val in value.items():
            memory_by_card[key] = {reset_key: val.copy()}

    orbs = [(OrbId(o.value), a) for o, a in state.orbs]

    new_state = BattleState(
        player=player,
        hand=hand,
        discard_pile=discard_pile,
        exhaust_pile=exhaust_pile,
        draw_pile=draw_pile,
        monsters=monsters,
        relics=relics,
        must_discard=state.must_discard,
        amount_to_discard=state.amount_to_discard,
        cards_discarded_this_turn=state.cards_discarded_this_turn,
        total_random_damage_dealt=state.total_random_damage_dealt,
        total_random_poison_added=state.total_random_poison_added,
        orbs=orbs,
        orb_slots=state.orb_slots,
        memory_general=state.memory_general.copy(),
        memory_by_card=memory_by_card,
        amount_scryed=state.amount_scryed,
        saved_block_for_next_turn=state.saved_block_for_next_turn,
        potions=potions,
        amount_to_exhaust=state.amount_to_exhaust,
    )

    new_state.draw_free_early = state.draw_free_early
    new_state.draw_free = state.draw_free
    new_state.draw_pay_early = state.draw_pay_early
    new_state.draw_pay = state.draw_pay
    new_state.time_warp_full = state.time_warp_full

    return new_state
