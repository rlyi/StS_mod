import logging

from spirecomm.communication.action import PlayCardAction, EndTurnAction, CardSelectAction, PotionAction

from engine.battle_state import BattleState, PLAY_DISCARD, PLAY_EXHAUST
from engine.converter import game_to_battle_state, battlestate_deepcopy
from engine.memory import MemoryItem, TheBotsMemoryBook
from engine.play_path import get_paths_bfs
from engine.comparators import (
    CommonGeneralComparator, BigFightComparator, GremlinNobComparator,
    ThreeSentriesComparator, ThreeSentriesTurn1Comparator,
    TransientComparator, WaitingLagavulinComparator,
)

log = logging.getLogger("GraphBattleAgent")

MAX_PATH_COUNT = 11_000

_DONT_USE_POTIONS = {
    'fairy in a bottle', 'smoke bomb', 'elixir potion',
    'liquid memories', 'snecko oil',
}


def _try_use_potion(game, turn: int):
    player = game.player
    if not player:
        return None
    hp_pct = player.current_hp / max(player.max_hp, 1) * 100
    room_type = str(getattr(game, 'room_type', '')).upper()
    monsters = [m for m in (game.monsters or []) if m is not None]

    is_elite = 'ELITE' in room_type
    is_boss = 'BOSS' in room_type
    is_event = 'EVENT' in room_type
    has_fungi_beast = any(getattr(m, 'name', '') == 'Fungi Beast' for m in monsters)

    should_use = (
        (is_boss and turn == 1) or
        (is_elite and (hp_pct <= 30 or (hp_pct <= 50 and turn == 1))) or
        (is_event and not has_fungi_beast and (hp_pct <= 30 or (hp_pct <= 50 and turn == 1)))
    )
    if not should_use:
        return None

    for idx, pot in enumerate(getattr(game, 'potions', [])):
        if not getattr(pot, 'can_use', False):
            continue
        if getattr(pot, 'potion_id', '').lower() in _DONT_USE_POTIONS:
            continue
        if getattr(pot, 'requires_target', False):
            alive = [m for m in monsters
                     if not getattr(m, 'is_gone', False) and getattr(m, 'current_hp', 0) > 0]
            if not alive:
                continue
            target = alive[-1]
            for m in monsters:
                if getattr(m, 'name', '') == 'Reptomancer' and not getattr(m, 'is_gone', False):
                    target = m
                    break
            return PotionAction(use=True, potion_index=idx, target_monster=target)
        return PotionAction(use=True, potion_index=idx)

    return None


def _lagavulin_worth_delaying(game) -> bool:
    try:
        from spirecomm.spire.card import CardType as SCT
        deck = getattr(game, 'deck', [])
        for card in deck:
            if getattr(card, 'type', None) == SCT.POWER:
                return True
        worth_names = {'terror', 'terror+', 'talk to the hand', 'talk to the hand+'}
        for card in deck:
            if getattr(card, 'name', '').lower() in worth_names:
                return True
        relic_ids = {getattr(r, 'relic_id', '').lower() for r in getattr(game, 'relics', [])}
        return 'warped tongs' in relic_ids or 'ice cream' in relic_ids
    except Exception:
        return True  # safe default: still wait


def _select_comparator(game):
    monsters = [m for m in (game.monsters or []) if m is not None]
    alive = [m for m in monsters
             if not getattr(m, 'is_gone', False) and getattr(m, 'current_hp', 0) > 0]
    names = {getattr(m, 'name', '') for m in monsters}
    floor = getattr(game, 'floor', 0)
    turn = getattr(game, 'turn', 1) or 1
    room_type = str(getattr(game, 'room_type', '')).upper()

    if floor in (33, 50):
        return BigFightComparator()
    if 'Gremlin Nob' in names:
        return GremlinNobComparator()
    if 'Sentry' in names and len(alive) == 3:
        return ThreeSentriesTurn1Comparator() if turn == 1 else ThreeSentriesComparator()
    if 'Lagavulin' in names and turn <= 2 and 'EVENT' not in room_type and _lagavulin_worth_delaying(game):
        return WaitingLagavulinComparator()
    if 'Transient' in names and len(alive) == 1:
        return TransientComparator()
    return CommonGeneralComparator()


def _get_paths(state: BattleState, max_count: int):
    return get_paths_bfs(state, max_count)


def _pick_best(paths: dict, original: BattleState, comparator):
    best = None
    for path in paths.values():
        path.state.end_turn()
        if best is None:
            best = path
        elif comparator.does_challenger_defeat_the_best(best.state, path.state, original):
            best = path
    return best


def _play_to_action(play, game):
    card_idx, target_idx = play
    if target_idx >= 0:
        monsters = [m for m in (game.monsters or []) if m is not None]
        if target_idx < len(monsters):
            return PlayCardAction(
                card_index=card_idx,
                target_index=monsters[target_idx].monster_index,
            )
    return PlayCardAction(card_index=card_idx)


def _get_cards_to_select(plays, play_type, game):
    hand = getattr(game, 'hand', [])
    cards = []
    for card_idx, pt in plays:
        if pt != play_type:
            break
        if card_idx < len(hand):
            cards.append(hand[card_idx])
    return cards


class GraphBattleAgent:
    handles_forced_discard = True  # used by main.py to route HAND_SELECT/GRID during combat here
    def __init__(self, max_path_count: int = MAX_PATH_COUNT):
        self._max = max_path_count
        self._memory = TheBotsMemoryBook.new_default()
        self._last_floor = -1
        self._last_turn = 0

    def reset_run(self):
        self._memory = TheBotsMemoryBook.new_default()
        self._last_floor = -1
        self._last_turn = 0
        log.info("GraphBattleAgent: run reset")

    def reset_battle(self):
        self._memory.set_new_battle_state()
        self._last_turn = 0
        log.info("GraphBattleAgent: battle reset (floor %s)", self._last_floor)

    def act(self, game):
        floor = getattr(game, 'floor', 0)
        turn = getattr(game, 'turn', 1) or 1

        if floor != self._last_floor:
            self._last_floor = floor
            self.reset_battle()

        if turn != self._last_turn:
            self._memory.set_new_turn_state()
            self._memory.memory_general[MemoryItem.LAST_KNOWN_TURN] = turn
            self._last_turn = turn

        potion_action = _try_use_potion(game, turn)
        if potion_action:
            log.info("floor=%s turn=%s → USE POTION", floor, turn)
            return potion_action

        try:
            state = game_to_battle_state(game, self._memory)
            comparator = _select_comparator(game)
            paths = _get_paths(state, self._max)
            best = _pick_best(paths, state, comparator)

            alive = [m for m in (game.monsters or []) if m and not getattr(m, 'is_gone', False) and m.current_hp > 0]
            monsters_str = " | ".join(f"{getattr(m,'name','?')} {m.current_hp}hp" for m in alive)
            hand_names = [getattr(c, 'name', getattr(c, 'card_id', '?')) for c in getattr(game, 'hand', [])]
            log.info("floor=%s turn=%s  [%s]  hand=%s  paths=%d  comp=%s",
                     floor, turn, monsters_str, hand_names, len(paths), type(comparator).__name__)

            if best and best.plays:
                play = best.plays[0]
                if play[1] in (PLAY_DISCARD, PLAY_EXHAUST):
                    cards = _get_cards_to_select(best.plays, play[1], game)
                    if cards:
                        names_str = ", ".join(getattr(c, 'name', '?') for c in cards)
                        action_str = "discard" if play[1] == PLAY_DISCARD else "exhaust"
                        log.info("  → %s: [%s]  (full path len=%d)", action_str, names_str, len(best.plays))
                        self._update_memory(state, play)
                        return CardSelectAction(cards)
                    log.warning("GraphBattleAgent: PLAY_DISCARD/EXHAUST but no cards resolved, ending turn")
                    return EndTurnAction()
                card = getattr(game, 'hand', [])[play[0]] if play[0] < len(getattr(game, 'hand', [])) else None
                card_name = getattr(card, 'name', getattr(card, 'card_id', '?')) if card else '?'
                target_name = alive[play[1]].name if play[1] >= 0 and play[1] < len(alive) else 'all'
                log.info("  → play: %s → %s  (full path len=%d)", card_name, target_name, len(best.plays))
                self._update_memory(state, play)
                return _play_to_action(play, game)
            else:
                log.info("  → END TURN (no plays)")

        except Exception:
            log.exception("GraphBattleAgent: BFS error")

        return EndTurnAction()

    def _update_memory(self, state: BattleState, play):
        try:
            tmp = battlestate_deepcopy(state)
            tmp.transform_from_play(play, is_first_play=False)
            self._memory.memory_general = tmp.memory_general.copy()
            self._memory.memory_by_card = {
                k: {rk: dict(rv) for rk, rv in v.items()}
                for k, v in tmp.memory_by_card.items()
            }
        except Exception:
            log.exception("GraphBattleAgent: memory update error")
