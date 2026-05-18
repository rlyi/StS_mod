"""
RuleMetaAgent — rule-based meta-agent ported from bottled_ai's handler logic.

Implements the same strategies as bottled_ai's requested_strike agent
(Ironclad build around Perfected Strike + combat cards).
"""
import logging
from agents.base_agent import BaseMetaAgent, _screen_type
from spirecomm.communication.action import (
    ChooseAction, ProceedAction, BuyPurgeAction, PotionAction,
)

log = logging.getLogger("RuleMeta")

# ---------------------------------------------------------------------------
# Card selection
# ---------------------------------------------------------------------------

# Desired cards with max copies to take, in PRIORITY ORDER (first = most desired).
_DESIRED_CARDS: dict[str, int] = {
    'perfected strike': 5,
    'battle trance': 2,
    'offering': 1,
    'reaper': 2,
    'twin strike': 2,
    'shockwave': 2,
    'thunderclap': 2,
    'dropkick': 2,
    'pommel strike': 2,
    'shrug it off': 2,
    'impervious': 2,
    'ghostly armor': 1,
    'flame barrier': 1,
    'blind': 1,
    'apotheosis': 1,
    'handofgreed': 1,
    'master of strategy': 1,
    'flash of steel': 1,
    'trip': 1,
    'dark shackles': 1,
    'swift strike': 1,
    'dramatic entrance': 1,
    'finesse': 1,
}

# ---------------------------------------------------------------------------
# Map path scoring  (ported from bottled_ai PathHandlerConfig / Path)
# ---------------------------------------------------------------------------

def _get_all_paths(node, game_map, max_y):
    """DFS — all paths from node to last row (or leaf)."""
    # Resolve from map to get populated children
    if game_map is not None:
        map_node = game_map.get_node(node.x, node.y)
        if map_node is not None:
            node = map_node
    if not node.children or node.y >= max_y:
        return [[node]]
    paths = []
    for child in node.children:
        for sub in _get_all_paths(child, game_map, max_y):
            paths.append([node] + sub)
    return paths or [[node]]


def _score_path(rooms, hp: float, max_hp: float, gold: float, act: int, game) -> float:
    """Score a path using bottled_ai's PathHandlerConfig defaults."""
    reward = 0.0
    survivability = 1.0

    def has(relic_name):
        return _has_relic(game, relic_name)

    for node in rooms:
        sym = str(getattr(node, 'symbol', '?')).upper()

        if sym == 'M':
            reward += 1.0
            if has('prayer wheel'):
                reward += 0.3
            if has('question card'):
                reward += 0.15
            gold += 15
            hp -= act * 5
            if has('meat on the bone') and hp / max_hp < 0.5:
                hp += 12
            if has('blood vial'):
                hp += 2
            if has('black blood'):
                hp += 12
            if has('burning blood'):
                hp += 6

        elif sym == 'E':
            reward += 1.0 + 1.5  # base + relic
            if has('question card'):
                reward += 0.15
            if has('black star'):
                reward += 1.5
            gold += 30
            hp -= (act + 1) * 15
            if has('meat on the bone') and hp / max_hp < 0.5:
                hp += 12
            if has('blood vial'):
                hp += 2
            if has('black blood'):
                hp += 12
            if has('burning blood'):
                hp += 6

        elif sym == 'T':
            reward += 1.5
            if has('matryoshka'):
                reward += 1.0
            if has('cursed key'):
                reward -= 1.5

        elif sym == 'R':
            if has('eternal feather'):
                deck_size = len(getattr(game, 'deck', []))
                hp += (deck_size // 5) * 3
            has_fusion   = has('fusion hammer')
            has_coffee   = has('coffee dripper')
            if not has_fusion and (has_coffee or hp / max_hp >= 0.6):
                reward += 1.1
            if not has_coffee and (has_fusion or hp / max_hp < 0.6):
                hp += max_hp * 0.3
                if has('regal pillow'):
                    hp += 15
                if has('dreamcatcher'):
                    reward += 0.5

        elif sym == '?':
            reward += 1.0 if act == 1 else 1.5

        elif sym == '$':
            if has('membership card'):
                gold_to_spend = min(gold, 300)
                gold -= gold_to_spend
            else:
                gold_to_spend = min(gold, 200) * 2
                gold -= gold_to_spend / 2
            reward += gold_to_spend / 100

        survive_barrier = max_hp / 4
        if hp < survive_barrier:
            survivability *= max((hp + survive_barrier * 2) / (survive_barrier * 3), 0)
        hp = min(max(hp, 0), max_hp)

    if act != 3:
        reward += gold / 200  # gold_after_boss_reward

    return reward + (survivability - 1) * 15  # survivability_reward_calculation


# ---------------------------------------------------------------------------
# Rest-site logic
# ---------------------------------------------------------------------------

_HP_HEAL_THRESHOLD    = 0.60
_HP_HEAL_FLOOR49      = 0.85
_CAMPFIRE_PRE_BOSS    = {15, 32, 49}
_HIGH_PRIORITY_UPGRADES = {'apotheosis', 'perfected strike'}
_REMOVAL_PRIORITY = ['defend', 'strike', 'defend+', 'strike+']

# ---------------------------------------------------------------------------
# Boss relic priority
# ---------------------------------------------------------------------------

_BOSS_RELIC_PRIORITY = [
    'sozu', 'runic dome', "philosopher's stone", 'ectoplasm',
    'velvet choker', 'cursed key', 'fusion hammer', 'snecko eye',
    'mark of pain', 'busted crown', 'coffee dripper', "slaver's collar",
    'runic cube', 'runic pyramid', 'black blood', 'calling bell',
    'empty cage', 'black star', 'sacred bark',
]

_ENERGY_RELICS = {
    'sozu', 'runic dome', "philosopher's stone", 'ectoplasm',
    'velvet choker', 'cursed key', 'fusion hammer', 'mark of pain',
    'busted crown', 'coffee dripper', 'nuclear battery',
}

# ---------------------------------------------------------------------------
# Shop purchases
# ---------------------------------------------------------------------------

_SHOP_RELICS = [
    'Bag of Marbles', 'Pen Nib', 'Strike Dummy', 'Paper Phrog',
    'Preserved Insect', 'Red Skull', 'Meat on the Bone', 'Eternal Feather',
    'Regal Pillow', "Lee's Waffle", 'Meal Ticket', 'Strawberry',
    'Toy Ornithopter', 'Pantograph', 'Pear', 'Orichalcum', 'Anchor',
    'Horn Cleat', 'Self-Forming Clay', 'Thread and Needle', 'Lantern',
    'Happy Flower', 'Bag of Preparation', 'Centennial Puzzle',
]

_SHOP_CARDS = ['Offering', 'Shockwave']

# ---------------------------------------------------------------------------
# Upgrade priority (choose_grid for_upgrade=True)
# ---------------------------------------------------------------------------

_UPGRADE_PRIORITY = [
    'apotheosis', 'perfected strike', 'bash', 'shockwave', 'uppercut',
    'battle trance', 'offering', 'blind', 'seeing red', 'dropkick',
    'flame barrier', 'twin strike', 'pommel strike', 'handofgreed',
    'thunderclap', 'shrug it off', 'impervious', 'ghostly armor',
    'master of strategy', 'flash of steel', 'trip', 'dark shackles',
    'swift strike', 'dramatic entrance', 'finesse',
]

# ---------------------------------------------------------------------------
# Combat reward — undesired relics
# ---------------------------------------------------------------------------

_UNDESIRED_RELICS = {'bottled flame', 'cloak clasp', 'dead branch'}

# ---------------------------------------------------------------------------
# Potion juggling — most desired first (same as bottled_ai default_desired_potions)
# ---------------------------------------------------------------------------

_DESIRED_POTIONS = [
    'fruit juice', 'fairy in a bottle', 'cultist potion', 'power potion',
    'potion of capacity', 'heart of iron', 'duplication potion',
    'distilled chaos', 'blessing of the forge', 'attack potion',
    'dexterity potion', 'ambrosia', 'fear potion', 'essence of steel',
    'strength potion', 'regen potion', 'blood potion', 'entropic brew',
    'liquid bronze', 'energy potion', 'skill potion', 'ancient potion',
    'weak potion', "gambler's brew", 'poison potion', 'colorless potion',
    'flex potion', 'swift potion', 'bottled miracle', 'fire potion',
    'explosive potion', 'speed potion', 'block potion', 'stance potion',
    'smoke bomb', 'elixir potion', 'liquid memories', 'snecko oil',
]

# ---------------------------------------------------------------------------
# Neow desired choices (priority order, first = most desired)
# ---------------------------------------------------------------------------

_NEOW_CHOICES = [
    'upgrade a card',
    'obtain a random common relic',
    'obtain 100 gold',
    'choose a card to obtain',
    'obtain 3 random potions',
    'choose a colorless card to obtain',
    'max hp +8',
    'obtain a random rare card',
    'enemies in your next three combats have 1 hp',
    'remove a card from your deck',
    'transform a card',
    'lose your starting relic obtain a random boss relic',
]


# ---------------------------------------------------------------------------
# RuleMetaAgent
# ---------------------------------------------------------------------------

class RuleMetaAgent(BaseMetaAgent):
    """Rule-based meta-agent ported from bottled_ai/requested_strike."""

    # --- choose_card ---------------------------------------------------------

    def choose_card(self, game) -> int:
        cards = getattr(getattr(game, 'screen', None), 'cards', [])
        deck = getattr(game, 'deck', [])
        deck_counts: dict[str, int] = {}
        for c in deck:
            k = getattr(c, 'name', getattr(c, 'card_id', '')).lower()
            deck_counts[k] = deck_counts.get(k, 0) + 1

        # Iterate DESIRED_CARDS in priority order — return FIRST match (like bottled_ai)
        card_names = [getattr(c, 'name', getattr(c, 'card_id', '')).lower() for c in cards]
        for desired, max_copies in _DESIRED_CARDS.items():
            if desired not in card_names:
                continue
            if deck_counts.get(desired, 0) >= max_copies:
                continue
            idx = card_names.index(desired)
            log.debug("choose_card: picking '%s' (idx=%d)", desired, idx)
            return idx

        log.debug("choose_card: skip")
        return -1

    def choose_card_forced(self, game) -> int:
        idx = self.choose_card(game)
        return max(idx, 0)

    # --- choose_path ---------------------------------------------------------

    def choose_path(self, game) -> int:
        s = getattr(game, 'screen', None)
        next_nodes = getattr(s, 'next_nodes', [])
        if not next_nodes:
            return 0
        if len(next_nodes) == 1:
            return 0

        game_map = getattr(game, 'map', None)
        if game_map is None:
            return self._choose_path_simple(game)

        act     = getattr(game, 'act', 1)
        hp      = float(game.current_hp)
        max_hp  = float(max(game.max_hp or 1, 1))
        gold    = float(getattr(game, 'gold', 0))

        # Find the highest y in the map (boss row)
        try:
            all_y = list(game_map.nodes.keys())
            max_y = max(all_y) if all_y else 14
        except Exception:
            return self._choose_path_simple(game)

        best_idx   = 0
        best_score = float('-inf')

        for i, screen_node in enumerate(next_nodes):
            map_node = game_map.get_node(screen_node.x, screen_node.y) or screen_node
            try:
                paths = _get_all_paths(map_node, game_map, max_y)
            except Exception:
                paths = [[map_node]]
            for path in paths:
                score = _score_path(path, hp, max_hp, gold, act, game)
                if score > best_score:
                    best_score = score
                    best_idx   = i

        syms = [str(getattr(n, 'symbol', '?')) for n in next_nodes]
        log.debug("choose_path: symbols=%s best_idx=%d score=%.2f", syms, best_idx, best_score)
        return best_idx

    def _choose_path_simple(self, game) -> int:
        """Fallback: score only the immediate next node by symbol and HP%."""
        nodes = getattr(getattr(game, 'screen', None), 'next_nodes', [])
        hp_pct = game.current_hp / max(game.max_hp or 1, 1)
        if hp_pct >= 0.70:
            scores = {'E': 10, '?': 7, '$': 6, 'M': 4, 'R': 3, 'T': 2, 'B': 1}
        elif hp_pct >= 0.40:
            scores = {'?': 8, '$': 7, 'R': 5, 'M': 4, 'E': 2, 'T': 2, 'B': 1}
        else:
            scores = {'R': 9, '$': 7, '?': 5, 'M': 3, 'T': 2, 'E': 1, 'B': 1}
        best_idx, best_score = 0, -1
        for i, node in enumerate(nodes):
            s = scores.get(str(getattr(node, 'symbol', '?')).upper(), 0)
            if s > best_score:
                best_score, best_idx = s, i
        return best_idx

    # --- choose_rest ---------------------------------------------------------

    def choose_rest(self, game) -> str:
        s      = getattr(game, 'screen', None)
        opts   = [str(o).upper() for o in getattr(s, 'rest_options', [])]
        floor  = getattr(game, 'floor', 0)
        hp_pct = game.current_hp / max(game.max_hp or 1, 1)

        has_pantograph     = _has_relic(game, 'pantograph')
        panto_covers       = has_pantograph and floor in _CAMPFIRE_PRE_BOSS and hp_pct >= 0.40
        panto_floor49      = has_pantograph and floor == 49 and hp_pct >= 0.60
        worth_healing      = hp_pct <= _HP_HEAL_THRESHOLD and not panto_covers
        worth_healing_f49  = floor == 49 and hp_pct <= _HP_HEAL_FLOOR49 and not panto_floor49

        can_rest  = any('REST'  in o for o in opts)
        can_smith = any('SMITH' in o for o in opts)
        can_toke  = any('TOKE'  in o for o in opts)
        can_lift  = any('LIFT'  in o for o in opts)
        can_dig   = any('DIG'   in o for o in opts)

        if can_rest and (worth_healing or worth_healing_f49):
            return 'REST'
        if can_toke and _has_removable_curse(game):
            return 'TOKE'
        if can_smith and _has_high_priority_upgrade(game):
            return 'SMITH'
        if can_lift and _girya_below_max(game):
            return 'LIFT'
        if can_dig:
            return 'DIG'
        if can_smith:
            return 'SMITH'
        if can_toke and _has_removal_priority_card(game):
            return 'TOKE'
        return 'REST'

    # --- choose_event --------------------------------------------------------

    def choose_event(self, game) -> int:
        s          = getattr(game, 'screen', None)
        event_name = getattr(s, 'event_name', '') or ''
        event_id   = getattr(s, 'event_id',   '') or ''
        options    = getattr(s, 'options', [])
        n          = len(options)

        hp_pct   = game.current_hp / max(game.max_hp or 1, 1) * 100
        has_ecto = _has_relic(game, 'ectoplasm')
        omamori  = _relic_counter(game, 'omamori')
        floor    = getattr(game, 'floor', 0)

        choice = _event_choice(event_name, event_id, hp_pct, n, has_ecto, omamori, floor, game)
        log.debug("choose_event: '%s' (id=%s) hp=%.0f%% → %d", event_name, event_id, hp_pct, choice)
        return choice

    # --- choose_boss_relic ---------------------------------------------------

    def choose_boss_relic(self, game) -> int:
        relics = getattr(getattr(game, 'screen', None), 'relics', [])
        act    = getattr(game, 'act', 1)

        prefs = _BOSS_RELIC_PRIORITY.copy()

        has_energy = any(
            getattr(r, 'relic_id', '').lower() in _ENERGY_RELICS
            for r in getattr(game, 'relics', [])
        )
        if act == 1 or has_energy:
            prefs = [p for p in prefs if p not in ('busted crown', 'coffee dripper')]
        if has_energy:
            prefs = [p for p in prefs if p != 'mark of pain']

        own_ids = {getattr(r, 'relic_id', '').lower() for r in getattr(game, 'relics', [])}
        if 'snecko eye'    in own_ids: prefs = [p for p in prefs if p != 'runic pyramid']
        if 'runic pyramid' in own_ids: prefs = [p for p in prefs if p != 'snecko eye']

        relic_ids  = [getattr(r, 'relic_id', '').lower() for r in relics]
        relic_low  = [getattr(r, 'name',     '').lower() for r in relics]
        for pref in prefs:
            for i in range(len(relics)):
                if pref in relic_ids[i] or pref in relic_low[i]:
                    return i
        return 0

    # --- choose_shop ---------------------------------------------------------

    def choose_shop(self, game) -> int:
        s      = getattr(game, 'screen', None)
        cards  = getattr(s, 'cards',   [])
        relics = getattr(s, 'relics',  [])
        gold   = getattr(game, 'gold', 0)

        deck_names = {
            getattr(c, 'name', getattr(c, 'card_id', '')).lower()
            for c in getattr(game, 'deck', [])
        }

        # 1. Perfected Strike
        for i, card in enumerate(cards):
            if getattr(card, 'name', getattr(card, 'card_id', '')).lower() == 'perfected strike':
                if gold >= getattr(card, 'price', 9999):
                    return i

        # 2. Membership Card (before other relics)
        for i, r in enumerate(relics):
            if getattr(r, 'name', '') == 'Membership Card' and gold >= getattr(r, 'price', 9999):
                return len(cards) + i

        # 3. Other relics by priority
        relic_names  = [getattr(r, 'name',  '') for r in relics]
        relic_prices = [getattr(r, 'price', 9999) for r in relics]
        for pref in _SHOP_RELICS:
            for i, rname in enumerate(relic_names):
                if rname == pref and gold >= relic_prices[i]:
                    return len(cards) + i

        # 4. Priority cards (one copy)
        for pref in _SHOP_CARDS:
            for i, card in enumerate(cards):
                name  = getattr(card, 'name', getattr(card, 'card_id', '')).lower()
                price = getattr(card, 'price', 9999)
                if name == pref.lower() and gold >= price and name not in deck_names:
                    return i

        return -1  # leave shop

    # --- choose_grid ---------------------------------------------------------

    def choose_grid(self, game, for_upgrade: bool = False) -> int:
        s     = getattr(game, 'screen', None)
        cards = getattr(s, 'cards', [])
        if not cards:
            return 0
        names = [getattr(c, 'name', getattr(c, 'card_id', '')).lower() for c in cards]

        priorities = _UPGRADE_PRIORITY if for_upgrade else _REMOVAL_PRIORITY
        for pref in priorities:
            for i, name in enumerate(names):
                if pref in name:
                    return i
        return 0

    # --- act() override (shop purge + combat reward) -------------------------

    def act(self, game):
        screen = _screen_type(game)

        if screen == 'SHOP_SCREEN':
            purge = self._try_shop_purge(game)
            if purge is not None:
                return purge

        if screen == 'COMBAT_REWARD':
            return self._handle_combat_reward(game)

        return super().act(game)

    def _try_shop_purge(self, game):
        s          = getattr(game, 'screen', None)
        if not getattr(s, 'purge_available', False):
            return None
        purge_cost = getattr(s, 'purge_cost', 9999)
        if getattr(game, 'gold', 0) < purge_cost:
            return None
        if _has_removable_curse(game) or _has_removal_priority_card(game):
            log.info("[SHOP] buying purge")
            return BuyPurgeAction()
        return None

    def _handle_combat_reward(self, game):
        s       = getattr(game, 'screen', None)
        rewards = getattr(s, 'rewards', [])

        for i, reward in enumerate(rewards):
            rt = str(getattr(reward, 'reward_type', '')).upper().split('.')[-1]

            if rt == 'POTION':
                potions   = getattr(game, 'potions', [])
                has_empty = any(
                    getattr(p, 'potion_id', 'Potion Slot') == 'Potion Slot'
                    for p in potions
                )
                if has_empty:
                    return ChooseAction(i)
                juggle = self._try_juggle_potion(game, reward)
                if juggle:
                    return juggle
                continue

            if rt == 'CARD':
                if getattr(self, '_card_skipped', False):
                    continue
                return ChooseAction(i)

            if rt == 'RELIC':
                relic      = getattr(reward, 'relic', None)
                relic_name = getattr(relic, 'name', '').lower() if relic else ''
                if relic_name in _UNDESIRED_RELICS:
                    log.debug("[COMBAT_REWARD] skipping undesired relic: %s", relic_name)
                    continue
                return ChooseAction(i)

            if rt in ('GOLD', 'RELIC_AND_GOLD', 'EMERALD_KEY', 'SAPPHIRE_KEY', 'STOLEN_GOLD'):
                return ChooseAction(i)

        self._card_skipped = False
        return ProceedAction()

    def _try_juggle_potion(self, game, reward):
        """Discard least desired held potion if reward potion is more desired."""
        reward_potion = getattr(reward, 'potion', None)
        if reward_potion is None:
            return None
        reward_name = getattr(reward_potion, 'potion_id', '').lower()

        held_potions = getattr(game, 'potions', [])
        held_names   = [getattr(p, 'potion_id', '').lower() for p in held_potions]
        all_names    = held_names + [reward_name]

        # Iterate from least desired to most desired
        for least in reversed(_DESIRED_POTIONS):
            if least not in all_names:
                continue
            if least == reward_name:
                return None  # reward is already the least — skip it
            for idx, name in enumerate(held_names):
                if name == least:
                    log.info("[COMBAT_REWARD] juggle: discard '%s' for '%s'", name, reward_name)
                    return PotionAction(use=False, potion_index=idx)
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _has_relic(game, relic_id_lower: str) -> bool:
    return any(
        getattr(r, 'relic_id', '').lower() == relic_id_lower
        for r in getattr(game, 'relics', [])
    )


def _relic_counter(game, relic_id_lower: str) -> int:
    for r in getattr(game, 'relics', []):
        if getattr(r, 'relic_id', '').lower() == relic_id_lower:
            return getattr(r, 'counter', 0)
    return 0


def _has_removable_curse(game) -> bool:
    for card in getattr(game, 'deck', []):
        ct   = str(getattr(card, 'type', '')).lower()
        name = getattr(card, 'name', '').lower()
        if ('curse' in ct or 'status' in ct) \
                and 'necronomicurse' not in name \
                and 'ascenders bane' not in name:
            return True
    return False


def _has_high_priority_upgrade(game) -> bool:
    for card in getattr(game, 'deck', []):
        name     = getattr(card, 'name', '').lower()
        upgraded = getattr(card, 'upgraded', False)
        if not upgraded and name in _HIGH_PRIORITY_UPGRADES:
            return True
    return False


def _has_removal_priority_card(game) -> bool:
    names = {getattr(c, 'name', '').lower() for c in getattr(game, 'deck', [])}
    return any(p in names for p in _REMOVAL_PRIORITY)


def _girya_below_max(game) -> bool:
    return _relic_counter(game, 'girya') < 2  # bottled_ai: < 2


def _has_card_in_deck(game, name_lower: str) -> bool:
    return any(
        getattr(c, 'name', getattr(c, 'card_id', '')).lower() == name_lower
        for c in getattr(game, 'deck', [])
    )


def _count_card_in_deck(game, name_lower: str) -> int:
    return sum(
        1 for c in getattr(game, 'deck', [])
        if getattr(c, 'name', getattr(c, 'card_id', '')).lower() == name_lower
    )


def _event_choice(event_name: str, event_id: str, hp_pct: float, n_options: int,
                  has_ecto: bool, omamori: int, floor: int, game) -> int:

    # --- Neow (floor 0) ------------------------------------------------------
    if event_id == 'Neow Event':
        options = getattr(getattr(game, 'screen', None), 'options', [])
        labels  = [getattr(o, 'label', getattr(o, 'text', '')).lower() for o in options]
        if any('leave' in l for l in labels):
            return 0
        for desired in _NEOW_CHOICES:
            for i, label in enumerate(labels):
                if desired in label:
                    return i
        return 0

    match event_name:

        # ACT 1 ---------------------------------------------------------------

        case "Big Fish":
            if hp_pct <= 30:
                return 0
            if omamori >= 1:
                return 2
            return 1

        case "The Cleric":
            options = getattr(getattr(game, 'screen', None), 'options', [])
            labels  = [getattr(o, 'label', getattr(o, 'text', '')).lower() for o in options]
            if hp_pct <= 65 and any('heal' in l for l in labels):
                return next(i for i, l in enumerate(labels) if 'heal' in l)
            purify = next((i for i, l in enumerate(labels) if 'purify' in l), -1)
            if purify >= 0:
                return purify
            if hp_pct >= 90:
                return next((i for i, l in enumerate(labels) if 'leave' in l), n_options - 1)
            return 0

        case "Dead Adventurer":
            return 1

        case "Golden Idol":
            if has_ecto:
                return n_options - 1
            if n_options == 2:
                return 0
            if omamori >= 1:
                return 0
            if hp_pct >= 50:   # requested_strike threshold
                return 1
            return 2

        case "Hypnotizing Mushrooms" | "Mushrooms":
            return 0 if hp_pct >= 40 else 1

        case "Living Wall":
            return 2

        case "Scrap Ooze":
            return 0

        case "Shining Light":
            return 0 if hp_pct >= 50 else 1   # requested_strike threshold

        case "The Ssssserpent":
            if omamori >= 1 and not has_ecto:
                return 0
            return 1

        case "World of Goop":
            if hp_pct >= 70 and not has_ecto:   # requested_strike threshold
                return 0
            return 1

        case "Wing Statue":
            return 0 if hp_pct >= 60 else 1    # requested_strike threshold

        # ACT 1, 2 ------------------------------------------------------------

        case "Face Trader":
            if hp_pct >= 75 and not has_ecto:
                return 0
            return 2

        # ACT 1, 2, 3 ---------------------------------------------------------

        case "A Note For Yourself":
            return 1

        case "Bonfire Spirits":
            return 0

        case "The Divine Fountain":
            return 0

        case "Duplicator":
            return 1

        case "Golden Shrine":
            return 1 if (omamori >= 1 and not has_ecto) else 0

        case "Lab":
            return 0

        case "Match and Keep!":
            return 0

        case "Ominous Forge":
            if omamori >= 1:
                return 1
            if floor >= 30:
                return 0
            return 1

        case "Purifier":
            return 0

        case "Transmogrifier":
            return 1

        case "Upgrade Shrine":
            return 0

        case "We Meet Again!":
            return 0

        case "The Woman in Blue":
            return 0

        # ACT 2 ---------------------------------------------------------------

        case "Ancient Writing":
            return 1

        case "Augmenter":
            return 2

        case "The Colosseum":
            return 0

        case "Council of Ghosts":
            if _has_relic(game, 'snecko eye') or _has_card_in_deck(game, 'bite'):
                return 1
            return 0

        case "Cursed Tome":
            return 1

        case "Forgotten Altar":
            return 0

        case "The Joust":
            return 0

        case "Knowing Skull":
            return 3

        case "The Library":
            return 1

        case "Masked Bandits":
            return 1 if hp_pct >= 65 else 0

        case "The Mausoleum":
            return 0 if omamori >= 1 else 1

        case "The Nest":
            return 0

        case "N'loth":
            return 2

        case "Old Beggar":
            return 0

        case "Pleading Vagrant":
            options = getattr(getattr(game, 'screen', None), 'options', [])
            labels  = [getattr(o, 'label', getattr(o, 'text', '')).lower() for o in options]
            if omamori >= 1:
                return next((i for i, l in enumerate(labels) if 'rob' in l), 0)
            gold_idx = next((i for i, l in enumerate(labels) if 'gold' in l), -1)
            return gold_idx if gold_idx >= 0 else n_options - 1

        case "Vampires(?)":
            if _has_card_in_deck(game, 'apparition'):
                return 1
            if _has_relic(game, 'strike dummy'):
                return 1
            if _count_card_in_deck(game, 'strike') >= 3:
                return 0
            return 1

        # ACT 2, 3 ------------------------------------------------------------

        case "Designer In-Spire":
            return 0

        # ACT 3 ---------------------------------------------------------------

        case "Falling":
            options    = getattr(getattr(game, 'screen', None), 'options', [])
            card_names = [getattr(o, 'label', getattr(o, 'text', '')).lower() for o in options]
            for worst in _REMOVAL_PRIORITY:
                for i, cn in enumerate(card_names):
                    if worst in cn:
                        return i
            for least in reversed(list(_DESIRED_CARDS.keys())):
                for i, cn in enumerate(card_names):
                    if least in cn:
                        return i
            return 0

        case "Mind Bloom":
            return 0

        case "Mysterious Sphere":
            return 0 if hp_pct >= 70 else 1

        case "Secret Portal":
            return 1

        case "Sensory Stone":
            return 0

        case "Tomb of Lord Red Mask":
            gold = getattr(game, 'gold', 0)
            return 0 if (_has_relic(game, 'red mask') or gold <= 130) else 1

        case "Winding Halls":
            if omamori >= 1 and hp_pct < 75:
                return 1
            if hp_pct <= 10:
                return 1
            return 2

        case _:
            log.debug("choose_event: unknown event '%s', defaulting to 0", event_name)
            return 0
