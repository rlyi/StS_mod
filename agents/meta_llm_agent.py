import json
import logging
import random
import re
import requests

from agents.base_agent import BaseMetaAgent

LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"
LM_MODEL      = "qwen/qwen3-8b"

_MAX_TOKENS = 128

_SYSTEM = """You are an expert Slay the Spire player controlling an Ironclad, Act 1 (floors 1-17), Ascension 0.
Respond ONLY with a valid JSON object, no extra text.

GAME MECHANICS:
- Each combat you draw cards, spend Energy (3/turn) to play them, then End Turn.
- Block resets each turn. Strength adds to all attack damage. Weak/Vulnerable last 3 turns.
- Deck size matters: smaller decks cycle faster and are more consistent.

MAP NODES (choose your path wisely):
- M = Monster: standard fight, low risk, gold+card reward
- E = Elite: harder fight, drops a Relic (powerful passive item) — worth it if HP > 60%
- R = Rest Site: REST heals 30% max HP, SMITH upgrades a card permanently
- ? = Unknown: usually an Event (often free rewards), sometimes Treasure
- $ = Shop: buy cards/relics/potions, can remove a card from deck (very valuable)
- T = Treasure: free chest, no fight

IRONCLAD STRATEGY:
- Early (floors 1-6): prioritize damage and deck consistency, grab Block if taking lots of damage
- Mid (floors 7-12): look for synergies (Strength scaling, Block generation, draw)
- Pre-boss (floors 13-17): save HP for boss, prefer REST over SMITH if below 50% HP
- Best cards to upgrade: Bash, Shrug It Off, Sentinel, Dropkick, Whirlwind
- Cards to avoid adding: Clumsy, Doubt, Pain, Shame (curses), Normality, Parasite
- Skip card rewards if your deck is already good — more cards = slower cycling
- Removing Strike/Defend via Shop purge is extremely valuable for consistency"""

# /no_think в конце user-сообщения отключает reasoning у Qwen3
_NO_THINK = " /no_think"

# ── Описания карт Ironclad ────────────────────────────────────────────
_CARD_DESC = {
    # Стартовые
    "Strike_R":         "Deal 6 damage.",
    "Defend_R":         "Gain 5 Block.",
    "Bash":             "Deal 8 damage. Apply 2 Vulnerable.",
    # Обычные атаки
    "Anger":            "Deal 6 damage. Add a copy to your discard pile.",
    "Body Slam":        "Deal damage equal to your current Block.",
    "Clash":            "Can only be played if every card in hand is an Attack. Deal 14 damage.",
    "Cleave":           "Deal 8 damage to ALL enemies.",
    "Clothesline":      "Deal 12 damage. Apply 2 Weak.",
    "Headbutt":         "Deal 9 damage. Put the top of your discard pile onto your draw pile.",
    "Heavy Blade":      "Deal 14 damage. Strength multiplies this card's damage by 3x.",
    "Iron Wave":        "Gain 5 Block. Deal 5 damage.",
    "Perfected Strike": "Deal 6 damage +2 per Strike in your deck.",
    "Pommel Strike":    "Deal 9 damage. Draw 1 card.",
    "Sword Boomerang":  "Deal 3 damage 3 times to a random enemy.",
    "Thunderclap":      "Deal 4 damage to ALL enemies. Apply 1 Vulnerable to ALL.",
    "Twin Strike":      "Deal 5 damage twice.",
    "Wild Strike":      "Deal 12 damage. Shuffle a Wound into your draw pile.",
    # Обычные скиллы
    "Armaments":        "Gain 5 Block. Upgrade a card in your hand for the rest of combat.",
    "Flex":             "Gain 2 Strength. At end of turn lose 2 Strength.",
    "Havoc":            "Play the top card of your draw pile and Exhaust it.",
    "Shrug It Off":     "Gain 8 Block. Draw 1 card.",
    "True Grit":        "Gain 7 Block. Exhaust a random card in your hand.",
    "Warcry":           "Draw 2 cards. Put 1 card from your hand on top of your draw pile. Exhaust.",
    # Необычные атаки
    "Blood for Blood":  "Costs 1 less Energy for each time you lost HP this combat. Deal 18 damage.",
    "Carnage":          "Ethereal. Deal 20 damage.",
    "Dropkick":         "Deal 5 damage. If enemy is Vulnerable: gain 1 Energy and draw 1 card.",
    "Hemokinesis":      "Lose 2 HP. Deal 15 damage.",
    "Pummel":           "Deal 2 damage 4 times. Exhaust.",
    "Rampage":          "Deal 8 damage. Permanently increase this card's damage by 5 each time played.",
    "Reckless Charge":  "Deal 7 damage. Shuffle a Dazed into your draw pile.",
    "Searing Blow":     "Deal 12 damage. Can be upgraded any number of times (+4 damage each).",
    "Sever Soul":       "Exhaust all non-Attack cards in hand. Deal 16 damage.",
    "Uppercut":         "Deal 13 damage. Apply 1 Weak and 1 Vulnerable.",
    "Whirlwind":        "Deal 5 damage to ALL enemies X times (X = Energy spent).",
    # Необычные скиллы/пауэры
    "Battle Trance":    "Draw 3 cards. You cannot draw additional cards this turn. Exhaust.",
    "Bloodletting":     "Lose 3 HP. Gain 2 Energy.",
    "Burning Pact":     "Exhaust 1 card. Draw 2 cards.",
    "Combust":          "Power. At end of each turn lose 1 HP and deal 5 damage to ALL enemies.",
    "Dark Embrace":     "Power. Whenever you Exhaust a card draw 1 card.",
    "Disarm":           "Enemy loses 2 Strength. Exhaust.",
    "Dual Wield":       "Create 1 copy of an Attack or Power card in hand.",
    "Entrench":         "Double your current Block.",
    "Evolve":           "Power. Whenever you draw a Status card draw 1 card.",
    "Fire Breathing":   "Power. Whenever you draw a Status or Curse deal 6 damage to ALL enemies.",
    "Flame Barrier":    "Gain 12 Block. Whenever you are attacked this turn deal 4 damage back.",
    "Ghostly Armor":    "Ethereal. Gain 10 Block.",
    "Infernal Blade":   "Add a random Attack to your hand (costs 0). Exhaust.",
    "Inflame":          "Power. Gain 2 Strength.",
    "Intimidate":       "Apply 1 Weak to ALL enemies. Exhaust.",
    "Metallicize":      "Power. At end of each turn gain 3 Block.",
    "Power Through":    "Add 2 Wounds to your hand. Gain 15 Block.",
    "Rage":             "Power. Whenever you play an Attack gain 3 Block.",
    "Rupture":          "Power. Whenever you lose HP from a card gain 1 Strength.",
    "Second Wind":      "Exhaust all non-Attack cards in hand. Gain 5 Block per card exhausted.",
    "Seeing Red":       "Gain 2 Energy. Exhaust.",
    "Sentinel":         "Gain 5 Block. If Exhausted gain 2 Energy.",
    "Shockwave":        "Apply 3 Weak and 3 Vulnerable to ALL enemies. Exhaust.",
    "Spot Weakness":    "If enemy intends to attack gain 3 Strength.",
    # Редкие атаки
    "Bludgeon":         "Deal 32 damage.",
    "Choke":            "Deal 12 damage. Whenever you play a card this turn enemy loses 3 Strength.",
    "Feed":             "Deal 10 damage. If fatal permanently gain 3 max HP. Exhaust.",
    "Fiend Fire":       "Exhaust your hand. Deal 7 damage for each card exhausted. Exhaust.",
    "Immolate":         "Deal 21 damage to ALL enemies. Add a Burn to your discard pile.",
    "Reaper":           "Deal 4 damage to ALL enemies. Heal HP equal to unblocked damage dealt.",
    # Редкие пауэры/скиллы
    "Barricade":        "Power. Block is no longer removed at start of your turn.",
    "Berserk":          "Power. Gain 1 Vulnerability. At start of each turn gain 1 Energy.",
    "Brutality":        "Power. At start of each turn lose 1 HP and draw 1 card.",
    "Corruption":       "Power. Skills cost 0. Whenever you play a Skill it is Exhausted.",
    "Dark Shackles":    "Enemy loses 9 Strength for the rest of this turn. Exhaust.",
    "Double Tap":       "This turn your next 1 Attack is played twice.",
    "Exhume":           "Put any Exhausted card into your hand. Exhaust.",
    "Impervious":       "Gain 30 Block. Exhaust.",
    "Juggernaut":       "Power. Whenever you gain Block deal 5 damage to a random enemy.",
    "Limit Break":      "Double your Strength. Exhaust.",
    "Offering":         "Lose 6 HP. Gain 2 Energy. Draw 3 cards. Exhaust.",
    "Champion":         "Power. Apply 1 Weak and Vulnerable to ALL enemies at start of each turn.",
    "Berserk":          "Power. Lose 1 HP at start of turn, gain 1 Energy.",
    # Статусы / проклятия
    "Wound":            "STATUS. Unplayable.",
    "Burn":             "STATUS. Unplayable. At end of your turn take 2 damage.",
    "Dazed":            "STATUS. Unplayable. Ethereal.",
    "Slimed":           "STATUS. Costs 0. Exhaust.",
    "AscendersBane":    "CURSE. Unplayable. Ethereal.",
    "Clumsy":           "CURSE. Unplayable. Ethereal.",
    "Decay":            "CURSE. Unplayable. At end of turn take 2 damage.",
    "Doubt":            "CURSE. Unplayable. At end of turn gain 1 Weak.",
    "Injury":           "CURSE. Unplayable.",
    "Normality":        "CURSE. Unplayable. You cannot play more than 3 cards this turn.",
    "Pain":             "CURSE. Costs 0. Whenever you play a card lose 1 HP.",
    "Parasite":         "CURSE. Unplayable. If transformed or removed lose 3 max HP.",
    "Pride":            "CURSE. Ethereal. At end of turn shuffle a copy into your draw pile.",
    "Regret":           "CURSE. Unplayable. At end of turn lose HP equal to cards in hand.",
    "Shame":            "CURSE. Unplayable. At start of turn gain 1 Frail.",
    "Writhe":           "CURSE. Unplayable. Always in your starting hand.",
}


def _card_info(name: str) -> str:
    """Возвращает описание карты или пустую строку если нет."""
    base = name.split("+")[0].strip()
    upgraded = "+" in name
    desc = _CARD_DESC.get(base, "")
    suffix = " (upgraded)" if upgraded and desc else ""
    return f"{desc}{suffix}" if desc else ""

_JSON_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)

# JSON schemas для structured output
_SCHEMA_INT = {
    "type": "json_schema",
    "json_schema": {
        "name": "choice_int",
        "schema": {
            "type": "object",
            "properties": {"choice": {"type": "integer"}},
            "required": ["choice"],
            "additionalProperties": False,
        },
    },
}
_SCHEMA_STR = {
    "type": "json_schema",
    "json_schema": {
        "name": "choice_str",
        "schema": {
            "type": "object",
            "properties": {"choice": {"type": "string"}},
            "required": ["choice"],
            "additionalProperties": False,
        },
    },
}

_NODE_DESC = {
    "M": "Monster fight",
    "?": "Unknown room (event or treasure)",
    "E": "Elite fight (harder enemy, better loot)",
    "R": "Rest Site (heal or upgrade a card)",
    "$": "Shop (buy cards, relics, potions)",
    "T": "Treasure chest",
}


def _ctx(game) -> str:
    hp      = getattr(game, "current_hp", 0) or 0
    max_hp  = getattr(game, "max_hp", 1) or 1
    gold    = getattr(game, "gold", 0)
    floor   = getattr(game, "floor", 1)
    deck    = getattr(game, "deck", []) or []
    relics  = getattr(game, "relics", []) or []
    potions = getattr(game, "potions", []) or []

    deck_names   = [getattr(c, "name", str(c)) for c in deck if c is not None]
    relic_names  = [getattr(r, "name", str(r)) for r in relics if r is not None]
    potion_names = [getattr(p, "name", str(p)) for p in potions
                    if p is not None and getattr(p, "can_use", False)]
    deck_preview = ", ".join(deck_names[:20]) + ("..." if len(deck_names) > 20 else "")

    return (
        f"Floor {floor}/17 | HP {hp}/{max_hp} ({hp/max_hp:.0%}) | Gold {gold}\n"
        f"Deck ({len(deck_names)}): {deck_preview}\n"
        f"Relics: {', '.join(relic_names) or 'none'}\n"
        f"Potions: {', '.join(potion_names) or 'none'}"
    )


class LLMMetaAgent(BaseMetaAgent):
    """Мета-агент: все решения принимает Qwen3 через LM Studio (OpenAI-compatible API)."""

    def __init__(self, url: str = LM_STUDIO_URL, model: str = LM_MODEL, timeout: int = 60):
        self.url     = url
        self.model   = model
        self.timeout = timeout
        self.log     = logging.getLogger("LLMMetaAgent")

    # ── LLM call ─────────────────────────────────────────────────────

    def _call(self, user_msg: str, schema=_SCHEMA_INT) -> dict:
        payload = {
            "model":           self.model,
            "messages":        [
                {"role": "system", "content": _SYSTEM},
                {"role": "user",   "content": user_msg},
            ],
            "temperature":     0.1,
            "max_tokens":      _MAX_TOKENS,
            "response_format": schema,
        }
        payload["messages"][-1]["content"] += _NO_THINK
        resp = requests.post(self.url, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"].strip()
        # Fallback: если structured output не сработал — ищем JSON регуляркой
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            m = _JSON_RE.search(text)
            if not m:
                raise ValueError(f"No JSON in LLM response: {text!r}")
            return json.loads(m.group())

    def _ask_index(self, prompt: str, n: int, fallback: int = 0) -> int:
        try:
            data = self._call(prompt)
            idx  = int(data.get("choice", fallback))
            return max(0, min(idx, n - 1))
        except Exception as e:
            self.log.warning("LLM index error: %s", e)
            return fallback

    # ── Decisions ────────────────────────────────────────────────────

    def choose_card(self, game) -> int:
        cards = getattr(getattr(game, "screen", None), "cards", [])
        if not cards:
            return -1

        names = [getattr(c, "name", str(c)) for c in cards]
        opts  = "\n".join(
            f"  {i}: {n}" + (f" — {_card_info(n)}" if _card_info(n) else "")
            for i, n in enumerate(names)
        )
        prompt = (
            f"{_ctx(game)}\n\n"
            f"CARD REWARD — pick the best card for your deck, or skip:\n{opts}\n\n"
            f'Return JSON: {{"choice": <0-{len(cards)-1} or -1 to skip>}}'
        )
        try:
            data   = self._call(prompt)
            choice = int(data.get("choice", 0))
            if choice < 0:
                self.log.info("LLM choose_card: SKIP")
                return -1
            choice = max(0, min(choice, len(cards) - 1))
            self.log.info("LLM choose_card: %d (%s)", choice, names[choice])
            return choice
        except Exception as e:
            self.log.warning("LLM card error: %s", e)
            return random.randrange(len(cards))

    def choose_path(self, game) -> int:
        nodes = getattr(getattr(game, "screen", None), "next_nodes", [])
        if not nodes:
            return 0

        symbols = [str(getattr(n, "symbol", "?")).upper() for n in nodes]
        opts    = "\n".join(
            f"  {i}: {_NODE_DESC.get(s, s)} [{s}]" for i, s in enumerate(symbols)
        )
        prompt = (
            f"{_ctx(game)}\n\n"
            f"MAP — choose your next node:\n{opts}\n\n"
            f'Return JSON: {{"choice": <0-{len(nodes)-1}>}}'
        )
        idx = self._ask_index(prompt, len(nodes), fallback=random.randrange(len(nodes)))
        self.log.info("LLM choose_path: %d (%s)", idx, symbols[idx] if idx < len(symbols) else "?")
        return idx

    def choose_rest(self, game) -> str:
        hp_pct   = game.current_hp / max(game.max_hp or 1, 1)
        heal_amt = int((game.max_hp or 0) * 0.3)
        prompt = (
            f"{_ctx(game)}\n\n"
            f"REST SITE:\n"
            f"  REST: heal {heal_amt} HP\n"
            f"  SMITH: permanently upgrade one card in your deck\n\n"
            f'Return JSON: {{"choice": "REST" or "SMITH"}}'
        )
        fallback = "REST" if hp_pct < 0.6 else "SMITH"
        try:
            data   = self._call(prompt, schema=_SCHEMA_STR)
            choice = str(data.get("choice", fallback)).upper().strip()
            result = choice if choice in ("REST", "SMITH") else fallback
            self.log.info("LLM choose_rest: %s", result)
            return result
        except Exception as e:
            self.log.warning("LLM rest error: %s", e)
            return fallback

    def choose_event(self, game) -> int:
        s       = getattr(game, "screen", None)
        options = getattr(s, "options", [])
        if not options:
            return 0

        event_name = getattr(s, "event_name", "Unknown Event")
        opt_texts  = [getattr(o, "text", str(o)) for o in options]
        opts       = "\n".join(f"  {i}: {t}" for i, t in enumerate(opt_texts))
        prompt = (
            f"{_ctx(game)}\n\n"
            f"EVENT: {event_name}\nOptions:\n{opts}\n\n"
            f'Return JSON: {{"choice": <0-{len(options)-1}>}}'
        )
        idx = self._ask_index(prompt, len(options), fallback=len(options) - 1)
        self.log.info("LLM choose_event '%s': %d", event_name, idx)
        return idx

    def choose_boss_relic(self, game) -> int:
        relics = getattr(getattr(game, "screen", None), "relics", [])
        if not relics:
            return 0

        names = [getattr(r, "name", str(r)) for r in relics]
        opts  = "\n".join(f"  {i}: {n}" for i, n in enumerate(names))
        prompt = (
            f"{_ctx(game)}\n\n"
            f"BOSS RELIC — choose one (these are powerful relics that shape your entire strategy):\n{opts}\n\n"
            f'Return JSON: {{"choice": <0-{len(relics)-1}>}}'
        )
        idx = self._ask_index(prompt, len(relics), fallback=0)
        self.log.info("LLM choose_boss_relic: %d (%s)", idx, names[idx])
        return idx

    def choose_shop(self, game) -> int:
        """Возвращает индекс в объединённом списке [cards | relics | potions] или -1 = выйти."""
        s       = getattr(game, "screen", None)
        gold    = getattr(game, "gold", 0)
        cards   = getattr(s, "cards",   [])
        relics  = getattr(s, "relics",  [])
        potions = getattr(s, "potions", [])

        all_items = cards + relics + potions
        if not all_items:
            return -1

        def fmt(item, idx):
            name  = getattr(item, "name", str(item))
            price = getattr(item, "price", None)
            if price is None:
                return f"  {idx}: {name} — ? gold"
            tag = "" if int(price) <= gold else " [too expensive]"
            return f"  {idx}: {name} — {price} gold{tag}"

        opts = "\n".join(fmt(item, i) for i, item in enumerate(all_items))
        prompt = (
            f"{_ctx(game)}\n\n"
            f"SHOP — you have {gold} gold. Items:\n{opts}\n\n"
            f'Return JSON: {{"choice": <0-{len(all_items)-1} or -1 to leave shop>}}'
        )
        try:
            data   = self._call(prompt)
            choice = int(data.get("choice", -1))
            if choice < 0:
                self.log.info("LLM choose_shop: leave")
                return -1
            choice = max(0, min(choice, len(all_items) - 1))
            name   = getattr(all_items[choice], "name", str(all_items[choice]))
            price  = getattr(all_items[choice], "price", 0)
            if isinstance(price, int) and price > gold:
                self.log.info("LLM choose_shop: %s too expensive (%d > %d) — leave", name, price, gold)
                return -1
            self.log.info("LLM choose_shop: %d (%s)", choice, name)
            return choice
        except Exception as e:
            self.log.warning("LLM shop error: %s", e)
            return -1

    def choose_grid(self, game, for_upgrade: bool = False) -> int:
        """Выбор карты в GRID (апгрейд, удаление, или дублирование)."""
        s      = getattr(game, "screen", None)
        cards  = getattr(s, "cards", [])
        if not cards:
            return 0

        is_purge   = getattr(s, "for_purge",   False)
        is_upgrade = for_upgrade or getattr(s, "for_upgrade", False)
        action     = "REMOVE (purge) the worst card" if is_purge else \
                     "UPGRADE the best card" if is_upgrade else \
                     "DUPLICATE the best card"

        names = [getattr(c, "name", str(c)) for c in cards]
        opts  = "\n".join(
            f"  {i}: {n}" + (f" — {_card_info(n)}" if _card_info(n) else "")
            for i, n in enumerate(names)
        )
        prompt = (
            f"{_ctx(game)}\n\n"
            f"GRID SELECTION — {action}:\n{opts}\n\n"
            f'Return JSON: {{"choice": <0-{len(cards)-1}>}}'
        )
        idx = self._ask_index(prompt, len(cards), fallback=0)
        self.log.info("LLM choose_grid (%s): %d (%s)", action.split()[0], idx, names[idx])
        return idx

    def choose_hand(self, game) -> int:
        """Выбор карты из руки (HAND_SELECT — для Armaments, Dual Wield и т.п.)."""
        s     = getattr(game, "screen", None)
        cards = getattr(s, "cards", [])
        if not cards:
            return 0

        names = [getattr(c, "name", str(c)) for c in cards]
        opts  = "\n".join(
            f"  {i}: {n}" + (f" — {_card_info(n)}" if _card_info(n) else "")
            for i, n in enumerate(names)
        )
        prompt = (
            f"{_ctx(game)}\n\n"
            f"HAND SELECT — choose the best card from your hand:\n{opts}\n\n"
            f'Return JSON: {{"choice": <0-{len(cards)-1}>}}'
        )
        idx = self._ask_index(prompt, len(cards), fallback=0)
        self.log.info("LLM choose_hand: %d (%s)", idx, names[idx])
        return idx
