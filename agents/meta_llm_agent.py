import json
import logging
import random
import re
import requests

from agents.base_agent import BaseMetaAgent

LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"
LM_MODEL      = "qwen/qwen3-8b"

_MAX_TOKENS = 128

_SYSTEM = (
    "You are an expert Slay the Spire player controlling an Ironclad, Act 1, Ascension 0. "
    "You know card synergies, relic effects, and optimal routing. "
    "Respond ONLY with a valid JSON object, no extra text."
)

# /no_think в конце user-сообщения отключает reasoning у Qwen3
_NO_THINK = " /no_think"

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
        opts  = "\n".join(f"  {i}: {n}" for i, n in enumerate(names))
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
        opts  = "\n".join(f"  {i}: {n}" for i, n in enumerate(names))
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
        opts  = "\n".join(f"  {i}: {n}" for i, n in enumerate(names))
        prompt = (
            f"{_ctx(game)}\n\n"
            f"HAND SELECT — choose the best card from your hand:\n{opts}\n\n"
            f'Return JSON: {{"choice": <0-{len(cards)-1}>}}'
        )
        idx = self._ask_index(prompt, len(cards), fallback=0)
        self.log.info("LLM choose_hand: %d (%s)", idx, names[idx])
        return idx
