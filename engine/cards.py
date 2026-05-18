from engine.enums import CardId, CardType, Cost


class Card:
    def __init__(self, card_id: CardId, upgrade: int, cost: int, needs_target: bool,
                 card_type: CardType, ethereal: bool = False, exhausts: bool = False,
                 uuid: str = "default"):
        self.id: CardId = card_id
        self.upgrade: int = upgrade
        self.cost: int = cost
        self.needs_target: bool = needs_target
        self.ethereal: bool = ethereal
        self.exhausts: bool = exhausts
        self.type: CardType = card_type
        self.uuid: str = uuid

    def get_state_string(self) -> str:
        return f"{self.id.value}{self.upgrade}{self.cost},"


def get_card(card_id: CardId, cost: int = None, upgrade: int = 0) -> Card:
    # ── Ironclad ──────────────────────────────────────────────────────────
    if card_id == CardId.STRIKE_R:
        return Card(card_id, upgrade, 1 if cost is None else cost, True, CardType.ATTACK)
    if card_id == CardId.DEFEND_R:
        return Card(card_id, upgrade, 1 if cost is None else cost, False, CardType.SKILL)
    if card_id == CardId.BASH:
        return Card(card_id, upgrade, 2 if cost is None else cost, True, CardType.ATTACK)
    if card_id == CardId.ANGER:
        return Card(card_id, upgrade, 0 if cost is None else cost, True, CardType.ATTACK)
    if card_id == CardId.CLEAVE:
        return Card(card_id, upgrade, 1 if cost is None else cost, True, CardType.ATTACK)
    if card_id == CardId.CLOTHESLINE:
        return Card(card_id, upgrade, 2 if cost is None else cost, True, CardType.ATTACK)
    if card_id == CardId.HEAVY_BLADE:
        return Card(card_id, upgrade, 2 if cost is None else cost, True, CardType.ATTACK)
    if card_id == CardId.IRON_WAVE:
        return Card(card_id, upgrade, 1 if cost is None else cost, True, CardType.ATTACK)
    if card_id == CardId.PERFECTED_STRIKE:
        return Card(card_id, upgrade, 2 if cost is None else cost, True, CardType.ATTACK)
    if card_id == CardId.POMMEL_STRIKE:
        return Card(card_id, upgrade, 1 if cost is None else cost, True, CardType.ATTACK)
    if card_id == CardId.SHRUG_IT_OFF:
        return Card(card_id, upgrade, 1 if cost is None else cost, False, CardType.SKILL)
    if card_id == CardId.THUNDERCLAP:
        return Card(card_id, upgrade, 1 if cost is None else cost, True, CardType.ATTACK)
    if card_id == CardId.TWIN_STRIKE:
        return Card(card_id, upgrade, 1 if cost is None else cost, True, CardType.ATTACK)
    if card_id == CardId.BLOODLETTING:
        return Card(card_id, upgrade, 0 if cost is None else cost, False, CardType.SKILL)
    if card_id == CardId.BLOOD_FOR_BLOOD:
        base = 4 if not upgrade else 3
        return Card(card_id, upgrade, base if cost is None else cost, True, CardType.ATTACK)
    if card_id == CardId.CARNAGE:
        return Card(card_id, upgrade, 2 if cost is None else cost, True, CardType.ATTACK, ethereal=True)
    if card_id == CardId.UPPERCUT:
        return Card(card_id, upgrade, 2 if cost is None else cost, True, CardType.ATTACK)
    if card_id == CardId.DISARM:
        return Card(card_id, upgrade, 1 if cost is None else cost, True, CardType.SKILL, exhausts=True)
    if card_id == CardId.DROPKICK:
        return Card(card_id, upgrade, 1 if cost is None else cost, True, CardType.ATTACK)
    if card_id == CardId.ENTRENCH:
        base = 2 if not upgrade else 1
        return Card(card_id, upgrade, base if cost is None else cost, False, CardType.SKILL)
    if card_id == CardId.FLAME_BARRIER:
        return Card(card_id, upgrade, 2 if cost is None else cost, False, CardType.SKILL)
    if card_id == CardId.GHOSTLY_ARMOR:
        return Card(card_id, upgrade, 1 if cost is None else cost, False, CardType.SKILL, ethereal=True)
    if card_id == CardId.HEMOKINESIS:
        return Card(card_id, upgrade, 1 if cost is None else cost, True, CardType.ATTACK)
    if card_id == CardId.INFLAME:
        return Card(card_id, upgrade, 1 if cost is None else cost, False, CardType.POWER)
    if card_id == CardId.FIRE_BREATHING:
        return Card(card_id, upgrade, 1 if cost is None else cost, False, CardType.POWER)
    if card_id == CardId.EVOLVE:
        return Card(card_id, upgrade, 1 if cost is None else cost, False, CardType.POWER)
    if card_id == CardId.BERSERK:
        return Card(card_id, upgrade, 0 if cost is None else cost, False, CardType.POWER)
    if card_id == CardId.DEMON_FORM:
        return Card(card_id, upgrade, 3 if cost is None else cost, False, CardType.POWER)
    if card_id == CardId.BURNING_PACT:
        return Card(card_id, upgrade, 1 if cost is None else cost, False, CardType.SKILL)
    if card_id == CardId.INTIMIDATE:
        return Card(card_id, upgrade, 0 if cost is None else cost, False, CardType.SKILL, exhausts=True)
    if card_id == CardId.PUMMEL:
        return Card(card_id, upgrade, 1 if cost is None else cost, True, CardType.ATTACK, exhausts=True)
    if card_id == CardId.SEEING_RED:
        base = 1 if not upgrade else 0
        return Card(card_id, upgrade, base if cost is None else cost, False, CardType.SKILL, exhausts=True)
    if card_id == CardId.SHOCKWAVE:
        return Card(card_id, upgrade, 2 if cost is None else cost, False, CardType.SKILL, exhausts=True)
    if card_id == CardId.WHIRLWIND:
        return Card(card_id, upgrade, Cost.x_cost, False, CardType.ATTACK)
    if card_id == CardId.BLUDGEON:
        return Card(card_id, upgrade, 3 if cost is None else cost, True, CardType.ATTACK)
    if card_id == CardId.FEED:
        return Card(card_id, upgrade, 1 if cost is None else cost, True, CardType.ATTACK, exhausts=True)
    if card_id == CardId.FIEND_FIRE:
        return Card(card_id, upgrade, 2 if cost is None else cost, True, CardType.ATTACK, exhausts=True)
    if card_id == CardId.IMMOLATE:
        return Card(card_id, upgrade, 2 if cost is None else cost, False, CardType.ATTACK)
    if card_id == CardId.IMPERVIOUS:
        return Card(card_id, upgrade, 2 if cost is None else cost, False, CardType.SKILL, exhausts=True)
    if card_id == CardId.LIMIT_BREAK:
        return Card(card_id, upgrade, 1 if cost is None else cost, False, CardType.SKILL, exhausts=not upgrade)
    if card_id == CardId.OFFERING:
        return Card(card_id, upgrade, 0 if cost is None else cost, False, CardType.SKILL, exhausts=True)
    if card_id == CardId.JAX:
        return Card(card_id, upgrade, 0 if cost is None else cost, False, CardType.SKILL)
    if card_id == CardId.BODY_SLAM:
        base = 1 if not upgrade else 0
        return Card(card_id, upgrade, base if cost is None else cost, True, CardType.ATTACK)
    if card_id == CardId.CLASH:
        return Card(card_id, upgrade, 0 if cost is None else cost, True, CardType.ATTACK)
    if card_id == CardId.FLEX:
        return Card(card_id, upgrade, 0 if cost is None else cost, False, CardType.SKILL)
    if card_id == CardId.WILD_STRIKE:
        return Card(card_id, upgrade, 1 if cost is None else cost, True, CardType.ATTACK)
    if card_id == CardId.DOUBLE_TAP:
        return Card(card_id, upgrade, 1 if cost is None else cost, False, CardType.SKILL)
    if card_id == CardId.BATTLE_TRANCE:
        return Card(card_id, upgrade, 0 if cost is None else cost, False, CardType.SKILL)
    if card_id == CardId.RAGE:
        return Card(card_id, upgrade, 0 if cost is None else cost, False, CardType.SKILL)
    if card_id == CardId.RAMPAGE:
        return Card(card_id, upgrade, 1 if cost is None else cost, True, CardType.ATTACK)
    if card_id == CardId.SWORD_BOOMERANG:
        return Card(card_id, upgrade, 1 if cost is None else cost, False, CardType.ATTACK)
    if card_id == CardId.JUGGERNAUT:
        return Card(card_id, upgrade, 2 if cost is None else cost, False, CardType.POWER)
    if card_id == CardId.METALLICIZE:
        return Card(card_id, upgrade, 1 if cost is None else cost, False, CardType.POWER)
    if card_id == CardId.RECKLESS_CHARGE:
        return Card(card_id, upgrade, 0 if cost is None else cost, True, CardType.ATTACK)
    if card_id == CardId.POWER_THROUGH:
        return Card(card_id, upgrade, 1 if cost is None else cost, False, CardType.SKILL)
    if card_id == CardId.SPOT_WEAKNESS:
        return Card(card_id, upgrade, 1 if cost is None else cost, True, CardType.SKILL)
    if card_id == CardId.REAPER:
        return Card(card_id, upgrade, 2 if cost is None else cost, False, CardType.ATTACK)
    if card_id == CardId.SEVER_SOUL:
        return Card(card_id, upgrade, 2 if cost is None else cost, True, CardType.ATTACK)
    if card_id == CardId.SECOND_WIND:
        return Card(card_id, upgrade, 1 if cost is None else cost, False, CardType.SKILL)
    if card_id == CardId.BARRICADE:
        base = 3 if not upgrade else 2
        return Card(card_id, upgrade, base if cost is None else cost, False, CardType.POWER)
    # ── Colorless ─────────────────────────────────────────────────────────
    if card_id == CardId.BANDAGE_UP:
        return Card(card_id, upgrade, 0 if cost is None else cost, False, CardType.SKILL)
    if card_id == CardId.BITE:
        return Card(card_id, upgrade, 1 if cost is None else cost, True, CardType.ATTACK)
    if card_id == CardId.BLIND:
        return Card(card_id, upgrade, 0 if cost is None else cost,
                    True if not upgrade else False, CardType.SKILL)
    if card_id == CardId.TRIP:
        return Card(card_id, upgrade, 0 if cost is None else cost,
                    True if not upgrade else False, CardType.SKILL)
    if card_id == CardId.DARK_SHACKLES:
        return Card(card_id, upgrade, 0 if cost is None else cost, False, CardType.SKILL, exhausts=True)
    if card_id == CardId.FLASH_OF_STEEL:
        return Card(card_id, upgrade, 0 if cost is None else cost, True, CardType.ATTACK)
    if card_id == CardId.SWIFT_STRIKE:
        return Card(card_id, upgrade, 0 if cost is None else cost, True, CardType.ATTACK)
    if card_id == CardId.APOTHEOSIS:
        base = 2 if not upgrade else 1
        return Card(card_id, upgrade, base if cost is None else cost, False, CardType.SKILL, exhausts=True)
    if card_id == CardId.HAND_OF_GREED:
        return Card(card_id, upgrade, 2 if cost is None else cost, True, CardType.ATTACK)
    if card_id == CardId.MASTER_OF_STRATEGY:
        return Card(card_id, upgrade, 0 if cost is None else cost, False, CardType.SKILL, exhausts=True)
    if card_id == CardId.APPARITION:
        return Card(card_id, upgrade, 1 if cost is None else cost, False, CardType.SKILL,
                    ethereal=True if not upgrade else False, exhausts=True)
    if card_id == CardId.DEEP_BREATH:
        return Card(card_id, upgrade, 0 if cost is None else cost, False, CardType.SKILL)
    if card_id == CardId.RITUAL_DAGGER:
        return Card(card_id, upgrade, 1 if cost is None else cost, True, CardType.ATTACK, exhausts=True)
    if card_id == CardId.ENLIGHTENMENT:
        return Card(card_id, upgrade, 0 if cost is None else cost, False, CardType.SKILL)
    if card_id == CardId.IMPATIENCE:
        return Card(card_id, upgrade, 0 if cost is None else cost, False, CardType.SKILL)
    if card_id == CardId.MAYHEM:
        base = 2 if not upgrade else 1
        return Card(card_id, upgrade, base if cost is None else cost, False, CardType.POWER)
    if card_id == CardId.PANACHE:
        return Card(card_id, upgrade, 0 if cost is None else cost, False, CardType.POWER)
    if card_id == CardId.SADISTIC_NATURE:
        return Card(card_id, upgrade, 0 if cost is None else cost, False, CardType.POWER)

    # ── Statuses ──────────────────────────────────────────────────────────
    if card_id == CardId.WOUND:
        return Card(card_id, 0, Cost.unplayable, False, CardType.STATUS)
    if card_id == CardId.DAZED:
        return Card(card_id, 0, Cost.unplayable, False, CardType.STATUS, ethereal=True)
    if card_id == CardId.VOID:
        return Card(card_id, 0, Cost.unplayable, False, CardType.STATUS, ethereal=True)
    if card_id == CardId.SLIMED:
        return Card(card_id, 0, 1 if cost is None else cost, False, CardType.STATUS, exhausts=True)
    if card_id == CardId.BURN:
        return Card(card_id, upgrade, Cost.unplayable, False, CardType.STATUS)

    # ── Curses ────────────────────────────────────────────────────────────
    if card_id == CardId.PAIN:
        return Card(card_id, 0, Cost.unplayable, False, CardType.CURSE)
    if card_id == CardId.REGRET:
        return Card(card_id, 0, Cost.unplayable, False, CardType.CURSE)
    if card_id == CardId.DECAY:
        return Card(card_id, 0, Cost.unplayable, False, CardType.CURSE)
    if card_id == CardId.DOUBT:
        return Card(card_id, 0, Cost.unplayable, False, CardType.CURSE)
    if card_id == CardId.SHAME:
        return Card(card_id, 0, Cost.unplayable, False, CardType.CURSE)
    if card_id == CardId.CURSE_OF_THE_BELL:
        return Card(card_id, 0, Cost.unplayable, False, CardType.CURSE)
    if card_id == CardId.PARASITE:
        return Card(card_id, 0, Cost.unplayable, False, CardType.CURSE)
    if card_id == CardId.INJURY:
        return Card(card_id, 0, Cost.unplayable, False, CardType.CURSE)
    if card_id == CardId.NORMALITY:
        return Card(card_id, 0, Cost.unplayable, False, CardType.CURSE)
    if card_id == CardId.NECRONOMICURSE:
        return Card(card_id, 0, Cost.unplayable, False, CardType.CURSE)
    if card_id == CardId.ASCENDERS_BANE:
        return Card(card_id, 0, Cost.unplayable, False, CardType.CURSE, ethereal=True)
    if card_id == CardId.CLUMSY:
        return Card(card_id, 0, Cost.unplayable, False, CardType.CURSE, ethereal=True)

    # ── Other classes (needed for cross-class spawns/effects) ─────────────
    if card_id == CardId.STRIKE_G:
        return Card(card_id, upgrade, 1 if cost is None else cost, True, CardType.ATTACK)
    if card_id == CardId.DEFEND_G:
        return Card(card_id, upgrade, 1 if cost is None else cost, False, CardType.SKILL)
    if card_id == CardId.STRIKE_P:
        return Card(card_id, upgrade, 1 if cost is None else cost, True, CardType.ATTACK)
    if card_id == CardId.DEFEND_P:
        return Card(card_id, upgrade, 1 if cost is None else cost, False, CardType.SKILL)
    if card_id == CardId.STRIKE_B:
        return Card(card_id, upgrade, 1 if cost is None else cost, True, CardType.ATTACK)
    if card_id == CardId.DEFEND_B:
        return Card(card_id, upgrade, 1 if cost is None else cost, False, CardType.SKILL)
    if card_id == CardId.SHIV:
        return Card(card_id, upgrade, 0 if cost is None else cost, True, CardType.ATTACK, exhausts=True)
    if card_id == CardId.INSIGHT:
        return Card(card_id, upgrade, 0 if cost is None else cost, False, CardType.SKILL, exhausts=True)
    if card_id == CardId.SMITE:
        return Card(card_id, upgrade, 1 if cost is None else cost, True, CardType.ATTACK, exhausts=True)
    if card_id == CardId.SAFETY:
        return Card(card_id, upgrade, 1 if cost is None else cost, False, CardType.SKILL, exhausts=True)
    if card_id == CardId.THROUGH_VIOLENCE:
        return Card(card_id, upgrade, 0 if cost is None else cost, True, CardType.ATTACK, exhausts=True)
    if card_id == CardId.BETA:
        base = 2 if not upgrade else 1
        return Card(card_id, upgrade, base if cost is None else cost, False, CardType.SKILL, exhausts=True)
    if card_id == CardId.OMEGA:
        return Card(card_id, upgrade, 3 if cost is None else cost, False, CardType.POWER)
    if card_id == CardId.TANTRUM:
        return Card(card_id, upgrade, 1 if cost is None else cost, True, CardType.ATTACK)
    if card_id == CardId.FLURRY_OF_BLOWS:
        return Card(card_id, upgrade, 0 if cost is None else cost, True, CardType.ATTACK)
    if card_id == CardId.WEAVE:
        return Card(card_id, upgrade, 0 if cost is None else cost, True, CardType.ATTACK)
    if card_id == CardId.SENTINEL:
        return Card(card_id, upgrade, 1 if cost is None else cost, False, CardType.SKILL)
    if card_id == CardId.FEEL_NO_PAIN:
        return Card(card_id, upgrade, 1 if cost is None else cost, False, CardType.POWER)
    if card_id == CardId.DARK_EMBRACE:
        base = 2 if not upgrade else 1
        return Card(card_id, upgrade, base if cost is None else cost, False, CardType.POWER)
    if card_id == CardId.CORRUPTION:
        base = 3 if not upgrade else 2
        return Card(card_id, upgrade, base if cost is None else cost, False, CardType.POWER)

    # ── Internal placeholder ──────────────────────────────────────────────
    if card_id == CardId.CARD_FROM_DRAW:
        return Card(card_id, 0, Cost.unplayable, False, CardType.FAKE)

    # Unknown → treat as unplayable fake
    return Card(CardId.FAKE, 0, Cost.unplayable, False, CardType.FAKE)
