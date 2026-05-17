import math
from typing import List, Optional

from engine.enums import CardId, PowerId, RelicId, PotionId


DEBUFFS: List[PowerId] = [
    PowerId.BIAS,
    PowerId.BLOCK_RETURN,
    PowerId.CHOKED,
    PowerId.CONFUSED,
    PowerId.CONSTRICTED,
    PowerId.DRAW_REDUCTION,
    PowerId.ENTANGLED,
    PowerId.FASTING,
    PowerId.FRAIL,
    PowerId.LOCK_ON,
    PowerId.MARK,
    PowerId.NO_DRAW,
    PowerId.POISON,
    PowerId.VULNERABLE,
    PowerId.WEAKENED,
    PowerId.WRAITH_FORM_POWER,
]

DEBUFFS_WHEN_NEGATIVE: List[PowerId] = [
    PowerId.STRENGTH,
    PowerId.DEXTERITY,
    PowerId.FOCUS,
]


def get_power_count(powers: dict, desired_powers: List[PowerId]) -> int:
    return sum(powers[p] for p in powers.keys() if p in desired_powers)


class Target:
    def __init__(self, is_player: bool, current_hp: int, max_hp: int, block: int,
                 powers: dict, relics: dict = None, potions: list = None):
        self.is_player: bool = is_player
        self.current_hp: int = current_hp
        self.max_hp: int = max_hp
        self.block: int = block
        self.powers: dict = powers
        self.relics: dict = {} if relics is None else relics
        self.potions: list = [] if potions is None else potions

    def inflict_damage(self, source, base_damage: int, hits: int, blockable: bool = True,
                       vulnerable_modifier: float = 1.5, is_attack: bool = True,
                       min_hp_damage: int = 1, is_orbs: bool = False,
                       card_id: CardId = None) -> int:
        damage = base_damage
        if self.powers.get(PowerId.VULNERABLE):
            damage = math.floor(damage * vulnerable_modifier)
        if is_orbs and self.powers.get(PowerId.LOCK_ON):
            damage = math.floor(damage * 1.5)

        health_damage_dealt = 0
        times_block_triggered = 0
        sharp_hide_done = False
        trigger_malleable_block = 0

        for _ in range(hits):
            hit_damage = damage

            if is_attack and self.powers.get(PowerId.BLOCK_RETURN):
                source.block += self.powers.get(PowerId.BLOCK_RETURN)
                times_block_triggered += 1

            if self.powers.get(PowerId.INTANGIBLE_PLAYER):
                hit_damage = 1
            if self.powers.get(PowerId.INTANGIBLE_ENEMY):
                hit_damage = 1

            if self.powers.get(PowerId.FLIGHT):
                hit_damage = math.floor(hit_damage * 0.5)

            if blockable and self.block:
                if self.block > hit_damage:
                    self.block -= hit_damage
                    hit_damage = 0
                else:
                    hit_damage -= self.block
                    self.block = 0
                    if source.relics.get(RelicId.HAND_DRILL):
                        self.add_powers({PowerId.VULNERABLE: 2}, source.relics, source.powers)

            if hit_damage > 0:
                if self.relics.get(RelicId.TORII) and hit_damage < 6:
                    hit_damage = 1
                if self.relics.get(RelicId.TUNGSTEN_ROD):
                    hit_damage -= 1

                if hit_damage > 0:
                    hit_damage = max(hit_damage, min_hp_damage)
                    if self.powers.get(PowerId.BUFFER):
                        self.powers[PowerId.BUFFER] -= 1
                        if not self.powers[PowerId.BUFFER]:
                            del self.powers[PowerId.BUFFER]
                        continue
                    self.current_hp -= hit_damage
                    health_damage_dealt += hit_damage
                    if is_attack and self.powers.get(PowerId.PLATED_ARMOR):
                        self.powers[PowerId.PLATED_ARMOR] -= 1
                    if is_attack and self.powers.get(PowerId.FLIGHT):
                        self.powers[PowerId.FLIGHT] -= 1
                    if is_attack and self.powers.get(PowerId.ANGRY):
                        if not self.powers.get(PowerId.STRENGTH):
                            self.powers[PowerId.STRENGTH] = 0
                        self.powers[PowerId.STRENGTH] += self.powers.get(PowerId.ANGRY)
                    if is_attack and self.powers.get(PowerId.MODE_SHIFT):
                        self.powers[PowerId.MODE_SHIFT] -= hit_damage
                    if is_attack and self.powers.get(PowerId.CURL_UP):
                        self.block = self.powers.get(PowerId.CURL_UP)
                        del self.powers[PowerId.CURL_UP]
                    if is_attack and self.powers.get(PowerId.MALLEABLE):
                        self.powers[PowerId.MALLEABLE] += 1
                        trigger_malleable_block += 1
                    if is_attack and source.powers.get(PowerId.ENVENOM):
                        self.add_powers({PowerId.POISON: 1}, source.relics, source.powers)

            plated_armor = self.powers.get(PowerId.PLATED_ARMOR, None)
            if plated_armor is not None and plated_armor < 1:
                del self.powers[PowerId.PLATED_ARMOR]

            flight = self.powers.get(PowerId.FLIGHT, None)
            if flight is not None and flight < 1:
                self.damage = 0
                self.hits = 0
                del self.powers[PowerId.FLIGHT]
            ms = self.powers.get(PowerId.MODE_SHIFT)
            if ms is not None and ms < 1:
                self.damage = 0
                self.hits = 0
                self.block = 20
                del self.powers[PowerId.MODE_SHIFT]

            if card_id:
                if card_id == CardId.WALLOP:
                    source.block += health_damage_dealt
                if card_id == CardId.REAPER:
                    source.heal(health_damage_dealt, True, self.relics)

            if is_attack and (self.powers.get(PowerId.FLAME_BARRIER) or self.powers.get(PowerId.THORNS)):
                source.inflict_damage(
                    source=self,
                    base_damage=self.powers.get(PowerId.FLAME_BARRIER, 0) + self.powers.get(PowerId.THORNS, 0),
                    hits=1,
                    vulnerable_modifier=1,
                    is_attack=False,
                )

            if is_attack and self.powers.get(PowerId.SHARP_HIDE) and not sharp_hide_done:
                source.inflict_damage(
                    source=self,
                    base_damage=self.powers.get(PowerId.SHARP_HIDE),
                    hits=1,
                    vulnerable_modifier=1,
                    is_attack=False,
                )
                sharp_hide_done = True

            if self.current_hp < 0:
                health_damage_dealt += self.current_hp
                self.current_hp = 0
                if self.relics.get(RelicId.LIZARD_TAIL):
                    if self.relics[RelicId.LIZARD_TAIL] != -2:
                        self.heal(math.floor(self.max_hp * 0.5), True, self.relics, is_revive=True)
                        self.relics[RelicId.LIZARD_TAIL] = -2
                    continue
                if PotionId.FAIRY_IN_A_BOTTLE in self.potions:
                    self.heal(math.floor(self.max_hp * 0.3), True, self.relics, is_revive=True)
                    self.potions.remove(PotionId.FAIRY_IN_A_BOTTLE)
                    continue
                break

            if source.current_hp <= 0:
                source.current_hp = 0
                break

        if trigger_malleable_block:
            block_to_add = self.powers.get(PowerId.MALLEABLE)
            for _ in range(trigger_malleable_block):
                self.block += block_to_add
                block_to_add -= 1

        if self.powers.get(PowerId.SHIFTING):
            self.add_powers({PowerId.STRENGTH: -health_damage_dealt}, source.relics, source.powers)

        return times_block_triggered

    def add_powers(self, powers: dict, relics: dict, source_powers: dict) -> list:
        applied_powers = []
        for power in powers:
            if self.powers.get(PowerId.ARTIFACT) and \
                    (power in DEBUFFS or (powers[power] < 0 and power in DEBUFFS_WHEN_NEGATIVE)):
                if self.powers[PowerId.ARTIFACT] == 1:
                    del self.powers[PowerId.ARTIFACT]
                else:
                    self.powers[PowerId.ARTIFACT] -= 1
                continue
            applied_powers.append(power)

            if power in self.powers:
                self.powers[power] += powers[power]
            else:
                self.powers[power] = powers[power]

            if source_powers and not self.is_player:
                if source_powers.get(PowerId.SADISTIC):
                    self.inflict_damage(self, source_powers.get(PowerId.SADISTIC), 1,
                                        vulnerable_modifier=1, is_attack=False)

            if relics:
                if relics.get(RelicId.CHAMPION_BELT) and power == PowerId.VULNERABLE:
                    self.add_powers({PowerId.WEAKENED: 1}, relics, source_powers)
                if relics.get(RelicId.SNECKO_SKULL) and power == PowerId.POISON:
                    self.powers[PowerId.POISON] += 1

        return applied_powers

    def get_state_string(self) -> str:
        state = f"{self.current_hp},{self.max_hp},{self.block}"
        power_keys = sorted([k.value for k in self.powers.keys()])
        for k in power_keys:
            state += k + str(self.powers[PowerId(k)]) + ","
        return state

    def heal(self, amount: int, is_player: bool, relics: dict, is_revive: bool = False):
        if is_player and relics.get(RelicId.MARK_OF_THE_BLOOM):
            return
        if self.current_hp <= 0 and not is_revive:
            return
        if is_player and relics.get(RelicId.MAGIC_FLOWER):
            self.current_hp += round(amount * 1.5)
        else:
            self.current_hp += amount
        if self.current_hp > self.max_hp:
            self.current_hp = self.max_hp


class Player(Target):
    def __init__(self, is_player: bool, current_hp: int, max_hp: int, block: int,
                 powers: dict, energy: int, relics: dict, potions: list):
        super().__init__(is_player, current_hp, max_hp, block, powers, relics, potions)
        self.energy: int = energy

    def get_state_string(self) -> str:
        return super().get_state_string() + str(self.energy) + ","


class Monster(Target):
    def __init__(self, is_player: bool, current_hp: int, max_hp: int, block: int,
                 powers: dict, damage: int = 0, hits: int = 0,
                 is_gone: bool = False, name: str = None):
        super().__init__(is_player, current_hp, max_hp, block, powers)
        self.damage: int = damage
        self.hits: int = hits
        self.is_gone: bool = is_gone
        self.name: str = name

    def get_state_string(self) -> str:
        return super().get_state_string() + f"{self.damage},{self.hits},"


def find_lowest_hp_monster(monsters: list) -> Optional[Monster]:
    lowest = None
    for monster in monsters:
        if monster.is_gone or monster.current_hp <= 0:
            continue
        if lowest is None or monster.current_hp < lowest.current_hp:
            lowest = monster
    return lowest
