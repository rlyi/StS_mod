import os
import logging
import numpy as np
from spirecomm.communication.action import PlayCardAction, EndTurnAction, PotionAction

from config import (
    INTENT_TO_IDX, OBS_SIZE, ACTION_SIZE, POTION_SLOTS,
    MODELS_DIR, INTENT_MAX_IDX, CARD_PROPERTIES,
)


# ── Вспомогательные функции (используются также в CombatEnv) ──────────

def game_to_obs(game) -> np.ndarray:
    """Преобразует состояние игры в вектор наблюдений (32 float32).

    Структура:
      [0]     player_hp / max_hp
      [1]     energy / 3
      [2]     block / 100
      [3]     strength / 10  (знаковый, clamp [-1, 1])
      [4]     dexterity / 10 (знаковый, clamp [-1, 1])
      [5]     vulnerable / 5
      [6]     weak / 5
      [7]     poison / 20
      [8]     deck_size / 30
      [9]     discard_size / 30
      [10..19] hand[0..4]: (card_id/100, cost/3) × 5
      [20..31] enemies[0..2]: (hp_norm, intent_norm, block_norm, damage_norm) × 3
    """
    obs = np.zeros(OBS_SIZE, dtype=np.float32)

    if game is None or game.player is None:
        return obs

    player = game.player

    # Игрок — базовые
    obs[0] = player.current_hp / max(player.max_hp, 1)
    obs[1] = player.energy / 3.0
    obs[2] = min(player.block / 100.0, 1.0)

    # Игрок — эффекты
    obs[3] = np.clip(_get_power(player, "Strength")  / 10.0, -1.0, 1.0)
    obs[4] = np.clip(_get_power(player, "Dexterity") / 10.0, -1.0, 1.0)
    obs[5] = min(_get_power(player, "Vulnerable") / 5.0, 1.0)
    obs[6] = min(_get_power(player, "Weakened")   / 5.0, 1.0)
    obs[7] = min(_get_power(player, "Poison")     / 20.0, 1.0)

    # Игрок — колода
    obs[8] = min(len(getattr(game, "deck",         [])) / 30.0, 1.0)
    obs[9] = min(len(getattr(game, "discard_pile", [])) / 30.0, 1.0)

    # Карты в руке (до 5) — 7 значений на карту
    for i, card in enumerate(game.hand[:5]):
        card_id  = card.card_id if hasattr(card, "card_id") else str(card)
        cost     = card.cost if (hasattr(card, "cost") and card.cost >= 0) else 0
        dmg, blk = CARD_PROPERTIES.get(card_id, (0, 0))
        type_str = _card_type_str(card)
        base     = 10 + i * 7
        obs[base]     = 1.0 if type_str == "ATTACK" else 0.0
        obs[base + 1] = 1.0 if type_str == "SKILL"  else 0.0
        obs[base + 2] = 1.0 if type_str == "POWER"  else 0.0
        obs[base + 3] = 1.0 if type_str in ("CURSE", "STATUS") else 0.0
        obs[base + 4] = min(dmg  / 20.0, 1.0)
        obs[base + 5] = min(blk  / 20.0, 1.0)
        obs[base + 6] = min(cost / 3.0,  1.0)

    # Живые враги (до 4)
    live = [m for m in game.monsters if m.current_hp > 0]
    for i, monster in enumerate(live[:4]):
        base = 45 + i * 4
        obs[base]     = monster.current_hp / max(monster.max_hp, 1)
        intent_str    = _intent_str(monster.intent)
        obs[base + 1] = INTENT_TO_IDX.get(intent_str, INTENT_MAX_IDX) / INTENT_MAX_IDX
        obs[base + 2] = min(monster.block / 100.0, 1.0)
        obs[base + 3] = min(_monster_damage(monster) / 30.0, 1.0)

    # Зелья (2 слота): [present, type_norm] × 2
    potions = getattr(game, "potions", [])
    for i in range(POTION_SLOTS):
        base = 61 + i * 2
        if i < len(potions) and potions[i].potion_id != "Potion Slot":
            obs[base]     = 1.0
            obs[base + 1] = _potion_type_norm(potions[i].potion_id)

    return obs


def get_valid_actions(game) -> list:
    """Булева маска допустимых действий (список из ACTION_SIZE bool)."""
    mask = [False] * ACTION_SIZE
    live = [m for m in game.monsters if m.current_hp > 0]

    for card_idx, card in enumerate(game.hand[:5]):
        if not getattr(card, "is_playable", True):
            continue
        has_target = getattr(card, "has_target", False)

        # Доп. проверка: ATTACK-карты почти всегда требуют цель
        if not has_target:
            card_type = str(getattr(card, "type", "")).upper()
            ct = card_type.split(".")[-1] if "." in card_type else card_type
            if ct == "ATTACK" and live:
                has_target = True

        if has_target:
            for enemy_group in range(1, min(len(live) + 1, 5)):
                mask[card_idx + enemy_group * 5] = True
        else:
            mask[card_idx] = True

    mask[25] = True  # Завершить ход — всегда можно

    # Зелья: действия 26-35 (слот 0: 26-30, слот 1: 31-35)
    potions = getattr(game, "potions", [])
    for slot in range(POTION_SLOTS):
        if slot >= len(potions):
            break
        p = potions[slot]
        if p.potion_id == "Potion Slot" or not getattr(p, "can_use", False):
            continue
        base = 26 + slot * 5
        if getattr(p, "requires_target", False):
            for ei in range(min(len(live), 4)):
                mask[base + 1 + ei] = True
        else:
            mask[base] = True

    return mask


def action_to_spirecomm(action: int, game):
    """Конвертирует целое действие в spirecomm Action.

    Пространство действий:
      0-4:   карта 0-4 без цели
      5-9:   карта 0-4 на врага 0
      10-14: карта 0-4 на врага 1
      15-19: карта 0-4 на врага 2
      20-24: карта 0-4 на врага 3
      25:    завершить ход
      26-30: зелье слот 0 (26=без цели, 27-30=враг 0-3)
      31-35: зелье слот 1 (31=без цели, 32-35=враг 0-3)
    """
    if action == 25:
        return EndTurnAction()

    live = [m for m in game.monsters if m.current_hp > 0]

    # Зелья
    if 26 <= action <= 35:
        slot          = (action - 26) // 5
        target_offset = (action - 26) % 5   # 0=без цели, 1-4=враг 0-3
        if target_offset == 0:
            return PotionAction(use=True, potion_index=slot)
        enemy_idx = target_offset - 1
        t_idx = live[enemy_idx].monster_index if enemy_idx < len(live) else live[0].monster_index if live else None
        return PotionAction(use=True, potion_index=slot, target_index=t_idx)

    # Карты
    card_idx     = action % 5
    target_group = action // 5

    if card_idx >= len(game.hand):
        return EndTurnAction()

    if target_group == 0:
        return PlayCardAction(card_index=card_idx)

    enemy_idx = target_group - 1
    if enemy_idx < len(live):
        return PlayCardAction(card_index=card_idx, target_index=live[enemy_idx].monster_index)
    if live:
        return PlayCardAction(card_index=card_idx, target_index=live[0].monster_index)
    return PlayCardAction(card_index=card_idx)


# ── Агент ─────────────────────────────────────────────────────────────

class CombatAgent:
    """Низкоуровневый боевой агент на основе PPO (Stable-Baselines3)."""

    def __init__(self):
        self.model = None
        self._try_load_model()

    def _try_load_model(self):
        import glob
        import logging
        _log = logging.getLogger("CombatAgent")
        try:
            from stable_baselines3 import PPO
            # Сначала ищем фиксированное имя, потом последний чекпоинт по шагам
            model_path = os.path.join(MODELS_DIR, "combat_ppo.zip")
            if not os.path.exists(model_path):
                checkpoints = sorted(
                    glob.glob(os.path.join(MODELS_DIR, "combat_ppo_*_steps.zip")),
                    key=lambda p: int(p.split("_")[-2]),
                )
                model_path = checkpoints[-1] if checkpoints else None
            if model_path and os.path.exists(model_path):
                self.model = PPO.load(model_path)
                _log.info("Модель загружена из %s", model_path)
            else:
                _log.info("Модель не найдена — используется случайная политика")
        except ImportError:
            _log.warning("stable-baselines3 не установлен — случайная политика")

    def predict(self, obs: np.ndarray, valid_actions: list | None = None) -> int:
        if self.model is not None:
            action, _ = self.model.predict(obs, deterministic=True)
            action = int(action)
        else:
            # Случайное допустимое действие
            if valid_actions:
                valid = [i for i, v in enumerate(valid_actions) if v]
                return int(np.random.choice(valid)) if valid else 15
            return int(np.random.randint(0, ACTION_SIZE))

        # Применяем маску
        if valid_actions and not valid_actions[action]:
            valid = [i for i, v in enumerate(valid_actions) if v]
            action = valid[0] if valid else 15

        return action

    def act(self, game):
        """Вернуть spirecomm Action для текущего состояния боя."""
        obs    = game_to_obs(game)
        valid  = get_valid_actions(game)
        action = self.predict(obs, valid)
        return action_to_spirecomm(action, game)


# ── Утилита ───────────────────────────────────────────────────────────

def _card_type_str(card) -> str:
    """Возвращает тип карты: ATTACK / SKILL / POWER / CURSE / STATUS."""
    s = str(getattr(card, "type", "")).upper()
    return s.split(".")[-1] if "." in s else s


def _intent_str(intent) -> str:
    """Нормализует intent в строку вида 'ATTACK', 'BUFF' и т.д."""
    s = str(intent).upper()
    return s.split(".")[-1] if "." in s else s


def _get_power(entity, power_id: str) -> float:
    """Возвращает количество стаков power_id у entity (0 если нет)."""
    for power in getattr(entity, "powers", []):
        if getattr(power, "power_id", "") == power_id:
            return float(getattr(power, "amount", 0))
    return 0.0


_HEALING_POTIONS  = {"HealthPotion", "FruitJuice", "BloodPotion", "LiquidMemories"}
_OFFENSIVE_POTIONS = {"FirePotion", "ExplosivePotion", "PoisonPotion",
                      "ThrowingKnivesPotion", "FearPotion", "WeakPotion", "SmokeBomb"}

def _potion_type_norm(potion_id: str) -> float:
    """Кодирует тип зелья в [0,1]: 0.33=лечение, 0.67=атака, 1.0=баф/другое."""
    if potion_id in _HEALING_POTIONS:   return 0.33
    if potion_id in _OFFENSIVE_POTIONS: return 0.67
    return 1.0


def _monster_damage(monster) -> float:
    """Суммарный урон атаки монстра в этот ход (0 если не атакует)."""
    dmg  = getattr(monster, "move_adjusted_damage", -1)
    hits = getattr(monster, "move_hits", 0)
    if dmg is None or dmg < 0 or not hits:
        return 0.0
    return float(dmg * hits)
