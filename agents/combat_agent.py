import os
import logging
import numpy as np
from spirecomm.communication.action import PlayCardAction, EndTurnAction

from config import (
    CARD_TO_IDX, INTENT_TO_IDX, OBS_SIZE, ACTION_SIZE,
    MODELS_DIR, CARD_IDX_UNKNOWN, INTENT_MAX_IDX,
)


# ── Вспомогательные функции (используются также в CombatEnv) ──────────

def game_to_obs(game) -> np.ndarray:
    """Преобразует состояние игры в вектор наблюдений (22 float32).

    Структура:
      [0]    player_hp / max_hp
      [1]    energy / 3
      [2]    block / 100
      [3..12] hand[0..4]: (card_id/100, cost/3) × 5
      [13..21] enemies[0..2]: (hp_norm, intent_norm, block_norm) × 3
    """
    obs = np.zeros(OBS_SIZE, dtype=np.float32)

    # Игрок
    obs[0] = game.player.current_hp / max(game.player.max_hp, 1)
    obs[1] = game.player.energy / 3.0
    obs[2] = min(game.player.block / 100.0, 1.0)

    # Карты в руке (до 5)
    for i, card in enumerate(game.hand[:5]):
        card_id = card.card_id if hasattr(card, "card_id") else str(card)
        cost = card.cost if (hasattr(card, "cost") and card.cost >= 0) else 0
        obs[3 + i * 2] = CARD_TO_IDX.get(card_id, CARD_IDX_UNKNOWN) / 100.0
        obs[4 + i * 2] = min(cost / 3.0, 1.0)

    # Живые враги (до 3)
    live = [m for m in game.monsters if m.current_hp > 0]
    for i, monster in enumerate(live[:3]):
        base = 13 + i * 3
        obs[base]     = monster.current_hp / max(monster.max_hp, 1)
        intent_str = _intent_str(monster.intent)
        obs[base + 1] = INTENT_TO_IDX.get(intent_str, INTENT_MAX_IDX) / INTENT_MAX_IDX
        obs[base + 2] = min(monster.block / 100.0, 1.0)

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
            # Нужна цель — разрешаем только действия с врагом
            for enemy_group in range(1, min(len(live) + 1, 3)):
                mask[card_idx + enemy_group * 5] = True
        else:
            # Цель не нужна
            mask[card_idx] = True

    mask[15] = True  # Завершить ход — всегда можно
    return mask


def action_to_spirecomm(action: int, game):
    """Конвертирует целое действие в spirecomm Action.

    Пространство действий:
      0-4:   карта 0-4 без цели
      5-9:   карта 0-4 на врага 0
      10-14: карта 0-4 на врага 1
      15:    завершить ход
    """
    if action == 15:
        return EndTurnAction()

    card_idx     = action % 5
    target_group = action // 5  # 0=без цели, 1=враг 0, 2=враг 1

    if card_idx >= len(game.hand):
        return EndTurnAction()

    card = game.hand[card_idx]
    live = [m for m in game.monsters if m.current_hp > 0]

    _alog = logging.getLogger("ActionDebug")
    card_id = getattr(card, "card_id", str(card))

    if target_group == 0:
        return PlayCardAction(card)

    enemy_idx = target_group - 1
    if enemy_idx < len(live):
        return PlayCardAction(card, target_index=live[enemy_idx].monster_index)
    if live:
        return PlayCardAction(card, target_index=live[0].monster_index)
    return PlayCardAction(card)


# ── Агент ─────────────────────────────────────────────────────────────

class CombatAgent:
    """Низкоуровневый боевой агент на основе PPO (Stable-Baselines3)."""

    def __init__(self):
        self.model = None
        self._try_load_model()

    def _try_load_model(self):
        import logging
        _log = logging.getLogger("CombatAgent")
        try:
            from stable_baselines3 import PPO
            model_path = os.path.join(MODELS_DIR, "combat_ppo.zip")
            if os.path.exists(model_path):
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

def _intent_str(intent) -> str:
    """Нормализует intent в строку вида 'ATTACK', 'BUFF' и т.д."""
    s = str(intent).upper()
    # Обрабатывает 'MonsterIntentType.ATTACK' → 'ATTACK'
    return s.split(".")[-1] if "." in s else s
