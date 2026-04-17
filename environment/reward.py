from config import (
    REWARD_WIN, REWARD_KILL_ENEMY, REWARD_DAMAGE_MULT,
    REWARD_LOSE, REWARD_DAMAGE_TAKEN_MULT, REWARD_TURN_PENALTY,
    REWARD_ACT1_BOSS, ACT1_BOSSES,
)


def calculate_reward(prev_game, curr_game) -> float:
    """Вычислить награду между двумя последовательными состояниями боя.

    Схема наград:
      +2.0   победа в бою (все враги убиты)
      +0.1   убийство врага
      +0.005 за единицу нанесённого урона (= 0.05 за 10 урона)
      -2.0   смерть игрока
      -0.01  за единицу полученного урона
      -0.01  штраф за каждый ход
    """
    reward = REWARD_TURN_PENALTY

    if prev_game is None or curr_game.player is None or prev_game.player is None:
        return reward

    # ── Урон по игроку ────────────────────────────────────────────────
    hp_delta = curr_game.player.current_hp - prev_game.player.current_hp
    if hp_delta < 0:
        reward += (-hp_delta) * REWARD_DAMAGE_TAKEN_MULT  # MULT отрицательный

    # ── Смерть игрока ─────────────────────────────────────────────────
    if curr_game.player.current_hp <= 0:
        reward += REWARD_LOSE
        return reward

    # ── Урон по врагам и убийства ─────────────────────────────────────
    prev_live = [m for m in prev_game.monsters if m.current_hp > 0]
    curr_live_names = {m.name for m in curr_game.monsters if m.current_hp > 0}

    for prev_m in prev_live:
        curr_m = _find_monster(curr_game.monsters, prev_m.name)
        if curr_m is None or curr_m.current_hp <= 0:
            # Враг умер
            reward += REWARD_KILL_ENEMY
        else:
            dmg = prev_m.current_hp - curr_m.current_hp
            if dmg > 0:
                reward += dmg * REWARD_DAMAGE_MULT

    # ── Победа (все враги убиты) ──────────────────────────────────────
    curr_all_dead = all(m.current_hp <= 0 for m in curr_game.monsters)
    if prev_live and curr_all_dead:
        reward += REWARD_WIN
        if any(m.name in ACT1_BOSSES for m in prev_game.monsters):
            reward += REWARD_ACT1_BOSS

    return reward


def _find_monster(monsters, name: str):
    """Найти монстра по имени (первого с таким именем)."""
    for m in monsters:
        if m.name == name:
            return m
    return None
