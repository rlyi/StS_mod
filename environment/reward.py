from config import (
    REWARD_WIN, REWARD_WIN_HP_MULT, REWARD_KILL_ENEMY, REWARD_DAMAGE_MULT,
    REWARD_LOSE, REWARD_DAMAGE_TAKEN_MULT, REWARD_TURN_PENALTY,
    REWARD_ACT1_BOSS, ACT1_BOSSES,
)


def calculate_reward(prev_game, curr_game) -> float:
    """Вычислить награду между двумя последовательными состояниями боя.

    Схема наград:
      +2.0 + 1.0×hp_pct  победа (все враги убиты + бонус за оставшийся HP)
      +0.5               убийство врага
      +0.01              за единицу реального урона (без overkill)
      -2.0               смерть игрока
      -0.03              за единицу полученного урона
      -0.01              штраф за каждый ход
      +10.0              победа над боссом Акта 1
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
    prev_live = [m for m in prev_game.monsters if m is not None
                 and getattr(m, "current_hp", 0) > 0 and not getattr(m, "is_gone", False)]

    for prev_m in prev_live:
        curr_m = _find_monster(curr_game.monsters, prev_m.name)
        if curr_m is None or curr_m.current_hp <= 0:
            reward += REWARD_KILL_ENEMY
        else:
            # min(dmg, prev_hp) исключает overkill
            dmg = prev_m.current_hp - curr_m.current_hp
            if dmg > 0:
                reward += min(dmg, prev_m.current_hp) * REWARD_DAMAGE_MULT

    # ── Победа (все враги убиты) ──────────────────────────────────────
    curr_all_dead = all(getattr(m, "current_hp", 0) <= 0 or getattr(m, "is_gone", False)
                        for m in curr_game.monsters if m is not None)
    if prev_live and curr_all_dead:
        hp_pct = curr_game.player.current_hp / max(curr_game.player.max_hp, 1)
        reward += REWARD_WIN + REWARD_WIN_HP_MULT * hp_pct
        if any(m.name in ACT1_BOSSES for m in prev_game.monsters):
            reward += REWARD_ACT1_BOSS

    return reward


def _find_monster(monsters, name: str):
    for m in monsters:
        if m is not None and m.name == name:
            return m
    return None
