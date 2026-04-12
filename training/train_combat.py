"""train_combat.py — обучение PPO-агента для боёв.

Запуск:
  1. Запустите Slay the Spire с ModTheSpire + CommunicationMod.
  2. В config.properties укажите:
       command=python C:\\StS_mod\\training\\train_combat.py
  3. Начните новую игру (Ironclad, Ascension 0).
  4. Агент начнёт обучение автоматически.

Прогресс:
  tensorboard --logdir C:\\StS_mod\\logs\\combat
"""

import os
import sys

# Добавляем корень проекта в путь, чтобы импорты работали
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback, EvalCallback
)
from stable_baselines3.common.monitor import Monitor

from environment.combat_env import CombatEnv
from config import MODELS_DIR, SEED

TOTAL_TIMESTEPS = 500_000
LOG_DIR  = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs", "combat")
SAVE_DIR = MODELS_DIR


def main():
    os.makedirs(LOG_DIR,  exist_ok=True)
    os.makedirs(SAVE_DIR, exist_ok=True)

    env = Monitor(CombatEnv())

    # Сохранять чекпоинт каждые 10 000 шагов
    checkpoint_cb = CheckpointCallback(
        save_freq=10_000,
        save_path=SAVE_DIR,
        name_prefix="combat_ppo",
    )

    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        seed=SEED,
        tensorboard_log=LOG_DIR,
        # Гиперпараметры (можно настроить)
        n_steps=512,
        batch_size=64,
        n_epochs=10,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
    )

    print(f"[train_combat] Начало обучения: {TOTAL_TIMESTEPS} шагов")
    print(f"[train_combat] TensorBoard: tensorboard --logdir {LOG_DIR}")

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=checkpoint_cb,
        progress_bar=True,
    )

    final_path = os.path.join(SAVE_DIR, "combat_ppo")
    model.save(final_path)
    print(f"[train_combat] Модель сохранена: {final_path}.zip")


if __name__ == "__main__":
    main()
