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

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

# ── Логирование в файл (stdout/stderr заняты протоколом CommunicationMod) ──
import logging

_LOG_DIR = os.path.join(_ROOT, "logs")
os.makedirs(_LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(_LOG_DIR, "train_combat.log"), encoding="utf-8"),
    ],
)
log = logging.getLogger("train_combat")

# Перенаправляем stderr в файл (stdout уже занят протоколом)
sys.stderr = open(os.path.join(_LOG_DIR, "train_errors.log"), "a", encoding="utf-8")

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from gymnasium.wrappers import TimeLimit

from environment.combat_env import CombatEnv
from config import MODELS_DIR, SEED

TOTAL_TIMESTEPS = 500_000
MAX_EPISODE_STEPS = 200  # принудительно завершать бесконечные бои
LOG_DIR  = os.path.join(_ROOT, "logs", "combat")
SAVE_DIR = MODELS_DIR


def _latest_checkpoint(save_dir: str):
    """Вернуть путь к последнему чекпоинту или None."""
    import glob
    files = glob.glob(os.path.join(save_dir, "combat_ppo_*_steps.zip"))
    if not files:
        return None
    # Сортируем по номеру шага в имени файла
    files.sort(key=lambda p: int(os.path.basename(p).split("_")[2]))
    return files[-1]


def main():
    os.makedirs(LOG_DIR,  exist_ok=True)
    os.makedirs(SAVE_DIR, exist_ok=True)

    log.info("Инициализация окружения...")
    env = ActionMasker(
        Monitor(TimeLimit(CombatEnv(), max_episode_steps=MAX_EPISODE_STEPS)),
        lambda e: e.unwrapped.action_masks(),
    )

    # Сохранять чекпоинт каждые 10 000 шагов
    checkpoint_cb = CheckpointCallback(
        save_freq=5_000,
        save_path=SAVE_DIR,
        name_prefix="combat_ppo",
    )

    # Загружаем чекпоинт если есть, иначе создаём новую модель
    checkpoint = _latest_checkpoint(SAVE_DIR)
    if checkpoint:
        log.info("Дообучение с чекпоинта: %s", checkpoint)
        model = MaskablePPO.load(
            checkpoint,
            env=env,
            device="cpu",
            tensorboard_log=LOG_DIR,
        )
    else:
        log.info("Чекпоинт не найден — начинаем с нуля")
        model = MaskablePPO(
            policy="MlpPolicy",
            env=env,
            verbose=0,
            seed=SEED,
            device="cpu",
            tensorboard_log=LOG_DIR,
            policy_kwargs={"net_arch": [128, 128]},
            n_steps=2048,
            batch_size=256,
            n_epochs=10,
            learning_rate=3e-4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
        )

    log.info("Начало обучения: %d шагов", TOTAL_TIMESTEPS)
    log.info("TensorBoard: tensorboard --logdir %s", LOG_DIR)

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=checkpoint_cb,
        progress_bar=False,       # прогресс-бар пишет в stdout
        reset_num_timesteps=checkpoint is None,  # продолжаем счётчик при дообучении
    )

    final_path = os.path.join(SAVE_DIR, "combat_ppo")
    model.save(final_path)
    log.info("Модель сохранена: %s.zip", final_path)


if __name__ == "__main__":
    main()
