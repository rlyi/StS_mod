# Slay the Spire AI

Автономный ИИ-агент для Slay the Spire на базе обучения с подкреплением + Decision Tree.  
Разрабатывается на Windows, запускается через CommunicationMod.

## Архитектура

```
Slay the Spire
      │
CommunicationMod (Java)
      │  stdin/stdout JSON  (stdout занят протоколом — весь вывод в logs/ai.log)
      ▼
main.py
  ├── screen == COMBAT  →  CombatAgent  (PPO, Stable-Baselines3)
  └── иначе             →  MetaAgent    (Decision Tree / Random Forest / правила)
```

**CombatAgent** — принимает решения в бою: какую карту сыграть, на кого, когда завершить ход, применить ли зелье.

**MetaAgent** — всё вне боя: выбор пути на карте, карты в награду, магазин, события, костёр, босс-реликвия.

Три реализации MetaAgent (переключается в `config.py` → `META_AGENT`):
- `"random"` — случайный baseline (`RandomMetaAgent`)
- `"tree"` — Decision Tree, обучен на датасете SlayTheData (`DecisionTreeMetaAgent`)
- `"forest"` — Random Forest, аналогичен DT но с sample_weight по победам (`RandomForestMetaAgent`)

## Ограничения (текущая версия)

| Параметр | Значение |
|----------|----------|
| Персонаж | Ironclad |
| Акт | 1 (этажи 1–17) |
| Сложность | Ascension 0 |
| Сид по умолчанию | 42 |

## Структура проекта

```
StS_mod/
├── main.py                        # Точка входа: координация агентов
├── benchmark.py                   # Сравнение мета-агентов (random/tree/forest × сиды)
├── config.py                      # Константы: OBS_SIZE=102, ACTION_SIZE=51, CARD_PROPERTIES, …
├── requirements.txt
├── agents/
│   ├── base_agent.py              # Абстрактный интерфейс BaseMetaAgent
│   ├── meta_agent.py              # RandomMetaAgent (baseline)
│   ├── meta_tree_agent.py         # DecisionTreeMetaAgent + жёсткие правила событий
│   └── meta_forest_agent.py       # RandomForestMetaAgent (наследует DT, другие pkl-пути)
├── environment/
│   ├── combat_env.py              # Gymnasium-обёртка: два потока (spirecomm + SB3)
│   └── reward.py                  # Функция награды
├── training/
│   ├── train_combat.py            # Обучение CombatAgent (PPO, 100k шагов, чекпоинты каждые 5k)
│   ├── train_meta.py              # Обучение DT по данным из data/runs/ (собственные логи)
│   ├── train_meta_slaythedata.py  # Обучение DT на датасете SlayTheData (лучшее качество)
│   └── train_meta_slaythedata_rf.py  # RF-вариант, sample_weight по победам
├── models/
│   ├── combat_ppo_*_steps.zip     # Чекпоинты PPO (5k–90k шагов, не в git)
│   ├── meta_card_slaythedata_dt.pkl
│   ├── meta_campfire_slaythedata_dt.pkl
│   ├── meta_path_slaythedata_dt.pkl
│   ├── meta_event_slaythedata_dt.pkl
│   ├── meta_shop_slaythedata_dt.pkl
│   └── meta_*_slaythedata_rf.pkl  # RF-версии тех же моделей
└── data/
    ├── benchmark_results.json     # Результаты benchmark.py
    └── cluster_the_spire.zip      # Датасет SlayTheData (Kaggle)
```

## Пространство наблюдений (OBS_SIZE = 102)

| Индексы | Описание |
|---------|----------|
| 0–13 | Игрок: hp, energy, block, strength, dexterity, vulnerable, weak, poison, metallicize, corruption, barricade, draw_pile, discard_size, turn |
| 14–69 | Рука (до 7 карт × 8): is_attack, is_skill, is_power, is_other, dmg/20, blk/20, cost/3, is_upgraded |
| 70–89 | Враги (до 4 × 5): hp_norm, intent_norm, block_norm, damage_norm, ritual_norm |
| 90–101 | Зелья (3 слота × 4): present, is_heal, is_attack, is_utility |

## Пространство действий (ACTION_SIZE = 51)

| Диапазон | Смысл |
|----------|-------|
| 0–6 | Карта 0–6 без цели (скиллы, пауэры) |
| 7–13 | Карта 0–6 → враг 0 |
| 14–20 | Карта 0–6 → враг 1 |
| 21–27 | Карта 0–6 → враг 2 |
| 28–34 | Карта 0–6 → враг 3 |
| 35 | Завершить ход |
| 36–40 | Зелье слот 0 (36=без цели, 37–40=враги 0–3) |
| 41–45 | Зелье слот 1 |
| 46–50 | Зелье слот 2 |

## Функция награды

| Событие | Значение |
|---------|----------|
| Победа в бою | +2.0 + hp_pct × 1.0 |
| Убийство врага | +0.5 |
| Урон врагу | +0.03 × min(dmg, enemy_hp) |
| Смерть | −2.0 |
| Урон игроку | −0.03 × dmg |
| Каждый ход | −0.01 |
| Победа над боссом Акт 1 | +10.0 |

## Установка

```bash
pip install spirecomm @ git+https://github.com/ForgottenArbiter/spirecomm.git
pip install stable-baselines3 scikit-learn gymnasium tensorboard numpy matplotlib
```

### Настройка CommunicationMod

Файл `%LOCALAPPDATA%\ModTheSpire\CommunicationMod\config.properties`:

```properties
command=python C:/StS_mod/main.py
runAtGameStart=true
```

> Важно: прямые слэши. Java перезаписывает файл и экранирует обратные.

## Запуск

```bash
# Играть (требует запущенной игры с CommunicationMod)
python main.py

# Benchmark: сравнение агентов на нескольких сидах
python benchmark.py
# Настройки AGENT / SEEDS / RUNS_PER_SEED — в начале файла

# TensorBoard
tensorboard --logdir logs/combat
```

## Обучение

```bash
# Боевой агент PPO (100k шагов, чекпоинты каждые 5k, нужна живая игра)
python training/train_combat.py

# Мета-агент: Decision Tree на датасете SlayTheData
python training/train_meta_slaythedata.py

# Мета-агент: Random Forest (sample_weight по победам)
python training/train_meta_slaythedata_rf.py
```

Датасет `data/cluster_the_spire.zip` — SlayTheData с Kaggle, содержит реальные пробежки.

## Технические детали

- **Два потока в CombatEnv**: фоновый поток читает stdin/пишет stdout (spirecomm Coordinator), основной поток — SB3 обучение. Общение через `state_q` / `action_q`.
- **signal_ready()** вызывается до загрузки SB3/sklearn — CommunicationMod даёт 10 секунд на старт.
- **Таргетинг**: `play X Y`, где Y — `monster_index` из массива `game.monsters`. Живые монстры фильтруются через `is_gone` и `current_hp > 0`.
- **Neow event** (floor 0): экран `EVENT` не принимает `choose` — ожидать ручного выбора.
- **Модели загружаются автоматически**: CombatAgent ищет `combat_ppo.zip`, если нет — берёт последний чекпоинт `combat_ppo_N_steps.zip`.

## Статус

| Компонент | Статус |
|-----------|--------|
| CommunicationMod интеграция | ✅ Готово |
| CombatAgent (PPO) | 🔄 Обучается (90k шагов сохранено) |
| MetaAgent Decision Tree | ✅ Готово (SlayTheData датасет) |
| MetaAgent Random Forest | ✅ Готово (sample_weight по победам) |
| Benchmark runner | ✅ Готово |
| MetaAgent с RL | ⏳ Планируется |
| Act 2+ | ⏳ Планируется |
