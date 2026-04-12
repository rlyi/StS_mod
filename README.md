# Slay the Spire AI

Автономный ИИ-агент для Slay the Spire на базе обучения с подкреплением.

## Архитектура

Агент состоит из двух компонентов:

**CombatAgent** — боевой агент (PPO via Stable-Baselines3)  
Принимает решения в бою: какую карту сыграть и на какого врага.

**MetaAgent** — мета-агент (Decision Tree + if/else правила)  
Навигация по карте, выбор карт в награду, магазин, события, отдых.

```
Slay the Spire
      │
CommunicationMod (Java)
      │  stdin/stdout JSON
      ▼
main.py  ──►  screen == NONE?  ──►  CombatAgent (PPO)
                    │
                    └── иначе ──►  MetaAgent (Decision Tree)
```

## Ограничения (текущая версия)

- Персонаж: **Ironclad**
- Акт: **1** (этажи 1–17)
- Сложность: **Ascension 0**
- Сид: **42** (воспроизводимость)

## Структура проекта

```
StS_mod/
├── main.py                 # Точка входа, коорди��ация агентов
├── config.py               # Константы: карты, интенты, пути
├── requirements.txt
├── agents/
│   ├── base_agent.py       # Абстрактный класс MetaAgent
│   ├── combat_agent.py     # CombatAgent: obs/action/PPO
│   └── meta_agent.py       # MetaAgent: навигация и экраны
├── environment/
│   ├── combat_env.py       # Gymnasium-окружение для PPO
│   └── reward.py           # Функция награды
├── training/
│   ├── train_combat.py     # Обучение PPO (500k шагов)
│   └── train_meta.py       # Обучение Decision Tree
└── models/                 # Сохранённые модели (не в git)
    ├── combat_ppo.zip
    └── meta_tree.pkl
```

## Установка

```bash
pip install spirecomm @ git+https://github.com/ForgottenArbiter/spirecomm.git
pip install stable-baselines3 scikit-learn gymnasium tensorboard numpy matplotlib
```

### Настройка CommunicationMod

В файле `%LOCALAPPDATA%\ModTheSpire\CommunicationMod\config.properties`:

```properties
command=python C:/StS_mod/main.py
runAtGameStart=true
```

> Важно: использовать прямые слэши. Java пе��езаписывает файл и экранирует обратные.

## Пространство состояний и действий

**Наблюдение (22 float32):**

| Индексы | Описание |
|---------|----------|
| 0 | HP игрока / max HP |
| 1 | Энергия / 3 |
| 2 | Блок / 100 |
| 3–12 | Карты в руке (до 5): card_id/100, cost/3 |
| 13–21 | Враги (до 3): hp_norm, intent_norm, block_norm |

**Действия (16 дискретных):**

| Действие | Смысл |
|----------|-------|
| 0–4 | Сыграть карту 0–4 без цели |
| 5–9 | Сыграть карту 0–4 на врага 0 |
| 10–14 | Сыграть карту 0–4 на врага 1 |
| 15 | Завершить ход |

## Обучение

```bash
# Боевой агент (PPO, ~500k шагов, нужна запущенная игра)
python training/train_combat.py

# Мета-агент (Decision Tree, по данным из runs/)
python training/train_meta.py
```

Ло��и TensorBoard:
```bash
tensorboard --logdir logs/combat
```

## Статус

| Этап | Описание | Статус |
|------|----------|--------|
| 1 | Подключение к игре (CommunicationMod) | ✅ Готово |
| 2 | Обучение CombatAgent (PPO) | 🔄 В работе |
| 3 | Обучение MetaAgent (Decision Tree) | ⏳ Ожидает |
| 4 | Оценка и итерация | ⏳ Ожидает |

## Технические детали

- **CommunicationMod → Python**: JSON через stdin/stdout. `stdout` занят протоколом — весь вывод только в `logs/ai.log`.
- **signal_ready()** вызывается до за��рузки тяжёлых зависимостей (SB3, sklearn) — CommunicationMod даёт только 10 секунд на старт.
- **Таргетинг**: `play X Y` где Y — 0-based индекс монстра в массиве `game.monsters`.
- **Neow event** (floor 0): экран `EVENT` не принимает команду `choose` — агент ожидает ручного выбора.
