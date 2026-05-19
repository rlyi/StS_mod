# Slay the Spire — Automated Deck Testing Tool

Инструмент для автоматического тестирования стратегий сборки колоды в Slay the Spire.  
Запускается через CommunicationMod, работает на Windows.

## Концепция

Система разделена на два агента:

- **GraphBattleAgent** — детерминированный BFS-агент для боя. Перебирает все возможные последовательности карт за ход и выбирает оптимальную по иерархии критериев. Не зависит от стратегии — оптимально разыгрывает любую колоду.
- **RuleMetaAgent** — правило-based агент для мета-игры. Полностью управляется через `config.py`: собирает целевую колоду (`TARGET_DECK`), выбирает путь на карте, управляет магазином, кострами, событиями, зельями и реликтами.

Такое разделение позволяет **изолировать боевую составляющую** — разница в результатах бенчмарка отражает только качество стратегии колоды, а не мастерство в бою.

## Архитектура

```
Slay the Spire
      │
CommunicationMod (Java)
      │  stdin/stdout JSON
      ▼
main.py
  ├── screen == COMBAT  →  GraphBattleAgent  (BFS, детерминированный оптимум)
  └── иначе             →  RuleMetaAgent     (правила из config.py)
```

## Настройка стратегии (config.py)

Все параметры стратегии задаются в `config.py` без изменения кода агентов.

### Целевая колода
```python
TARGET_DECK: dict[str, int] = {
    'perfected strike': 5,   # карта → максимум копий
    'battle trance':    2,
    'offering':         1,
    ...
}
```

### Приоритеты действий
```python
CAMPFIRE_PRIORITY: list[str] = ['rest', 'smith', 'toke', 'lift', 'dig']
SHOP_PRIORITY:     list[str] = ['relics', 'cards', 'purge']
REMOVAL_PRIORITY:  list[str] = ['defend', 'strike']
UPGRADE_PRIORITY:  list[str] = ['apotheosis', 'perfected strike', 'bash', ...]
BOSS_RELIC_PRIORITY: list[str] = ['sozu', 'runic dome', ...]
DESIRED_POTIONS:   list[str] = ['fruit juice', 'fairy in a bottle', ...]
```

### Пороги
```python
HP_HEAL_THRESHOLD:      float = 0.60   # лечиться если HP% ниже этого
HP_HEAL_THRESHOLD_BOSS: float = 0.85   # порог перед финальным боссом
```

### Коэффициенты выбора пути на карте
```python
PATH_FIGHT_REWARD:    float = 1.0   # обычный бой (M)
PATH_ELITE_REWARD:    float = 1.0   # элита
PATH_RELIC_REWARD:    float = 1.5   # реликт (элита, сундук)
PATH_UPGRADE_REWARD:  float = 1.1   # костёр
PATH_EVENT_REWARD_A1: float = 1.0   # событие в акте 1
PATH_EVENT_REWARD_A2: float = 1.5   # событие в актах 2–3
PATH_SURVIVABILITY_K: float = 15.0  # штраф за опасные пути
```

## Бенчмарк

```python
BENCHMARK_MODE          = True   # включить бенчмарк
BENCHMARK_RUNS_PER_SEED = 1      # забегов на сид
BENCHMARK_SEEDS         = [101, 202, ...]  # 100 сидов
```

Результаты сохраняются в `benchmark_rule.json`:
```json
{
  "seed": 101,
  "floor": 51,
  "win": true,
  "deck_completion": 0.85,
  "deck_at_end": ["Perfected Strike", "Perfected Strike+", ...]
}
```

**Метрики:** винрейт (полное прохождение 3 актов), средний этаж, процент сборки целевой колоды.

## Боевой агент: как работает BFS

1. Из текущего состояния (рука, энергия, монстры, паверы) генерируются все возможные последовательности карт — до 11 000 уникальных состояний
2. Каждое конечное состояние оценивается иерархией из 29 критериев (не умереть → выиграть → минимум урона → убить монстров → ...)
3. Выбирается первый ход лучшего пути

Специальные компараторы для нестандартных врагов:
| Враг | Стратегия |
|------|-----------|
| Gremlin Nob | Не тянуть карты скиллами (Nob усиливается) |
| Three Sentries | Убивать крайних монстров приоритетнее |
| Lagavulin | Ждать пробуждения (не атаковать первые 2 хода) |
| Transient | Не тратить ресурсы (умрёт сам через 3 хода) |
| Боссы (эт. 33, 50) | Приоритет накопления паверов |

## Установка

```bash
pip install spirecomm @ git+https://github.com/ForgottenArbiter/spirecomm.git
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
# Живая игра (требует запущенной игры с CommunicationMod)
python main.py

# Бенчмарк (BENCHMARK_MODE = True в config.py)
python main.py
```

## Структура проекта

```
StS_mod/
├── main.py                    # Точка входа: живая игра и BenchmarkRunner
├── config.py                  # Все параметры стратегии
├── agents/
│   ├── base_agent.py          # Базовый класс агентов
│   ├── graph_battle_agent.py  # BFS боевой агент
│   ├── meta_rule_agent.py     # Rule-based мета-агент
│   └── meta_llm_agent.py      # LLM мета-агент (альтернатива)
├── engine/
│   ├── battle_state.py        # Симулятор боя
│   ├── card_effects.py        # Эффекты всех Ironclad-карт
│   ├── comparators.py         # Иерархия критериев для BFS
│   ├── converter.py           # Конвертер game → BattleState
│   ├── memory.py              # Память между ходами (Rampage и др.)
│   └── play_path.py           # BFS по путям карт
└── logs/
    ├── ai.log                 # Лог живой игры
    └── benchmark.log          # Лог бенчмарка
```

## Персонаж и ограничения

| Параметр | Значение |
|----------|----------|
| Персонаж | Ironclad |
| Акты | 1–3 (полное прохождение) |
| Сложность | Ascension 0 |
| Случайные карты | True Grit, Armaments — не реализованы |
