"""dashboard.py — Streamlit-дашборд для StS AI.

Запуск:
    python -m streamlit run dashboard.py
"""

import importlib
import json
import os
import re

import pandas as pd
import streamlit as st

import config as _cfg

_ROOT        = os.path.dirname(os.path.abspath(__file__))
_CONFIG_PATH = os.path.join(_ROOT, "config.py")
_CONFIGS_DIR = os.path.join(_ROOT, "configs")
_ACTIVE_FILE = os.path.join(_CONFIGS_DIR, ".active")
_BASE_NAME   = "base"

os.makedirs(_CONFIGS_DIR, exist_ok=True)


# ── Управление профилями ──────────────────────────────────────────────────────

def _list_profiles() -> list[str]:
    names = [f[:-5] for f in os.listdir(_CONFIGS_DIR) if f.endswith(".json")]
    names = sorted(n for n in names if n != _BASE_NAME)
    return [_BASE_NAME] + names

def _load_profile(name: str) -> dict:
    path = os.path.join(_CONFIGS_DIR, f"{name}.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def _save_profile(name: str, data: dict):
    path = os.path.join(_CONFIGS_DIR, f"{name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def _delete_profile(name: str):
    path = os.path.join(_CONFIGS_DIR, f"{name}.json")
    if os.path.exists(path):
        os.remove(path)

def _get_active() -> str:
    if os.path.exists(_ACTIVE_FILE):
        with open(_ACTIVE_FILE, encoding="utf-8") as f:
            name = f.read().strip()
        if os.path.exists(os.path.join(_CONFIGS_DIR, f"{name}.json")):
            return name
    return _BASE_NAME

def _set_active(name: str):
    with open(_ACTIVE_FILE, "w", encoding="utf-8") as f:
        f.write(name)

def _apply_to_config_py(data: dict):
    """Записывает значения профиля в config.py."""
    with open(_CONFIG_PATH, encoding="utf-8") as f:
        text = f.read()

    # BENCHMARK_MODE
    text = re.sub(
        r"^(BENCHMARK_MODE\s*=\s*)\w+",
        lambda m: f"{m.group(1)}{data['BENCHMARK_MODE']}",
        text, flags=re.MULTILINE,
    )

    # Числовые параметры
    scalar_keys = [
        "PATH_FIGHT_REWARD", "PATH_ELITE_REWARD", "PATH_RELIC_REWARD",
        "PATH_CURSE_LOSS", "PATH_UPGRADE_REWARD", "PATH_EVENT_REWARD_A1",
        "PATH_EVENT_REWARD_A2", "PATH_GOLD_SHOP_DIV", "PATH_GOLD_END_DIV",
        "PATH_SURVIVABILITY_K", "HP_HEAL_THRESHOLD", "HP_HEAL_THRESHOLD_BOSS",
    ]
    for key in scalar_keys:
        val = data.get(key)
        if val is None:
            continue
        text = re.sub(
            rf"^({re.escape(key)}\s*=\s*)[\d.]+",
            lambda m, v=val: f"{m.group(1)}{v}",
            text, flags=re.MULTILINE,
        )

    # TARGET_DECK
    deck = data.get("TARGET_DECK", {})
    deck_lines = ["TARGET_DECK: dict[str, int] = {\n"]
    for card, copies in deck.items():
        deck_lines.append(f"    {repr(card)}: {copies},\n")
    deck_lines.append("}")
    new_deck = "".join(deck_lines)
    text = re.sub(
        r"TARGET_DECK: dict\[str, int\] = \{[^}]*\}",
        new_deck,
        text, flags=re.DOTALL,
    )

    with open(_CONFIG_PATH, "w", encoding="utf-8") as f:
        f.write(text)


# ── Страница ──────────────────────────────────────────────────────────────────

st.set_page_config(page_title="StS AI Dashboard", layout="wide")
st.title("Slay the Spire — AI Dashboard")

importlib.reload(_cfg)

# ── Панель профилей (всегда видна) ───────────────────────────────────────────

profiles = _list_profiles()
active   = _get_active()

st.subheader("Профиль конфига")
col_sel, col_new, col_del, col_apply = st.columns([3, 2, 2, 2])

with col_sel:
    selected = st.selectbox("Выбрать профиль", profiles,
                            index=profiles.index(active) if active in profiles else 0,
                            label_visibility="collapsed")

with col_new:
    new_name = st.text_input("Название", placeholder="новый профиль",
                             label_visibility="collapsed", key="new_profile_name")
with col_del:
    can_delete = selected != _BASE_NAME
    if st.button("Удалить", disabled=not can_delete,
                 help="Нельзя удалить базовый профиль"):
        _delete_profile(selected)
        if _get_active() == selected:
            _set_active(_BASE_NAME)
            _apply_to_config_py(_load_profile(_BASE_NAME))
        st.rerun()

with col_apply:
    if st.button("Создать копию", disabled=not new_name.strip()):
        name = new_name.strip()
        if name not in profiles:
            _save_profile(name, _load_profile(selected))
        _set_active(name)
        _apply_to_config_py(_load_profile(name))
        st.rerun()

is_base = (selected == _BASE_NAME)
if selected != active:
    if st.button(f"Применить профиль «{selected}»", type="primary"):
        _set_active(selected)
        _apply_to_config_py(_load_profile(selected))
        importlib.reload(_cfg)
        st.success(f"Профиль «{selected}» применён — перезапусти main.py.")
        st.rerun()

if is_base:
    st.info("Базовый профиль — только для чтения. Создай копию чтобы редактировать.")

st.divider()

# ── Вкладки ───────────────────────────────────────────────────────────────────

tab_results, tab_config, tab_log = st.tabs(["Результаты", "Конфиг", "Лог"])


# ── Вкладка 1: Результаты ────────────────────────────────────────────────────

with tab_results:
    importlib.reload(_cfg)
    results_file = _cfg.BENCHMARK_RESULTS_FILE

    if not os.path.exists(results_file):
        st.info(f"Файл результатов не найден: `{results_file}`\n\nВключи бенчмарк в конфиге и запусти main.py.")
    else:
        with open(results_file, encoding="utf-8") as f:
            results = json.load(f)

        if not results:
            st.info("Результатов пока нет.")
        else:
            df    = pd.DataFrame(results)
            total = len(df)
            wins  = int(df["win"].sum())

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Забегов",       total)
            c2.metric("Побед",         wins)
            c3.metric("Винрейт",       f"{wins / total * 100:.1f}%")
            c4.metric("Средний этаж",  f"{df['floor'].mean():.1f}")
            c5.metric("Сборка колоды", f"{df['deck_completion'].mean() * 100:.1f}%")

            st.divider()
            col_l, col_r = st.columns(2)
            with col_l:
                st.subheader("Распределение этажей")
                st.bar_chart(df["floor"].value_counts().sort_index())
            with col_r:
                st.subheader("Сборка колоды, %")
                deck_pct = (df["deck_completion"] * 100).round(0).astype(int)
                st.bar_chart(deck_pct.value_counts().sort_index())

            st.divider()
            st.subheader("Все забеги")
            display = df[["seed", "floor", "win", "deck_completion"]].copy()
            display["deck_completion"] = (display["deck_completion"] * 100).map("{:.0f}%".format)
            display["win"] = display["win"].map({True: "✓", False: "✗"})
            st.dataframe(display, use_container_width=True, hide_index=True)


# ── Вкладка 2: Конфиг ────────────────────────────────────────────────────────

with tab_config:
    profile_data = _load_profile(selected)

    st.subheader("Режим")
    bm = st.toggle("Бенчмарк (BENCHMARK_MODE)",
                   value=bool(profile_data.get("BENCHMARK_MODE", False)),
                   disabled=is_base)

    st.divider()
    st.subheader("Коэффициенты маршрута")

    def _num(label, key, lo=0.0, hi=5.0, step=0.1):
        return st.number_input(label, min_value=lo, max_value=hi,
                               value=float(profile_data.get(key, 1.0)),
                               step=step, key=f"cfg_{key}", disabled=is_base)

    col1, col2 = st.columns(2)
    with col1:
        pf   = _num("Обычный бой",            "PATH_FIGHT_REWARD")
        pe   = _num("Элита (без реликта)",     "PATH_ELITE_REWARD")
        pr   = _num("Реликт (элита/сундук)",   "PATH_RELIC_REWARD")
        pcl  = _num("Штраф Cursed Key",        "PATH_CURSE_LOSS")
        pu   = _num("Костёр (апгрейд)",        "PATH_UPGRADE_REWARD")
    with col2:
        pea1 = _num("Событие (акт 1)",         "PATH_EVENT_REWARD_A1")
        pea2 = _num("Событие (акты 2–3)",      "PATH_EVENT_REWARD_A2")
        pgs  = _num("Делитель золота магазин", "PATH_GOLD_SHOP_DIV", 10.0, 500.0, 10.0)
        pge  = _num("Делитель золота остаток", "PATH_GOLD_END_DIV",  10.0, 500.0, 10.0)
        psk  = _num("Штраф выживаемости K",   "PATH_SURVIVABILITY_K", 0.0, 50.0, 1.0)

    st.divider()
    st.subheader("HP-пороги лечения")
    col3, col4 = st.columns(2)
    with col3:
        hp_t  = st.slider("Обычный порог", 0.0, 1.0,
                          float(profile_data.get("HP_HEAL_THRESHOLD", 0.6)), 0.05,
                          key="cfg_hp", disabled=is_base)
    with col4:
        hp_tb = st.slider("Перед боссом", 0.0, 1.0,
                          float(profile_data.get("HP_HEAL_THRESHOLD_BOSS", 0.85)), 0.05,
                          key="cfg_hp_boss", disabled=is_base)

    st.divider()
    st.subheader("Целевая колода")
    deck_df = pd.DataFrame(
        [{"Карта": k, "Макс. копий": v}
         for k, v in profile_data.get("TARGET_DECK", {}).items()]
    )
    edited_deck = st.data_editor(
        deck_df, num_rows="dynamic" if not is_base else "fixed",
        use_container_width=True, hide_index=True,
        key="deck_editor", disabled=is_base,
    )

    st.divider()
    if not is_base:
        if st.button("Сохранить и применить", type="primary"):
            new_deck = {
                str(row["Карта"]).strip().lower(): int(row["Макс. копий"])
                for _, row in edited_deck.iterrows()
                if pd.notna(row["Карта"]) and pd.notna(row["Макс. копий"])
            }
            updated = {
                "PATH_FIGHT_REWARD":    pf,  "PATH_ELITE_REWARD":    pe,
                "PATH_RELIC_REWARD":    pr,  "PATH_CURSE_LOSS":      pcl,
                "PATH_UPGRADE_REWARD":  pu,  "PATH_EVENT_REWARD_A1": pea1,
                "PATH_EVENT_REWARD_A2": pea2,"PATH_GOLD_SHOP_DIV":   pgs,
                "PATH_GOLD_END_DIV":    pge, "PATH_SURVIVABILITY_K": psk,
                "HP_HEAL_THRESHOLD":    hp_t,"HP_HEAL_THRESHOLD_BOSS": hp_tb,
                "BENCHMARK_MODE":       bm,  "TARGET_DECK":          new_deck,
            }
            _save_profile(selected, updated)
            _set_active(selected)
            _apply_to_config_py(updated)
            importlib.reload(_cfg)
            st.success("Сохранено и применено. Перезапусти main.py.")
            st.rerun()


# ── Вкладка 3: Лог ───────────────────────────────────────────────────────────

with tab_log:
    log_choice = st.radio("Файл", ["benchmark.log", "ai.log"], horizontal=True)
    log_path   = os.path.join(_ROOT, "logs", log_choice)
    n_lines    = st.slider("Последних строк", 20, 500, 100, step=20, key="log_lines")

    col_btn, _ = st.columns([1, 5])
    with col_btn:
        st.button("Обновить", key="log_refresh")

    if not os.path.exists(log_path):
        st.info(f"Файл не найден: `{log_path}`")
    else:
        with open(log_path, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        st.text_area("", value="".join(lines[-n_lines:]), height=500, key="log_content")
