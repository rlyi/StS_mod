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

# ── Профили: вспомогательные функции ─────────────────────────────────────────

def _list_profiles() -> list[str]:
    names = sorted(
        f[:-5] for f in os.listdir(_CONFIGS_DIR)
        if f.endswith(".json") and f[:-5] != _BASE_NAME
    )
    return [_BASE_NAME] + names

@st.cache_data
def _load_profile(name: str) -> dict:
    with open(os.path.join(_CONFIGS_DIR, f"{name}.json"), encoding="utf-8") as f:
        return json.load(f)

def _save_profile(name: str, data: dict):
    with open(os.path.join(_CONFIGS_DIR, f"{name}.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    _load_profile.clear()

def _delete_profile(name: str):
    path = os.path.join(_CONFIGS_DIR, f"{name}.json")
    if os.path.exists(path):
        os.remove(path)
    _load_profile.clear()

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

def _make_seeds(n: int) -> list[int]:
    return [(i + 1) * 101 for i in range(n)]

def _apply_to_config_py(data: dict):
    with open(_CONFIG_PATH, encoding="utf-8") as f:
        text = f.read()

    text = re.sub(
        r"^(BENCHMARK_MODE\s*=\s*)\w+",
        lambda m: f"{m.group(1)}{data['BENCHMARK_MODE']}",
        text, flags=re.MULTILINE,
    )

    for key in [
        "PATH_FIGHT_REWARD", "PATH_ELITE_REWARD", "PATH_RELIC_REWARD",
        "PATH_CURSE_LOSS", "PATH_UPGRADE_REWARD", "PATH_EVENT_REWARD_A1",
        "PATH_EVENT_REWARD_A2", "PATH_GOLD_SHOP_DIV", "PATH_GOLD_END_DIV",
        "PATH_SURVIVABILITY_K", "HP_HEAL_THRESHOLD", "HP_HEAL_THRESHOLD_BOSS",
    ]:
        val = data.get(key)
        if val is None:
            continue
        text = re.sub(
            rf"^({re.escape(key)}\s*=\s*)[\d.]+",
            lambda m, v=val: f"{m.group(1)}{v}",
            text, flags=re.MULTILINE,
        )

    seeds = _make_seeds(int(data.get("BENCHMARK_SEEDS_COUNT", 100)))
    rows  = [seeds[i:i+10] for i in range(0, len(seeds), 10)]
    seeds_str = "BENCHMARK_SEEDS         = [\n"
    for row in rows:
        seeds_str += "    " + ", ".join(str(s) for s in row) + ",\n"
    seeds_str += "]"
    text = re.sub(
        r"BENCHMARK_SEEDS\s*=\s*\[[^\]]*\]",
        seeds_str, text, flags=re.DOTALL,
    )

    deck  = data.get("TARGET_DECK", {})
    lines = ["TARGET_DECK: dict[str, int] = {\n"]
    for card, copies in deck.items():
        lines.append(f"    {repr(card)}: {copies},\n")
    lines.append("}")
    text = re.sub(
        r"TARGET_DECK: dict\[str, int\] = \{[^}]*\}",
        "".join(lines), text, flags=re.DOTALL,
    )

    with open(_CONFIG_PATH, "w", encoding="utf-8") as f:
        f.write(text)

# ── Валидация названия профиля ────────────────────────────────────────────────

_NAME_RE = re.compile(r'^[a-zA-Zа-яёА-ЯЁ0-9][a-zA-Zа-яёА-ЯЁ0-9 _-]*$')

def _validate_name(name: str, existing: list[str]) -> str | None:
    if not name:
        return None
    if len(name) > 15:
        return "Максимум 15 символов"
    if not _NAME_RE.match(name):
        return "Только буквы, цифры, пробел, _ и -"
    if "  " in name or "__" in name or "--" in name:
        return "Нельзя два одинаковых символа подряд"
    if name in existing:
        return "Такое название уже занято"
    return None

# ── Диалог удаления ───────────────────────────────────────────────────────────

@st.dialog("Удалить профиль")
def _delete_dialog(name: str):
    st.write(f"Удалить профиль **«{name}»**?")
    st.caption("Это действие нельзя отменить.")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Удалить", type="primary", use_container_width=True):
            _delete_profile(name)
            if _get_active() == name:
                _set_active(_BASE_NAME)
                _apply_to_config_py(_load_profile(_BASE_NAME))
            st.rerun()
    with c2:
        if st.button("Отмена", use_container_width=True):
            st.rerun()

# ── Страница ──────────────────────────────────────────────────────────────────

st.set_page_config(page_title="StS AI Dashboard", layout="wide")
st.title("Slay the Spire — AI Dashboard")

profiles = _list_profiles()
active   = _get_active()

# ── Панель профилей ───────────────────────────────────────────────────────────

st.subheader("Профиль конфига")

col_sel, col_right = st.columns([2, 3])

with col_sel:
    selected = st.selectbox(
        "Профиль", profiles,
        index=profiles.index(active) if active in profiles else 0,
        label_visibility="collapsed",
    )
    sel_cols = st.columns([2, 1])
    with sel_cols[0]:
        if selected != active and selected != _BASE_NAME:
            if st.button("Применить", type="primary", use_container_width=True):
                _set_active(selected)
                _apply_to_config_py(_load_profile(selected))
                importlib.reload(_cfg)
                st.rerun()
        elif selected == active:
            st.caption("✓ Активный профиль")
    with sel_cols[1]:
        if selected != _BASE_NAME:
            if st.button("Удалить", use_container_width=True):
                _delete_dialog(selected)

with col_right:
    new_name      = st.text_input(
        "Новый профиль", placeholder="название...",
        max_chars=15, label_visibility="collapsed",
    )
    name_stripped = new_name.strip()
    error         = _validate_name(name_stripped, profiles)

    st.caption(f"⚠ {error}" if (name_stripped and error) else " ")

    btn_c1, btn_c2 = st.columns(2)
    with btn_c1:
        if st.button("Создать", use_container_width=True,
                     help="Создать новый профиль с базовыми настройками"):
            if name_stripped and not error:
                _save_profile(name_stripped, _load_profile(_BASE_NAME))
                _set_active(name_stripped)
                _apply_to_config_py(_load_profile(name_stripped))
                st.rerun()
    with btn_c2:
        if st.button("Копировать", use_container_width=True,
                     help=f"Скопировать профиль «{selected}»"):
            if name_stripped and not error:
                _save_profile(name_stripped, _load_profile(selected))
                _set_active(name_stripped)
                _apply_to_config_py(_load_profile(name_stripped))
                st.rerun()

if selected == _BASE_NAME:
    st.caption("Базовый профиль — только для чтения.")

st.divider()

# ── Вкладки ───────────────────────────────────────────────────────────────────

tab_results, tab_config = st.tabs(["Результаты", "Конфиг"])

# ── Вкладка 1: Результаты ─────────────────────────────────────────────────────

with tab_results:
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
    is_base      = (selected == _BASE_NAME)

    st.subheader("Режим")
    bm = st.toggle("Бенчмарк", value=bool(profile_data.get("BENCHMARK_MODE", False)),
                   disabled=is_base)
    if bm:
        seeds_count = st.slider(
            "Количество сидов", min_value=10, max_value=200,
            value=int(profile_data.get("BENCHMARK_SEEDS_COUNT", 100)),
            step=10, disabled=is_base,
            help="Сиды генерируются автоматически: 101, 202, 303 ...",
        )
    else:
        seeds_count = int(profile_data.get("BENCHMARK_SEEDS_COUNT", 100))

    st.divider()

    st.subheader("Целевая колода")
    deck_df = pd.DataFrame(
        [{"Карта": k, "Макс. копий": v}
         for k, v in profile_data.get("TARGET_DECK", {}).items()]
    )
    edited_deck = st.data_editor(
        deck_df,
        num_rows="dynamic" if not is_base else "fixed",
        use_container_width=True,
        hide_index=True,
        key="deck_editor",
        disabled=is_base,
    )

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
        hp_tb = st.slider("Перед боссом",  0.0, 1.0,
                          float(profile_data.get("HP_HEAL_THRESHOLD_BOSS", 0.85)), 0.05,
                          key="cfg_hp_boss", disabled=is_base)

    st.divider()

    if not is_base:
        if st.button("Сохранить и применить", type="primary"):
            new_deck = {
                str(row["Карта"]).strip().lower(): int(row["Макс. копий"])
                for _, row in edited_deck.iterrows()
                if pd.notna(row["Карта"]) and pd.notna(row["Макс. копий"])
            }
            updated = {
                "BENCHMARK_MODE":        bm,
                "BENCHMARK_SEEDS_COUNT": seeds_count,
                "TARGET_DECK":           new_deck,
                "PATH_FIGHT_REWARD":     pf,  "PATH_ELITE_REWARD":    pe,
                "PATH_RELIC_REWARD":     pr,  "PATH_CURSE_LOSS":      pcl,
                "PATH_UPGRADE_REWARD":   pu,  "PATH_EVENT_REWARD_A1": pea1,
                "PATH_EVENT_REWARD_A2":  pea2,"PATH_GOLD_SHOP_DIV":   pgs,
                "PATH_GOLD_END_DIV":     pge, "PATH_SURVIVABILITY_K": psk,
                "HP_HEAL_THRESHOLD":     hp_t,"HP_HEAL_THRESHOLD_BOSS": hp_tb,
            }
            _save_profile(selected, updated)
            _set_active(selected)
            _apply_to_config_py(updated)
            importlib.reload(_cfg)
            st.success("Сохранено и применено. Перезапусти main.py.")
            st.rerun()
