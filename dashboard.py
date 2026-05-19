"""dashboard.py — Streamlit-дашборд для StS AI.

Запуск:
    streamlit run dashboard.py
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


st.set_page_config(page_title="StS AI Dashboard", layout="wide")
st.title("Slay the Spire — AI Dashboard")

tab_results, tab_config, tab_log = st.tabs(["Результаты", "Конфиг", "Лог"])


# ── Вкладка 1: Результаты ─────────────────────────────────────────────────────

with tab_results:
    importlib.reload(_cfg)
    results_file = _cfg.BENCHMARK_RESULTS_FILE

    if not os.path.exists(results_file):
        st.info(f"Файл результатов не найден: `{results_file}`\n\nЗапусти бенчмарк (`BENCHMARK_MODE = True` в конфиге).")
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
            c1.metric("Забегов",         total)
            c2.metric("Побед",           wins)
            c3.metric("Винрейт",         f"{wins / total * 100:.1f}%")
            c4.metric("Средний этаж",    f"{df['floor'].mean():.1f}")
            c5.metric("Сборка колоды",   f"{df['deck_completion'].mean() * 100:.1f}%")

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
    importlib.reload(_cfg)

    with open(_CONFIG_PATH, encoding="utf-8") as f:
        config_text = f.read()

    # ── Режим ────────────────────────────────────────────────────────────────
    st.subheader("Режим запуска")
    benchmark_mode = st.toggle("Бенчмарк (BENCHMARK_MODE)", value=bool(_cfg.BENCHMARK_MODE))

    st.divider()

    # ── PATH-коэффициенты ────────────────────────────────────────────────────
    st.subheader("Коэффициенты карты маршрута")

    def _num(label: str, attr: str, lo=0.0, hi=5.0, step=0.1):
        return st.number_input(label, min_value=lo, max_value=hi,
                               value=float(getattr(_cfg, attr)),
                               step=step, key=attr)

    col1, col2 = st.columns(2)
    with col1:
        pf   = _num("Обычный бой",         "PATH_FIGHT_REWARD")
        pe   = _num("Элита (без реликта)",  "PATH_ELITE_REWARD")
        pr   = _num("Реликт (элита/сундук)","PATH_RELIC_REWARD")
        pcl  = _num("Штраф Cursed Key",     "PATH_CURSE_LOSS")
        pu   = _num("Костёр (апгрейд)",     "PATH_UPGRADE_REWARD")
    with col2:
        pea1 = _num("Событие (акт 1)",      "PATH_EVENT_REWARD_A1")
        pea2 = _num("Событие (акты 2–3)",   "PATH_EVENT_REWARD_A2")
        pgs  = _num("Делитель золота магазин", "PATH_GOLD_SHOP_DIV", 10.0, 500.0, 10.0)
        pge  = _num("Делитель золота остаток", "PATH_GOLD_END_DIV",  10.0, 500.0, 10.0)
        psk  = _num("Штраф выживаемости K", "PATH_SURVIVABILITY_K", 0.0, 50.0, 1.0)

    st.divider()

    # ── HP-пороги ────────────────────────────────────────────────────────────
    st.subheader("HP-пороги лечения")
    col3, col4 = st.columns(2)
    with col3:
        hp_t  = st.slider("Обычный порог (HP_HEAL_THRESHOLD)",
                          0.0, 1.0, float(_cfg.HP_HEAL_THRESHOLD), 0.05)
    with col4:
        hp_tb = st.slider("Перед боссом (HP_HEAL_THRESHOLD_BOSS)",
                          0.0, 1.0, float(_cfg.HP_HEAL_THRESHOLD_BOSS), 0.05)

    st.divider()

    # ── TARGET_DECK ──────────────────────────────────────────────────────────
    st.subheader("Целевая колода (TARGET_DECK)")
    deck_df = pd.DataFrame(
        [{"Карта": k, "Макс. копий": v} for k, v in _cfg.TARGET_DECK.items()]
    )
    edited_deck = st.data_editor(
        deck_df, num_rows="dynamic", use_container_width=True,
        hide_index=True, key="deck_editor",
    )

    st.divider()

    # ── Сохранить ────────────────────────────────────────────────────────────
    if st.button("Сохранить изменения", type="primary"):
        new_text = config_text

        # BENCHMARK_MODE
        new_text = re.sub(
            r"^(BENCHMARK_MODE\s*=\s*)\w+",
            lambda m: f"{m.group(1)}{benchmark_mode}",
            new_text, flags=re.MULTILINE,
        )

        # Числовые параметры
        scalars = {
            "PATH_FIGHT_REWARD":    pf,  "PATH_ELITE_REWARD":    pe,
            "PATH_RELIC_REWARD":    pr,  "PATH_CURSE_LOSS":      pcl,
            "PATH_UPGRADE_REWARD":  pu,  "PATH_EVENT_REWARD_A1": pea1,
            "PATH_EVENT_REWARD_A2": pea2,"PATH_GOLD_SHOP_DIV":   pgs,
            "PATH_GOLD_END_DIV":    pge, "PATH_SURVIVABILITY_K": psk,
            "HP_HEAL_THRESHOLD":    hp_t,"HP_HEAL_THRESHOLD_BOSS": hp_tb,
        }
        for name, val in scalars.items():
            new_text = re.sub(
                rf"^({re.escape(name)}\s*=\s*)[\d.]+",
                lambda m, v=val: f"{m.group(1)}{v}",
                new_text, flags=re.MULTILINE,
            )

        # TARGET_DECK — заменяем весь блок
        new_deck_lines = ["TARGET_DECK: dict[str, int] = {\n"]
        for _, row in edited_deck.iterrows():
            card  = str(row["Карта"]).strip().lower()
            copies = int(row["Макс. копий"]) if pd.notna(row["Макс. копий"]) else 1
            new_deck_lines.append(f"    {repr(card)}: {copies},\n")
        new_deck_lines.append("}\n")
        new_deck_str = "".join(new_deck_lines)

        new_text = re.sub(
            r"TARGET_DECK: dict\[str, int\] = \{[^}]*\}",
            new_deck_str.rstrip("\n"),
            new_text, flags=re.DOTALL,
        )

        with open(_CONFIG_PATH, "w", encoding="utf-8") as f:
            f.write(new_text)
        st.success("config.py сохранён. Перезапусти main.py чтобы применить.")
        st.rerun()

    # ── Raw-редактор ─────────────────────────────────────────────────────────
    with st.expander("Редактировать config.py напрямую"):
        raw = st.text_area("config.py", value=config_text, height=400, key="raw_config")
        if st.button("Сохранить raw"):
            with open(_CONFIG_PATH, "w", encoding="utf-8") as f:
                f.write(raw)
            st.success("Сохранено.")
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
        tail = "".join(lines[-n_lines:])
        st.text_area("", value=tail, height=500, key="log_content")
