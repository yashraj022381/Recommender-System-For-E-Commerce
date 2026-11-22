"""Run:  streamlit run app/streamlit_app.py   (after build_artifacts.py and evaluate_all.py)"""
import os
import time

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from recsys.config import PATHS

from recsys.evaluation.abtest import exposure, funnel, simulate, top_lists
from recsys.evaluation.evaluator import Evaluator
from recsys.serving.service import BUNDLE, MODEL_ORDER, RecService
from recsys.llm.assistant import Assistant, ShopTools
from recsys.llm.client import MODEL, make_client
from recsys.llm.explain import explain_in_words


st.set_page_config(page_title="E-commerce RecSys", page_icon="🛍️", layout="wide")

if not BUNDLE.exists():
    st.error("No model bundle found. Run `python scripts/build_artifacts.py` first.")
    st.stop()


@st.cache_resource(show_spinner="Loading models (once)...")
def get_service():
    return RecService()


@st.cache_resource
def get_eval():
    return Evaluator(svc.data, split="test", ks=(10,))


@st.cache_data(show_spinner="Building top-10 lists for every test user...")
def get_exposure(model_name):
    ev = get_eval()
    return exposure(top_lists(svc.models[model_name], ev), ev, svc.price, svc.oos)


svc = get_service()
INFLATED = {"TwoStage (rank)", "Rule: fresh x favorite", "Final: Generative + shelf rules"}

@st.cache_resource
def get_llm():
    return make_client()            # None without ANTHROPIC_API_KEY


LLM = get_llm()


@st.cache_data(show_spinner="Asking the model...", max_entries=500)
def nl_explain(model, user, item):
    lines = svc.explain(model, user, item)["lines"]
    return explain_in_words(LLM, svc.titles[item], lines)

def item_label(i):
    return f"{i} · {svc.titles[i] or 'Item ' + str(i)}"


def badges(c):
    out = ""
    if c["new"]:
        out += " 🆕"
    if c["oos"]:
        out += " ⛔ out of stock (simulated)"
    return out


def show_cards(cards, model=None, user_id=None, bought=(), key=""):
    for c in cards:
        with st.container(border=True):
            hit = "  ✅ bought later" if c["item_id"] in bought else ""
            st.markdown(f"**{c['rank']}. {c['title']}**{badges(c)}{hit}  \n"
                        f"{c['category']} · ${c['price']:.2f}")
            if user_id is not None and model is not None:
                with st.expander("Why this?"):
                    for line in svc.explain(model, user_id, c["item_id"])["lines"]:
                        st.write("• " + line)
                    if LLM is not None and st.button("Say it in words",
                                                     key=f"nl_{key}_{user_id}_{c['item_id']}"):
                        text, source = nl_explain(model, user_id, c["item_id"])
                        st.info(text)
                        st.caption(f"source: {source}")
                        st.caption("LLM features: " + (f"on ({MODEL})" if LLM else "off (set GROQ_API_KEY to enable)"))
                        

st.title("🛍️ E-commerce Recommender System")
st.caption("Synthetic shop data · 16 models · all clicks, carts and stock flags in this app are SIMULATED")


tab_rec, tab_cmp, tab_new, tab_live, tab_ab, tab_chat = st.tabs(
    ["Recommendations", "Compare models", "New shopper", "Live clicks", "Simulated A/B stream", "Assistant"])
# ------------------------------------------------------------------ 1. recommendations
with tab_rec:
    mode = st.radio("Start from", ["A shopper", "A product"], horizontal=True)
    if mode == "A shopper":
        c1, c2 = st.columns([1, 3])
        with c1:
            user = st.selectbox("Shopper ID", svc.demo_users, key="rec_user")
            prof = svc.user_profile(user)
            st.markdown("**Profile**")
            if prof["cold"]:
                st.warning("New shopper: no history, so every model falls back to popularity.")
            st.write(f"Liked items: {prof['n_liked']}")
            st.write("Favorite aisles: " + (", ".join(f"{n} ({s:.0%})" for n, s in prof["top_categories"]) or "none"))
            st.markdown("**Most recent likes**")
            for c in prof["recent"]:
                st.caption(f"{c['title']} · {c['category']}")
            reveal = st.checkbox("Mark items they really bought next (test period)", value=True)
        with c2:
            chosen = st.multiselect("Models to compare (up to 4)", MODEL_ORDER, max_selections=4,
                                    default=["UserKNN", "Generative", "Final: Generative + shelf rules"])
            k = st.slider("Items per list", 5, 20, 10)
            if chosen:
                cols = st.columns(len(chosen))
                bought = set(prof["future"]) if reveal else set()
                for col, name in zip(cols, chosen):
                    cards, ms = svc.recommend(name, user, k)
                    hits = sum(c["item_id"] in bought for c in cards)
                    with col:
                        st.subheader(name)
                        st.caption(f"{ms:.1f} ms" + (f" · {hits} hit(s)" if reveal else ""))
                        show_cards(cards, name, user, bought, key=name)
    else:
        item = st.selectbox("Product", range(svc.n_items), format_func=item_label)
        st.markdown(f"**{svc.titles[item]}** · {svc.cats_name[item]} · ${svc.price[item]:.2f}"
                    + (" 🆕 (no purchase history)" if svc.fresh[item] else ""))
        cols = st.columns(len(svc.SIMILAR_METHODS))
        for col, method in zip(cols, svc.SIMILAR_METHODS):
            with col:
                st.subheader(method)
                #show_cards(svc.similar_items(item, method, 8))
                cards = svc.similar_items(item, method, 8)
                if cards:
                    show_cards(cards)
                else:
                    st.info("No purchases yet, so there are no co-purchases for this item.")
        st.caption("Co-purchase = bought by the same shoppers. Description = similar words. "
                   "Two-Tower = learned codes. A new item has no co-purchases, so only the other two can help.")

# ------------------------------------------------------------------ 2. comparison
with tab_cmp:
    path = PATHS.artifacts / "leaderboard.csv"
    if not path.exists():
        st.info("Run `python scripts/evaluate_all.py` to create the leaderboard.")
    else:
        lb = pd.read_csv(path, index_col=0)
        lb["note"] = ["generator-inflated, see README" if m in INFLATED else "" for m in lb.index]
        metric = st.selectbox("Metric", [c for c in lb.columns if c != "note"],
                              index=list(lb.columns).index("ndcg@10"))
        s = lb[metric].sort_values()
        st.plotly_chart(px.bar(x=s.values, y=s.index, orientation="h", labels={"x": metric, "y": ""}),
                        use_container_width=True)
        fig = px.scatter(lb.reset_index(names="model"), x="coverage", y="ndcg@10", text="model",
                         hover_data=["novelty", "latency_ms"], title="Accuracy vs catalog coverage")
        fig.update_traces(textposition="top center")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(lb.style.format(precision=4), use_container_width=True)
        st.caption("One seed per model. Seed noise is about ±0.005 NDCG for the neural models, so gaps of "
                   "that size are ties. The three flagged models profit from a rule built into the synthetic "
                   "data generator (new items in the user's favorite aisle).")

# ------------------------------------------------------------------ 3. new shopper
with tab_new:
    st.write("A brand-new shopper has no history. Ask for a few liked products and match them to the rest.")
    picks = st.multiselect("Products they like (1-5)", range(svc.n_items), format_func=item_label,
                           max_selections=5, key="onboard")
    if picks:
        show_cards(svc.from_items(picks, 10))
        st.caption("Method: average of co-purchase and description similarity to the chosen products, "
                   "with a tiny popularity tie-breaker.")

# ------------------------------------------------------------------ 4. live clicks
with tab_live:
    c1, c2 = st.columns(2)
    luser = c1.selectbox("Shopper", svc.demo_users, key="live_user")
    lmodel = c2.selectbox("Model", MODEL_ORDER, index=MODEL_ORDER.index("SASRec"), key="live_model")
    key = (luser, lmodel)
    if st.session_state.get("live_key") != key:
        st.session_state.live_key, st.session_state.clicks = key, []
    base_ids = {c["item_id"] for c in svc.recommend(lmodel, luser, 10)[0]}
    cards, mode_txt, ms = svc.live_recommend(lmodel, luser, st.session_state.clicks, 10)
    st.info(f"{mode_txt}  ·  {ms:.1f} ms")
    st.write("Clicks so far: " + (", ".join(svc.titles[i] for i in st.session_state.clicks) or "none"))
    if st.button("Reset clicks"):
        st.session_state.clicks = []
        st.rerun()
    for c in cards:
        with st.container(border=True):
            a, b = st.columns([5, 1])
            fresh_in = " ⬆ new in list" if st.session_state.clicks and c["item_id"] not in base_ids else ""
            a.markdown(f"**{c['rank']}. {c['title']}**{badges(c)}{fresh_in}  \n{c['category']} · ${c['price']:.2f}")
            if b.button("Click", key=f"click_{c['item_id']}"):
                st.session_state.clicks.append(c["item_id"])
                st.rerun()

# ------------------------------------------------------------------ 5. simulated A/B stream
with tab_ab:
    st.warning("SIMULATION: visitors are replayed from test-period purchases and every click, cart and buy "
               "probability is an assumption. It shows how noisy a small test is, not real customer behavior.")
    c1, c2, c3, c4 = st.columns(4)
    ma = c1.selectbox("Arm A", MODEL_ORDER, index=MODEL_ORDER.index("UserKNN"))
    mb = c2.selectbox("Arm B", MODEL_ORDER, index=MODEL_ORDER.index("Generative"))
    n_vis = c3.slider("Visitors", 200, 3000, 1000, step=100)
    sess = c4.slider("Sessions per visitor", 1, 10, 5)
    if st.button("▶ Start stream"):
        ea, eb = get_exposure(ma), get_exposure(mb)
        rng = np.random.default_rng(int(time.time()) % 10_000)
        order = rng.choice(len(ea["rel"]), n_vis, replace=True)
        arm = rng.random(n_vis) < 0.5
        chart, text, bar = st.empty(), st.empty(), st.progress(0.0)
        dfs_a, dfs_b, rows = [], [], []
        for s in range(0, n_vis, 50):
            sel, a_mask = order[s:s + 50], arm[s:s + 50]
            if a_mask.any():
                dfs_a.append(simulate(ea, sel[a_mask], rng, sess))
            if (~a_mask).any():
                dfs_b.append(simulate(eb, sel[~a_mask], rng, sess))
            ra = pd.concat(dfs_a).revenue.mean() if dfs_a else 0.0
            rb = pd.concat(dfs_b).revenue.mean() if dfs_b else 0.0
            rows.append({f"A: {ma}": ra, f"B: {mb}": rb})
            chart.line_chart(pd.DataFrame(rows))
            text.caption(f"{min(s + 50, n_vis)} visitors · revenue per visitor  A {ra:.2f}  B {rb:.2f}")
            bar.progress(min((s + 50) / n_vis, 1.0))
            time.sleep(0.05)
        da, db = pd.concat(dfs_a), pd.concat(dfs_b)
        tab = pd.DataFrame({f"A: {ma}": funnel(da, sess), f"B: {mb}": funnel(db, sess)}).T
        tab["users"] = tab["users"].astype(int)
        st.dataframe(tab.style.format(precision=4), use_container_width=True)
        st.caption("Run it twice: the curves cross and the winner changes. That is what a test that is too small looks like.")

# ------------------------------------------------------------------ 6. chat assistant
with tab_chat:
    st.caption("The assistant can only recommend products returned by the store's own tools. "
               + ("Powered by an LLM." if LLM else "No API key: plain keyword search instead of an LLM."))
    who_raw = st.selectbox("Chat as shopper", [-1] + svc.demo_users, key="chat_user",
                           format_func=lambda u: "(new shopper, no history)" if u == -1 else str(u))
    who = None if who_raw == -1 else who_raw
    if st.session_state.get("chat_who", "unset") != who_raw:       # new shopper = fresh conversation
        st.session_state.chat_who, st.session_state.chat, st.session_state.turns = who_raw, [], 0
        st.session_state.assistant = Assistant(ShopTools(svc, who), LLM)
    for m in st.session_state.chat:
        with st.chat_message(m["role"]):
            st.markdown(m["text"])
            if m.get("cards"):
                show_cards(m["cards"])
    prompt = st.chat_input("e.g. a fantasy game under $20, or something like Rpg Game 1")
    if prompt:
        if st.session_state.turns >= 20:
            st.warning("Turn limit for this demo session. Pick another shopper to start over.")
        else:
            history = [{"role": m["role"], "content": m["text"]} for m in st.session_state.chat]
            with st.spinner("Thinking..."):
                res = st.session_state.assistant.ask(prompt, history)
            st.session_state.chat += [{"role": "user", "text": prompt},
                                      {"role": "assistant", "text": res["text"], "cards": res["cards"]}]
            st.session_state.turns += 1
            st.rerun()
