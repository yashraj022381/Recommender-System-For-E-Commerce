"""SIMULATED online A/B test. All probabilities below are ASSUMPTIONS, not measurements."""
import numpy as np
import pandas as pd
from scipy import stats

SIM = dict(examine_top=0.8,                       # chance the top slot is looked at
           p_click_rel=0.35, p_click_irr=0.01,    # click given examined: loved vs other toy
           p_cart_rel=0.40, p_cart_irr=0.10,      # add-to-cart given click
           p_buy_rel=0.60, p_buy_irr=0.30)        # purchase given add-to-cart


def top_lists(model, ev, k: int = 10) -> np.ndarray:
    """The top-k list each test user would be shown (owned items removed)."""
    if hasattr(model, "set_context"):
        model.set_context(ev.context)
    out = []
    for s in range(0, len(ev.users), 1024):
        u = ev.users[s:s + 1024]
        sc = np.array(model.score(u), dtype=np.float32, copy=True)
        h = ev.history[u].tocoo()
        sc[h.row, h.col] = -np.inf
        out.append(ev._topk(sc)[:, :k])
    return np.vstack(out)



def exposure(lists, ev, price, oos) -> dict:
    """Per shown slot: is it loved (really bought later)? its price? is it out of stock?"""
    truth = ev.truth[ev.users].toarray() > 0
    return {"rel": np.take_along_axis(truth, lists, 1),
            "price": np.asarray(price)[lists],
            "oos": np.asarray(oos, dtype=bool)[lists]}


def simulate(exp, idx, rng, sessions: int = 10, p: dict = SIM) -> pd.DataFrame:
    """Simulate `sessions` visits for the users `idx`. Returns per-user funnel counts."""
    rel, price, oos = exp["rel"][idx], exp["price"][idx], exp["oos"][idx]
    n, k = rel.shape
    exam = p["examine_top"] / np.log2(np.arange(k) + 2.0)            # position bias
    shape = (n, sessions, k)
    pc = (np.where(rel, p["p_click_rel"], p["p_click_irr"]) * exam)[:, None, :]
    click = rng.random(shape) < pc
    cart = click & (rng.random(shape) < np.where(rel, p["p_cart_rel"], p["p_cart_irr"])[:, None, :])
    want = cart & (rng.random(shape) < np.where(rel, p["p_buy_rel"], p["p_buy_irr"])[:, None, :])

    o = oos[:, None, :]
    buy, lost = want & ~o, want & o                                  # sold-out toys can't be bought            
    pr = price[:, None, :]
    return pd.DataFrame({
        "clicks": click.sum((1, 2)), "carts": cart.sum((1, 2)), "purchases": buy.sum((1, 2)),
        "buy_sessions": buy.any(2).sum(1), "revenue": (buy * pr).sum((1, 2)),
        "lost": (lost * pr).sum((1, 2))})


def funnel(df: pd.DataFrame, sessions: int, k: int = 10) -> pd.Series:
    n = len(df)
    return pd.Series({
        "users": n,
        "CTR": df.clicks.sum() / (n * sessions * k),
        "add-to-cart rate": df.carts.sum() / max(df.clicks.sum(), 1),
        "conversion rate": df.buy_sessions.sum() / (n * sessions),
        "revenue/user": df.revenue.mean(),
        "lost to OOS/user": df.lost.mean()})

def revenue_lift(a, b, n_boot: int = 1000, seed: int = 0):
    """Relative revenue-per-user lift of B over A, with a bootstrap 95% interval."""
    rng = np.random.default_rng(seed)
    ra, rb = a["revenue"].to_numpy(), b["revenue"].to_numpy()
    ma = rng.choice(ra, (n_boot, len(ra))).mean(1)
    mb = rng.choice(rb, (n_boot, len(rb))).mean(1)
    lift = mb / np.maximum(ma, 1e-9) - 1
    point = rb.mean() / max(ra.mean(), 1e-9) - 1
    return float(point), [float(v) for v in np.percentile(lift, [2.5, 97.5])]

def repeat_ab(exp_a, exp_b, reps: int = 200, sessions: int = 10, seed: int = 0, alpha: float = 0.05):
    """Run the whole experiment `reps` times (new coin flips, new randomness).
    Returns (share significant in either direction, share significant with B better)."""
    rng = np.random.default_rng(seed)
    n = len(exp_a["rel"])
    any_sig = b_better = 0
    for _ in range(reps):
        arm = rng.random(n) < 0.5
        a = simulate(exp_a, np.flatnonzero(arm), rng, sessions)
        b = simulate(exp_b, np.flatnonzero(~arm), rng, sessions)
        t = stats.ttest_ind(b["revenue"], a["revenue"], equal_var=False)
        sig = bool(t.pvalue < alpha)
        any_sig += sig
        b_better += sig and (b["revenue"].mean() > a["revenue"].mean())
    return any_sig / reps, b_better /reps

