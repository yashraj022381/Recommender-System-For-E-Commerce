"""Grade the sieve: did it keep the items the user later wanted?"""
import numpy as np
import pandas as pd



def candidate_report(masks: dict, ev, users, cold_items=None) -> pd.DataFrame:
    truth = ev.truth[users].toarray() > 0
    n_rel = truth.sum(1)
    ok = n_rel > 0
    rows = {}
    for name, m in masks.items():
        hit = (m & truth).sum(1)
        row = {"recall": float((hit[ok] / n_rel[ok]).mean()), "avg_size": float(m.sum(1).mean())}
        if cold_items is not None:
               t = truth & cold_items[None, :]
               row["cold_item_recall"] = float((m & t).sum() / max(t.sum(), 1))
        rows[name] = row
    return pd.DataFrame(rows).T
