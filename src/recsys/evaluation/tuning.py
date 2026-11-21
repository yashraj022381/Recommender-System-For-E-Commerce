"""Grid search on the validation split."""
import time

from recsys.evaluation.evaluator import summary


def tune(cls, grid, data, ev_val, label=None, metric="ndcg@10"):
    """Try every config in `grid`; return (best_cfg, best_score, best_model)."""
    label = label or cls.__name__
    results = []
    for cfg in grid:
        t0 = time.time()
        model = cls(**cfg).fit(data)
        score = summary(ev_val.evaluate(model))[metric]
        print(f"  {label:<8} {str(cfg):<42} val {metric} = {score:.4f}   ({time.time() - t0:.1f}s)")
        results.append((score, cfg, model))

    best_score, best_cfg, best_model = max(results, key=lambda r: r[0])
    print(f"  -> best {label}: {best_cfg}  (tried {len(results)} configs)\n")
    return best_cfg, best_score, best_model
