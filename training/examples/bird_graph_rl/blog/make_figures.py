"""Figures and the numbers behind them for the Fireworks post. Reads run folders only.

Every value drawn is also written to ``figures/numbers.json`` so a sentence in the post can be
checked against a file.

    python bird_graph_rl/blog/make_figures.py
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from fw_e3_check import ARMS  # noqa: E402
from milestone1_check import RUNS, boot, load  # noqa: E402
from run2_check import category  # noqa: E402

OUT = HERE / "figures"
RUN = RUNS / "fw_rl_27b_run1b"
LABEL = {"base": "Base model", "control": "Untrained adapter\n(same serving route)", "trained": "After RL\n(37 updates)"}
COLOR = {"base": "#8a8f98", "control": "#b9a26a", "trained": "#2f6fde"}
K = 4


def style(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    P = {a: [load(n) for n in names] for a, names in ARMS.items()}
    q = sorted(set.intersection(*(set(p) for ps in P.values() for p in ps)))
    mean = {a: {key: {i: sum(float(p[i][key]) for p in ps) / K for i in q} for key in ("correct", "lenient")} for a, ps in P.items()}
    numbers: dict = {"n_questions": len(q), "samples_per_question": K}

    # ---- figure 1: accuracy by arm, with the interval of each arm's mean over questions
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    numbers["accuracy"] = {}
    for ax, (key, title) in zip(axes, (("correct", "Strict: exactly the reference rows"), ("lenient", "Lenient: right row count, all values present"))):
        for x, a in enumerate(ARMS):
            pt, lo, hi = boot(list(mean[a][key].values()))
            numbers["accuracy"][f"{key}/{a}"] = {"mean": pt, "ci95": [lo, hi]}
            ax.bar(x, pt, color=COLOR[a], width=0.62)
            ax.errorbar(x, pt, yerr=[[pt - lo], [hi - pt]], color="#222", capsize=4, lw=1.2)
            ax.text(x, hi + 0.012, f"{pt:.3f}", ha="center", fontsize=10)
        ax.set_xticks(range(3), [LABEL[a] for a in ARMS], fontsize=9)
        ax.set_title(title, fontsize=10.5)
        ax.set_ylim(0.45, 0.78)
        style(ax)
    axes[0].set_ylabel("Share of answers correct (186 questions x 4 samples)")
    fig.suptitle("Qwen 3.8 27B on 186 held-out human questions, before and after RL on generated questions", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_accuracy.png", dpi=150)
    plt.close(fig)

    # ---- figure 2: every pass
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    numbers["strict_per_pass"] = {}
    for x, a in enumerate(ARMS):
        ys = [sum(p[i]["correct"] for i in q) / len(q) for p in P[a]]
        numbers["strict_per_pass"][a] = [round(y, 4) for y in ys]
        ax.scatter([x + d for d in (-0.12, -0.04, 0.04, 0.12)], ys, s=70, color=COLOR[a], edgecolor="#222", lw=0.6, zorder=3)
        ax.hlines(sum(ys) / len(ys), x - 0.25, x + 0.25, color="#222", lw=1.4)
    ax.set_xticks(range(3), [LABEL[a] for a in ARMS], fontsize=9)
    ax.set_ylabel("Strict accuracy of one full pass (186 questions)")
    ax.set_title("Each dot is one complete pass over the benchmark; the line is the arm's mean", fontsize=10.5)
    style(ax)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_passes.png", dpi=150)
    plt.close(fig)

    # ---- figure 3: what happened to the 744 samples
    order = [("strict", "Exactly right", "#2f6fde"), ("lenient_only", "Right values, wrong shape", "#8fb4f2"),
             ("wrong", "Wrong", "#d9d9d9"), ("wrong_turn_cap", "Ran out of turns", "#a6a6a6"), ("no_query", "No query", "#555")]
    cats = {a: Counter(category(p[i]) for p in ps for i in q) for a, ps in P.items()}
    numbers["sample_outcomes"] = {a: dict(c) for a, c in cats.items()}
    fig, ax = plt.subplots(figsize=(8.4, 3.6))
    for y, a in enumerate(reversed(list(ARMS))):
        left = 0
        for key, name, color in order:
            n = cats[a].get(key, 0)
            if not n:
                continue
            ax.barh(y, n, left=left, color=color, edgecolor="white", label=name if y == 0 else None)
            if n >= 18:
                ax.text(left + n / 2, y, str(n), ha="center", va="center", fontsize=9, color="white" if key == "strict" else "#222")
            left += n
    ax.set_yticks(range(3), [LABEL[a].replace("\n", " ") for a in reversed(list(ARMS))], fontsize=9)
    ax.set_xlabel("Samples (186 questions x 4)")
    ax.set_xlim(0, 744)
    ax.legend(ncol=4, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, 1.2), frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_outcomes.png", dpi=150)
    plt.close(fig)

    # ---- figure 4: where the gain sits. Untrained = base and control pooled (8 samples).
    untr = {i: (sum(p[i]["correct"] for p in P["base"]) + sum(p[i]["correct"] for p in P["control"])) / 8 for i in q}
    tr = mean["trained"]["correct"]
    groups = {"Never solved untrained\n(0 of 8)": [i for i in q if untr[i] == 0],
              "Sometimes solved\n(1 to 7 of 8)": [i for i in q if 0 < untr[i] < 1],
              "Always solved untrained\n(8 of 8)": [i for i in q if untr[i] == 1]}
    numbers["by_untrained_group"] = {}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [1.15, 1]})
    for x, (name, ids) in enumerate(groups.items()):
        u, t = sum(untr[i] for i in ids) / len(ids), sum(tr[i] for i in ids) / len(ids)
        numbers["by_untrained_group"][name.replace("\n", " ")] = {"n": len(ids), "untrained": round(u, 4), "trained": round(t, 4),
                                                                 "questions_trained_above_zero": sum(tr[i] > 0 for i in ids)}
        axes[0].bar(x - 0.19, u, 0.36, color="#a9a18c", label="Untrained (8 samples)" if x == 0 else None)
        axes[0].bar(x + 0.19, t, 0.36, color=COLOR["trained"], label="After RL (4 samples)" if x == 0 else None)
        axes[0].text(x - 0.19, u + 0.02, f"{u:.2f}", ha="center", fontsize=9)
        axes[0].text(x + 0.19, t + 0.02, f"{t:.2f}", ha="center", fontsize=9)
    axes[0].set_xticks(range(3), [f"{name}\nn = {len(ids)}" for name, ids in groups.items()], fontsize=9)
    axes[0].set_ylim(0, 1.12)
    axes[0].set_ylabel("Strict accuracy")
    axes[0].set_title("Grouped by how the untrained model did\n(extremes regress to the mean by construction)", fontsize=10)
    axes[0].legend(fontsize=8.5, frameon=False, loc="upper left")
    style(axes[0])
    change = Counter(round(tr[i] - untr[i], 3) for i in q)
    numbers["per_question_change_histogram"] = {str(k): v for k, v in sorted(change.items())}
    xs = sorted(change)
    axes[1].bar(xs, [change[x] for x in xs], width=0.1, color=[COLOR["trained"] if x > 0 else "#c0504d" if x < 0 else "#bbb" for x in xs])
    axes[1].set_xlabel("Change in a question's success rate\n(after RL minus untrained)")
    axes[1].set_ylabel("Questions")
    axes[1].set_title("Many small shifts, few flips", fontsize=10)
    style(axes[1])
    fig.tight_layout()
    fig.savefig(OUT / "fig4_where.png", dpi=150)
    plt.close(fig)

    # ---- figure 5: the training run
    rows = list(csv.DictReader((RUN / "curves.csv").open()))
    x = [int(r["update"]) for r in rows]
    col = lambda k: [float(r[k]) for r in rows]  # noqa: E731

    def rolling(v: list[float], w: int = 5) -> list[float]:
        return [sum(v[max(0, i - w + 1): i + 1]) / len(v[max(0, i - w + 1): i + 1]) for i in range(len(v))]

    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    panels = [("reward_all_rollouts", "Mean reward, all 64 rollouts of the update", True),
              ("share_groups_kept", "Share of the 8 questions that gave a training signal", True),
              ("loss_sum", "Loss (importance-sampling surrogate, summed)", False),
              ("sampling_entropy", "Sampling entropy", True),
              ("seconds_total", "Seconds per update", False),
              ("meter_usd_upper_cumulative", "Cumulative cost, USD (meter, upper bound)", False)]
    for ax, (key, title, smooth) in zip(axes.flat, panels):
        y = col(key)
        ax.plot(x, y, color="#9bb7e8" if smooth else COLOR["trained"], marker="o", ms=3, lw=1, label="per update")
        if smooth:
            ax.plot(x, rolling(y), color=COLOR["trained"], lw=2, label="mean of last 5")
            ax.legend(fontsize=8, frameon=False)
        if key == "loss_sum":
            ax.axhline(0, color="#444", lw=0.8)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("update")
        style(ax)
    fig.suptitle("The training run: 37 updates of 8 questions x 8 rollouts, Qwen 3.8 27B on Fireworks serverless", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "fig5_training.png", dpi=150)
    plt.close(fig)
    numbers["training"] = {"updates": len(rows), "rollouts": sum(int(r["rollouts"]) for r in rows), "groups": sum(int(r["groups"]) for r in rows),
                           "groups_kept": round(sum(float(r["share_groups_kept"]) * int(r["groups"]) for r in rows)),
                           "groups_all_correct": round(sum(float(r["share_groups_all_correct"]) * int(r["groups"]) for r in rows)),
                           "groups_all_wrong": round(sum(float(r["share_groups_all_wrong"]) * int(r["groups"]) for r in rows)),
                           "trained_tokens": sum(int(r["trained_tokens"]) for r in rows), "seconds": round(sum(col("seconds_total"))),
                           "meter_usd_upper": float(rows[-1]["meter_usd_upper_cumulative"]),
                           "reward_first9": round(sum(col("reward_all_rollouts")[:9]) / 9, 4), "reward_last9": round(sum(col("reward_all_rollouts")[-9:]) / 9, 4)}

    # ---- figure 6: by difficulty tier
    tiers: dict[str, list[int]] = {}
    for i in q:
        tiers.setdefault(P["base"][0][i]["difficulty"], []).append(i)
    numbers["by_tier"] = {}
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    for x0, t in enumerate(("simple", "moderate", "challenging")):
        ids = tiers[t]
        numbers["by_tier"][t] = {"n": len(ids)}
        for d, a in zip((-0.25, 0, 0.25), ARMS):
            v = sum(mean[a]["correct"][i] for i in ids) / len(ids)
            numbers["by_tier"][t][a] = round(v, 4)
            ax.bar(x0 + d, v, 0.23, color=COLOR[a], label=LABEL[a].replace("\n", " ") if x0 == 0 else None)
            ax.text(x0 + d, v + 0.012, f"{v:.2f}", ha="center", fontsize=8.5)
    ax.set_xticks(range(3), [f"{t}\n(n = {len(tiers[t])})" for t in ("simple", "moderate", "challenging")])
    ax.set_ylabel("Strict accuracy")
    ax.set_title("By the benchmark's own difficulty labels", fontsize=10.5)
    ax.legend(fontsize=8.5, frameon=False)
    style(ax)
    fig.tight_layout()
    fig.savefig(OUT / "fig6_tiers.png", dpi=150)
    plt.close(fig)

    (OUT / "numbers.json").write_text(json.dumps(numbers, indent=1))
    print(json.dumps({k: numbers[k] for k in ("accuracy", "by_untrained_group", "by_tier", "training")}, indent=1))


if __name__ == "__main__":
    main()
