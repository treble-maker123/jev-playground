import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import sys
    import time
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import numpy as np
    import pandas as pd

    NOTEBOOK_DIR = mo.notebook_dir() or Path.cwd()
    DATA_DIR = NOTEBOOK_DIR / "data"

    # src/ isn't a package, so put it on the path to import jev.py from it.
    sys.path.append(str(NOTEBOOK_DIR / "src"))
    from jev import ask, choice

    return DATA_DIR, alt, ask, choice, mo, np, pd, time


@app.cell
def _(DATA_DIR, pd):
    aita = pd.read_csv(DATA_DIR / "aita" / "AITA_labeled_posts.csv")
    aita
    return (aita,)


@app.cell
def _(aita):
    # what are the labels
    aita["verdict"].unique()
    return


@app.cell
def _(aita, alt):
    filtered = aita[aita["verdict"].isin(["NTA", "YTA", "ESH", "NAH"])]

    # included

    # NTA - not the asshole, i.e. the poster acted reasonably
    # YTA - yes the asshole, i.e. the poster acted as an asshole
    # ESH - everyone sucks here, i.e. everyone's an asshole here

    # excluded
    # NAH - no asshole here, confusing with NTA
    # INFO - not enough info
    # YWBTA - you would be the asshole - hypothetical
    # YWNBTA - you would not be the asshole - hypothetical

    alt.Chart(filtered["verdict"].value_counts().reset_index()).mark_bar().encode(
        x="count:Q",
        y=alt.Y("verdict:N", sort="-x"),
    )
    return (filtered,)


@app.cell
def _():
    # let's set some guidelines

    # The four labels are a 2x2 over "who behaved badly": the poster, the
    # other party, both, or neither.

    guidelines = {
        "NTA": (
            "Not the asshole. The poster acted reasonably and someone else in "
            "the story is at fault. Minor imperfection on the poster's part "
            "doesn't move the verdict, they just have to not be the one who "
            "crossed the line."
        ),
        "YTA": (
            "You're the asshole. The poster crossed the line, by the act "
            "itself or by how they handled it. They control the framing and "
            "still come out looking bad, so a messy reaction from the other "
            "party doesn't save them."
        ),
        "ESH": (
            "Everyone sucks here. Both sides behaved badly, though not "
            "necessarily equally. The usual shape: a real grievance answered "
            "with a response way out of proportion, leaving nobody clean."
        ),
        "NAH": (
            "No assholes here. Nobody is at fault: incompatible needs, bad "
            "circumstances, or a reasonable disagreement. Someone being hurt "
            "is not proof that anyone wronged them, and there's no villain "
            "to hand out here."
        ),
    }
    guidelines
    return (guidelines,)


@app.cell
def _(DATA_DIR, filtered, pd):
    # A stratified draw: we fix the per-class counts rather than let the
    # natural 78/18/3/2 split decide them. Recall is computed within a class,
    # so the rare verdicts need bodies on the ground. ESH and NAH are taken
    # whole - 75 and 48 posts are all that exist - which pins their recall
    # error bars at roughly +/-11 and +/-14 points however much we spend.
    ALLOCATION = {"NTA": 150, "YTA": 150, "ESH": 75, "NAH": 48}
    LABELS = list(ALLOCATION)
    SEED = 0

    available = filtered["verdict"].value_counts()
    sample = (
        pd.concat(
            [
                filtered[filtered["verdict"] == label].sample(
                    min(n, int(available[label])), random_state=SEED
                )
                for label, n in ALLOCATION.items()
            ]
        )
        .reset_index(drop=True)
        .sample(frac=1, random_state=SEED)  # interleave, so a partial run still spans classes
        .reset_index(drop=True)
    )

    # Inference is keyed by sample size, so re-running the notebook reads the
    # file instead of paying for the calls again.
    RUNS_DIR = DATA_DIR / "runs"
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    CACHE = RUNS_DIR / f"verdicts_n{len(sample)}.csv"

    # Natural prevalence, measured on everything we have rather than on the
    # draw. This is what the reweighting below corrects back toward.
    PRIOR = (available[LABELS] / available[LABELS].sum()).to_dict()
    return ALLOCATION, CACHE, LABELS, PRIOR, SEED, sample


@app.cell
def _(
    ALLOCATION,
    CACHE,
    LABELS,
    ask,
    choice,
    guidelines,
    mo,
    pd,
    sample,
    time,
):
    # Skip the whole run if the keyed file is already on disk. A run that dies
    # partway leaves a .partial alongside it, and the next attempt resumes from
    # there instead of re-buying the verdicts it already has.
    def _judge(post):
        answer = ask(
            {"title": post.post_title, "post": post.post_content},
            {
                "verdict": choice(
                    "Which AITA verdict does this post deserve?", guidelines
                )
            },
        )["verdict"]
        return {
            "post_id": post.post_id,
            "post_title": post.post_title,
            "post_content": post.post_content,
            "reddit": post.verdict,
            "jev": answer["choice"],
            "confidence": answer["confidence"],
            **{f"p_{label}": answer["probabilities"].get(label) for label in LABELS},
        }

    _partial = CACHE.with_suffix(".partial")

    if CACHE.exists():
        judged = pd.read_csv(CACHE)
        _counts = judged["reddit"].value_counts().to_dict()
        _drift = {k: (v, _counts.get(k, 0)) for k, v in ALLOCATION.items() if _counts.get(k, 0) != v}
        cache_note = (
            f"Loaded {len(judged)} cached verdicts from `{CACHE.name}`"
            + (f" — ⚠️ per-class counts differ from ALLOCATION: {_drift}" if _drift else "")
            + ". Delete the file to re-run inference."
        )
    else:
        _rows = pd.read_csv(_partial).to_dict("records") if _partial.exists() else []
        _seen = {r["post_id"] for r in _rows}
        _todo = sample[~sample["post_id"].isin(_seen)]
        _started = time.perf_counter()

        with mo.status.progress_bar(
            total=len(_todo),
            title="Convening the jury",
            subtitle=f"{len(_seen)} already cached" if _seen else "warming up",
            completion_title="Verdicts in",
            show_rate=True,
            show_eta=True,
        ) as _bar:
            for _i, _post in enumerate(_todo.itertuples(), start=1):
                _row = _judge(_post)
                _rows.append(_row)
                # Append as we go; a crash costs us nothing but the call in flight.
                pd.DataFrame([_row]).to_csv(
                    _partial, mode="a", header=not _partial.exists(), index=False
                )
                _bar.update(
                    increment=1,
                    subtitle=f"{_i / (time.perf_counter() - _started):.1f} decisions/sec",
                )

        judged = pd.DataFrame(_rows)
        judged.to_csv(CACHE, index=False)
        _partial.unlink(missing_ok=True)
        cache_note = f"Ran {len(_todo)} calls and wrote `{CACHE.name}`."

    mo.md(cache_note)
    return (judged,)


@app.cell
def _(judged, mo):
    # Which post the viewer is showing. Reactive state, because three
    # different controls all need to move the same cursor. It lives in its
    # own cell so it survives the controls re-rendering.
    get_i, set_i = mo.state(0)
    n_posts = len(judged)
    return get_i, n_posts, set_i


@app.cell
def _(judged, mo, n_posts, set_i):
    def _step(delta):
        return lambda _: set_i(lambda i: (i + delta) % n_posts)

    prev_post = mo.ui.button(label="◀ prev", on_click=_step(-1))
    next_post = mo.ui.button(label="next ▶", on_click=_step(1))

    # A scrollable, searchable, sortable list of every verdict. "#" is the
    # row's position, which is what the cursor is in terms of.
    picker = mo.ui.table(
        judged[["post_title", "reddit", "jev", "confidence"]].reset_index(
            names="#"
        ),
        selection="single",
        initial_selection=[0],
        page_size=8,
        max_height=280,
        on_change=lambda rows: (
            set_i(int(rows["#"].iloc[0])) if rows is not None and len(rows) else None
        ),
    )

    mo.vstack(
        [
            mo.hstack(
                [prev_post, next_post],
                justify="start",
                gap=1,
                align="center",
            ),
            picker,
        ]
    )
    return


@app.cell
def _(get_i, judged, mo):
    VERDICT_COLORS = {
        "NTA": "#2a9d8f",
        "YTA": "#d7263d",
        "ESH": "#e76f51",
        "NAH": "#457b9d",
    }

    # Braces are CSS, not f-string interpolation, so keep this block unformatted.
    GAVEL_CSS = """
    <style>
    @keyframes slam {
      0%   { transform: scale(2.4) rotate(-12deg); opacity: 0 }
      60%  { transform: scale(0.92) rotate(2deg);  opacity: 1 }
      100% { transform: scale(1) rotate(0deg) }
    }
    @keyframes grow { from { width: 0 } }
    .jev-bar > div { animation: grow 0.6s ease-out both }
    </style>
    """

    def _bars(row):
        return "".join(
            f'''<div style="display:flex;align-items:center;gap:.6rem;margin:.2rem 0">
                 <code style="width:3rem;opacity:.65">{label}</code>
                 <div class="jev-bar" style="flex:1;height:.7rem;border-radius:99px;
                      background:rgba(128,128,128,.15);overflow:hidden">
                   <div style="width:{row[f"p_{label}"]:.1%};height:100%;
                        background:{color};border-radius:99px"></div>
                 </div>
                 <span style="width:3rem;text-align:right;opacity:.65;
                       font-variant-numeric:tabular-nums">{row[f"p_{label}"]:.0%}</span>
               </div>'''
            for label, color in VERDICT_COLORS.items()
        )

    _row = judged.iloc[get_i()]
    _agreed = _row["jev"] == _row["reddit"]

    mo.Html(
        GAVEL_CSS
        + f'''
        <div style="display:flex;gap:2rem;align-items:flex-start;flex-wrap:wrap">
          <div style="min-width:14rem">
            <div style="font:700 4rem/1 system-ui;color:{VERDICT_COLORS[_row["jev"]]};
                 animation:slam .45s cubic-bezier(.2,1.6,.4,1) both">{_row["jev"]}</div>
            <div style="opacity:.65;margin-top:.4rem">
              {_row["confidence"]:.0%} confident &middot;
              {"reddit agreed" if _agreed else f"reddit said {_row['reddit']}"}
            </div>
            <div style="margin-top:1rem;max-width:22rem">{_bars(_row)}</div>
          </div>
          <div style="flex:1;min-width:18rem">
            <div style="font:600 1.05rem/1.4 system-ui;margin-bottom:.5rem">
              {_row["post_title"]}</div>
            <div style="max-height:16rem;overflow:auto;opacity:.8;line-height:1.5;
                 white-space:pre-wrap">{_row["post_content"]}</div>
          </div>
        </div>'''
    )
    return


@app.cell
def _(mo):
    mo.md("""
    # Analysis

    Plain accuracy is unusable here: always answering NTA scores ~78% on
    the natural distribution, so any number has to be read against that
    floor. The split that organises everything below is which metrics
    depend on how common each class is:

    - **Recall** is computed inside a single class, so the class mix cannot
      touch it. The stratified draw measures it directly.
    - **Precision and accuracy** do depend on the mix. On a balanced draw
      they flatter the rare classes, so they are also reported
      *importance-reweighted* back to natural prevalence.
    """)
    return


@app.cell
def _(np, pd):
    Z95 = 1.959963984540054

    def wilson(successes, n, z=Z95):
        """CI for a single proportion. Used for per-class recall, where the
        textbook normal interval misbehaves: with 48 NAH posts a perfect score
        should read [0.93, 1.0], not the [1.0, 1.0] a bootstrap would give."""
        if n == 0:
            return (np.nan, np.nan)
        p = successes / n
        d = 1 + z**2 / n
        centre = (p + z**2 / (2 * n)) / d
        half = z / d * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
        return (max(0.0, centre - half), min(1.0, centre + half))

    def cm_metrics(cm):
        """Every scalar below is a function of the confusion matrix alone.

        `cm` is (..., K, K) with true class on axis -2 and predicted on axis
        -1, so a stack of bootstrap matrices runs through the same code as a
        single one. Entries are counts, or weights once reweighted.
        """
        tp = np.einsum("...ii->...i", cm)
        support = cm.sum(-1)  # per true class
        predicted = cm.sum(-2)  # per predicted class
        total = cm.sum((-1, -2))

        with np.errstate(invalid="ignore", divide="ignore"):
            recall = np.where(support > 0, tp / support, np.nan)
            precision = np.where(predicted > 0, tp / predicted, 0.0)
            denom = precision + recall
            f1 = np.where(denom > 0, 2 * precision * recall / denom, 0.0)
            accuracy = tp.sum(-1) / total
            # Chance agreement from the marginals; this is what makes kappa
            # readable under a 78% majority class.
            p_chance = (support * predicted).sum(-1) / total**2
            kappa = (accuracy - p_chance) / (1 - p_chance)

        return {
            "recall": recall,
            "precision": precision,
            "f1": f1,
            "support": support,
            "accuracy": accuracy,  # == micro-F1 for single-label multiclass
            "balanced_accuracy": np.nanmean(recall, axis=-1),
            "macro_f1": f1.mean(-1),
            "weighted_f1": (f1 * support).sum(-1) / total,
            "kappa": kappa,
        }

    def average_precision(scores, positive, weights):
        """Threshold-free ranking quality for one class, weighted so the answer
        is the one you would see at natural prevalence. Preferred over ROC-AUC
        for ESH and NAH, where ROC looks good at 2% prevalence almost by
        construction."""
        order = np.argsort(-np.asarray(scores, dtype=float))
        hit = np.asarray(positive, dtype=float)[order]
        w = np.asarray(weights, dtype=float)[order]
        cum_pos = np.cumsum(hit * w)
        cum_all = np.cumsum(w)
        precision_at_k = cum_pos / cum_all
        total_pos = (hit * w).sum()
        return (precision_at_k * hit * w).sum() / total_pos if total_pos else np.nan

    def confusion(frame, labels):
        """True class on the rows, jev's pick on the columns."""
        return (
            pd.crosstab(frame["reddit"], frame["jev"])
            .reindex(index=labels, columns=labels, fill_value=0)
            .to_numpy(dtype=float)
        )

    def bootstrap_cms(cm, b=10_000, seed=0):
        """Stratified bootstrap, in closed form.

        Within a true class, every metric here sees only the tally of predicted
        labels -- so resampling that class's posts with replacement *is* a
        multinomial draw over its confusion row, and `b` replicates cost one
        vectorised call each. Resampling across the whole draw instead would let
        the per-class counts wander off the allocation we deliberately chose,
        inflating every interval.
        """
        rng = np.random.default_rng(seed)
        support = cm.sum(1)
        rows = np.where(
            support[:, None] > 0, cm / np.where(support[:, None] > 0, support[:, None], 1), 1 / cm.shape[1]
        )
        return np.stack(
            [rng.multinomial(int(support[i]), rows[i], size=b) for i in range(len(cm))],
            axis=1,
        )

    def stratified_subset(frame, fraction, seed=0):
        """A stratified subset of already-judged rows, keeping the class mix.

        Subsets are nested: each class is shuffled once under `seed` and we take
        a prefix, so the n=100 subset is contained in the n=200 one. That is
        what makes the learning curve below readable rather than jumpy.
        """
        if fraction >= 1.0:
            return frame.reset_index(drop=True)
        parts = [
            group.sample(frac=1, random_state=seed).iloc[
                : max(1, round(len(group) * fraction))
            ]
            for _, group in frame.groupby("reddit", sort=False)
        ]
        return pd.concat(parts).reset_index(drop=True)

    return (
        average_precision,
        bootstrap_cms,
        cm_metrics,
        confusion,
        stratified_subset,
        wilson,
    )


@app.cell
def _(mo):
    # Which of the judged rows the analysis runs on. This sits *after* inference
    # on purpose: the API calls are already paid for and cached, so subsetting
    # here is free and costs nothing to redo. Nothing above this cell is
    # affected by the control.
    subset = mo.ui.dropdown(
        options={"full draw": 1.0, "75%": 0.75, "50%": 0.5, "25%": 0.25, "10%": 0.1},
        value="full draw",
        label="analyse",
    )
    subset
    return (subset,)


@app.cell
def _(SEED, judged, mo, stratified_subset, subset):
    evaluated = stratified_subset(judged, subset.value, seed=SEED)
    mo.md(
        f"Metrics below are computed on **{len(evaluated)} of {len(judged)}** "
        f"judged posts: {dict(evaluated['reddit'].value_counts())}."
    )
    return (evaluated,)


@app.cell
def _(LABELS, PRIOR, confusion, evaluated, np, pd):
    # The observed confusion matrix, plus the same matrix reweighted to natural
    # prevalence. Each row is scaled by prior / share-in-draw, so an NTA post
    # counts for ~5 and an NAH post for ~0.1. Row scaling cancels inside a row,
    # which is exactly why recall survives the reweighting unchanged and
    # precision does not.
    cm = confusion(evaluated, LABELS)

    share = cm.sum(1) / cm.sum()
    weights = np.array([PRIOR[label] for label in LABELS]) / share
    cm_natural = cm * weights[:, None]

    # Per-row weight, aligned to `evaluated`, for the weighted PR-AUC below.
    row_weight = evaluated["reddit"].map(dict(zip(LABELS, weights))).to_numpy()

    cm_display = pd.DataFrame(cm.astype(int), index=LABELS, columns=LABELS)
    cm_display.index.name = "reddit"
    cm_display.columns.name = "jev"
    cm_display
    return cm, cm_natural, row_weight, weights


@app.cell
def _(LABELS, alt, cm, mo, pd):
    # Row-normalised, i.e. each cell is "of the posts reddit called X, the
    # share jev called Y". The diagonal is recall.
    _long = (
        pd.DataFrame(cm / cm.sum(1, keepdims=True), index=LABELS, columns=LABELS)
        .rename_axis("reddit")
        .reset_index()
        .melt("reddit", var_name="jev", value_name="share")
    )

    _heat = (
        alt.Chart(_long)
        .mark_rect()
        .encode(
            x=alt.X("jev:N", sort=LABELS, title="jev said"),
            y=alt.Y("reddit:N", sort=LABELS, title="reddit said"),
            color=alt.Color(
                "share:Q", scale=alt.Scale(scheme="blues"), legend=None
            ),
        )
    )
    _text = (
        alt.Chart(_long)
        .mark_text(fontSize=13)
        .encode(
            x=alt.X("jev:N", sort=LABELS),
            y=alt.Y("reddit:N", sort=LABELS),
            text=alt.Text("share:Q", format=".0%"),
            color=alt.condition(
                alt.datum.share > 0.5, alt.value("white"), alt.value("#333")
            ),
        )
    )

    mo.vstack(
        [
            mo.md("### Where the disagreement lives"),
            (_heat + _text).properties(width=320, height=260),
        ]
    )
    return


@app.cell
def _(LABELS, cm, cm_metrics, cm_natural, np, pd, wilson):
    # Per class. Recall is identical in both framings, so it is reported once;
    # precision and F1 are shown as measured on the draw and as reweighted,
    # because the rare-class numbers move a lot between the two.
    _m, _n = cm_metrics(cm), cm_metrics(cm_natural)
    _tp = np.einsum("ii->i", cm)

    per_class = pd.DataFrame(
        {
            "support": _m["support"].astype(int),
            "recall": _m["recall"],
            "recall 95% CI": [
                "[{:.2f}, {:.2f}]".format(*wilson(k, int(n)))
                for k, n in zip(_tp, _m["support"])
            ],
            "precision (draw)": _m["precision"],
            "precision (natural)": _n["precision"],
            "f1 (draw)": _m["f1"],
            "f1 (natural)": _n["f1"],
        },
        index=LABELS,
    ).round(3)
    per_class
    return


@app.cell
def _(SEED, bootstrap_cms, cm, cm_metrics, cm_natural, np, pd, weights):
    # Interval estimates come from the stratified bootstrap in `bootstrap_cms`.
    # Recall gets Wilson above instead: it is a single proportion, and with 48
    # NAH posts a perfect score has to read [0.93, 1.0], where a bootstrap can
    # only ever say [1.0, 1.0].
    B = 10_000
    cm_boot = bootstrap_cms(cm, B, SEED)
    cm_boot_natural = cm_boot * weights[:, None]

    def _ci(values):
        return "[{:.3f}, {:.3f}]".format(*np.percentile(values, [2.5, 97.5]))

    _keys = ["balanced_accuracy", "macro_f1", "weighted_f1", "accuracy", "kappa"]
    _draw, _nat = cm_metrics(cm), cm_metrics(cm_natural)
    _draw_b, _nat_b = cm_metrics(cm_boot), cm_metrics(cm_boot_natural)

    summary = pd.DataFrame(
        {
            "on the draw": [round(float(_draw[k]), 3) for k in _keys],
            "95% CI": [_ci(_draw_b[k]) for k in _keys],
            "at natural prevalence": [round(float(_nat[k]), 3) for k in _keys],
            "95% CI (natural)": [_ci(_nat_b[k]) for k in _keys],
        },
        index=[
            "balanced accuracy (mean per-class recall)",
            "macro-F1 (rare classes count equally)",
            "weighted-F1 (by support)",
            "accuracy (== micro-F1)",
            "Cohen's kappa (chance-corrected)",
        ],
    )
    summary
    return


@app.cell
def _(
    LABELS,
    SEED,
    alt,
    bootstrap_cms,
    cm_metrics,
    confusion,
    judged,
    mo,
    np,
    pd,
    stratified_subset,
):
    # Would more API calls buy a tighter answer? This is the one legitimate use
    # of genuine subsetting (as opposed to the bootstrap's resampling): take
    # nested stratified subsets of the judged rows, and watch the interval
    # width. If it is still falling at the full draw, more calls would help; if
    # it has flattened, we are against the wall imposed by only 48 NAH posts
    # existing at all, and no amount of spending moves it.
    #
    # Fewer replicates than the summary above, since this runs one bootstrap per
    # point and the width is all we read off it.
    _rows = []
    for _frac in (0.1, 0.25, 0.5, 0.75, 1.0):
        _sub = stratified_subset(judged, _frac, seed=SEED)
        _cm = confusion(_sub, LABELS)
        _boot = cm_metrics(bootstrap_cms(_cm, 2_000, SEED))
        for _metric in ("balanced_accuracy", "macro_f1"):
            _lo, _hi = np.percentile(_boot[_metric], [2.5, 97.5])
            _rows.append(
                {
                    "n": len(_sub),
                    "metric": _metric.replace("_", " "),
                    "estimate": float(cm_metrics(_cm)[_metric]),
                    "lo": _lo,
                    "hi": _hi,
                    "CI width (pp)": (_hi - _lo) * 100,
                }
            )

    learning_curve = pd.DataFrame(_rows)

    _base = alt.Chart(learning_curve).encode(
        x=alt.X("n:Q", title="posts judged", scale=alt.Scale(zero=False))
    )
    _band = _base.mark_area(opacity=0.2).encode(
        y=alt.Y("lo:Q", title="metric"), y2="hi:Q", color="metric:N"
    )
    _line = _base.mark_line(point=True).encode(y="estimate:Q", color="metric:N")

    mo.vstack(
        [
            mo.md("### Is the evaluation itself big enough?"),
            (_band + _line).properties(width=420, height=240),
            learning_curve.pivot_table(
                index="n", columns="metric", values="CI width (pp)"
            ).round(1),
        ]
    )
    return


@app.cell
def _(LABELS, PRIOR, cm, cm_metrics, cm_natural, mo, np):
    # The floors. A number only means something above the better of these.
    _always_nta = max(PRIOR.values())
    _nat, _draw = cm_metrics(cm_natural), cm_metrics(cm)

    # Collapsing to "is the poster at fault" separates real misreadings from
    # losses on the NTA/NAH and YTA/ESH boundaries, which are blurry for humans
    # too. Weighted by the same natural-prevalence weights.
    _at_fault = np.array([label in ("YTA", "ESH") for label in LABELS])
    _binary = np.array(
        [
            [
                cm_natural[np.ix_(_at_fault == t, _at_fault == p)].sum()
                for p in (False, True)
            ]
            for t in (False, True)
        ]
    )
    _bin_m = cm_metrics(_binary)

    mo.md(
        f"""
        ### Baselines and the adjacent-class question

        | | value |
        |---|---|
        | always-NTA accuracy, natural prevalence | **{_always_nta:.1%}** |
        | always-NTA balanced accuracy | **25.0%** |
        | jev accuracy, natural prevalence | **{float(_nat["accuracy"]):.1%}** |
        | jev balanced accuracy | **{float(_draw["balanced_accuracy"]):.1%}** |
        | jev kappa | **{float(_nat["kappa"]):.3f}** |

        Collapsed to the binary *poster at fault* ({{YTA, ESH}}) vs *not*
        ({{NTA, NAH}}), at natural prevalence: accuracy
        **{float(_bin_m["accuracy"]):.1%}**, balanced accuracy
        **{float(_bin_m["balanced_accuracy"]):.1%}**, kappa
        **{float(_bin_m["kappa"]):.3f}**. A large gap between this and macro-F1
        above means the model reads the posts and is losing on a four-way
        boundary, not misunderstanding them.
        """
    )
    return


@app.cell
def _(LABELS, PRIOR, average_precision, evaluated, pd, row_weight):
    # Ranking quality, ignoring the argmax entirely. Separates "cannot spot ESH
    # at all" from "spots ESH but the decision rule never picks it", which is
    # the failure mode a 78% prior produces. Chance level is the class's own
    # prevalence, so the lift over that column is the whole signal.
    ranking = pd.DataFrame(
        [
            {
                "label": label,
                "PR-AUC (natural)": average_precision(
                    evaluated[f"p_{label}"], evaluated["reddit"] == label, row_weight
                ),
                "chance": PRIOR[label],
            }
            for label in LABELS
        ]
    ).set_index("label")
    ranking["lift"] = ranking["PR-AUC (natural)"] / ranking["chance"]
    ranking.round(3)
    return


@app.cell
def _(LABELS, PRIOR, cm_metrics, confusion, evaluated, mo, np, pd, weights):
    # Prior correction. Argmax over probabilities shaped by a 78/18/3/2 prior
    # will almost never emit ESH or NAH, so divide each probability by that
    # prior before taking the argmax. `alpha` scales how hard we push: 0 is
    # plain argmax, 1 divides by the full prior. Sweeping it shows the trade
    # rather than asserting one point on it -- rare-class recall is bought with
    # majority-class accuracy, and this is the exchange rate.
    #
    # Reading a best alpha off this table is fitting one parameter to the test
    # set. For a number you would quote, pick alpha on a held-out split.
    _p = evaluated[[f"p_{label}" for label in LABELS]].to_numpy(dtype=float)
    _prior = np.array([PRIOR[label] for label in LABELS])

    def _rule(alpha):
        _picked = (_p / _prior**alpha).argmax(1)
        _cm = confusion(
            evaluated.assign(jev=[LABELS[i] for i in _picked]), LABELS
        )
        _draw, _nat = cm_metrics(_cm), cm_metrics(_cm * weights[:, None])
        return {
            "balanced accuracy": float(_draw["balanced_accuracy"]),
            "macro-F1 (natural)": float(_nat["macro_f1"]),
            "accuracy (natural)": float(_nat["accuracy"]),
            "ESH recall": float(_draw["recall"][LABELS.index("ESH")]),
            "NAH recall": float(_draw["recall"][LABELS.index("NAH")]),
        }

    prior_correction = pd.DataFrame(
        {f"alpha={a:g}": _rule(a) for a in (0, 0.25, 0.5, 0.75, 1.0)}
    ).round(3)

    mo.vstack(
        [
            mo.md("### Does the decision rule or the model lose the rare classes?"),
            prior_correction,
        ]
    )
    return


@app.cell
def _(evaluated, mo):
    # Reddit verdicts are crowd votes on self-reported stories, not ground
    # truth. Some share of these rows are jev being right, and the only way to
    # know is to read them. Feed this to the viewer above by pointing it here.
    disagreements = evaluated[
        evaluated["jev"] != evaluated["reddit"]
    ].reset_index(drop=True)
    mo.vstack(
        [
            mo.md(
                f"### {len(disagreements)} disagreements "
                f"({len(disagreements) / len(evaluated):.0%} of the analysed set)"
            ),
            disagreements[["post_title", "reddit", "jev", "confidence"]],
        ]
    )
    return


if __name__ == "__main__":
    app.run()
