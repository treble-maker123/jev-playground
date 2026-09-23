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
def _(DATA_DIR, filtered):
    # Every post with a label we keep, no stratified draw: the class mix is the
    # natural one (roughly 78/18/3/2), so accuracy means what it says and
    # recall is read per class off the confusion matrix. ESH and NAH are small
    # whatever we do (75 and 48 posts exist), so their recall intervals stay
    # wide; that is a property of the data, not of the sample.
    LABELS = ["NTA", "YTA", "ESH", "NAH"]
    SEED = 0

    # Shuffled, so a partial run still spans classes.
    sample = filtered.sample(frac=1, random_state=SEED).reset_index(drop=True)

    # Inference is keyed by sample size, so re-running the notebook reads the
    # file instead of paying for the calls again.
    RUNS_DIR = DATA_DIR / "runs"
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    CACHE = RUNS_DIR / f"verdicts_n{len(sample)}.csv"
    return CACHE, LABELS, RUNS_DIR, sample


@app.cell
def _(CACHE, LABELS, RUNS_DIR, ask, choice, guidelines, mo, pd, sample, time):
    # Skip the whole run if the keyed file is already on disk. A run that dies
    # partway leaves a .partial alongside it, and the next attempt resumes from
    # there instead of re-buying the verdicts it already has. Verdicts from any
    # other run, finished or partial, are reused too: they are the same posts
    # under the same guidelines, so a post found in any of them is not re-judged.
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
        cache_note = (
            f"Loaded {len(judged)} cached verdicts from `{CACHE.name}`. "
            "Delete the file to re-run inference."
        )
    else:
        # Everything already paid for, from any run: finished files and
        # partials alike. Our own .partial is included here too, so a resume
        # needs no special case.
        _files = sorted(RUNS_DIR.glob("verdicts_n*.csv")) + sorted(
            RUNS_DIR.glob("verdicts_n*.partial")
        )
        _rows = (
            pd.concat([pd.read_csv(f) for f in _files])
            .drop_duplicates("post_id")
            .to_dict("records")
            if _files
            else []
        )
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

        judged = (
            pd.DataFrame(_rows)
            .drop_duplicates("post_id")
            .pipe(lambda d: d[d["post_id"].isin(sample["post_id"])])
            .reset_index(drop=True)
        )
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
    return next_post, picker, prev_post


@app.cell
def _(get_i, judged, mo, next_post, picker, prev_post):
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

    _panel = mo.Html(
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

    mo.vstack(
        [
            mo.hstack(
                [prev_post, next_post],
                justify="start",
                gap=1,
                align="center",
            ),
            picker,
            _panel,
        ]
    )
    return


@app.cell
def _(mo):
    mo.md("""
    # Analysis

    Every labelled post is judged, so the class mix is the natural one and
    the numbers below need no reweighting. Two things are reported:

    - **Accuracy** is the top-line. Read it against the always-NTA floor,
      since NTA is ~78% of posts.
    - **Recall per class** is the analysis: of the posts reddit called X,
      the share jev also called X. It is what tells us which verdicts jev
      finds and which it loses.
    """)
    return


@app.cell
def _(np, pd):
    Z95 = 1.959963984540054

    def wilson(successes, n, z=Z95):
        """CI for a single proportion. Used for accuracy and per-class recall,
        where the textbook normal interval misbehaves at the edges and for
        small classes (75 ESH, 48 NAH)."""
        if n == 0:
            return (np.nan, np.nan)
        p = successes / n
        d = 1 + z**2 / n
        centre = (p + z**2 / (2 * n)) / d
        half = z / d * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
        return (max(0.0, centre - half), min(1.0, centre + half))

    def confusion(frame, labels):
        """True class on the rows, jev's pick on the columns."""
        return (
            pd.crosstab(frame["reddit"], frame["jev"])
            .reindex(index=labels, columns=labels, fill_value=0)
            .to_numpy(dtype=int)
        )

    return confusion, wilson


@app.cell
def _(LABELS, confusion, judged, mo, np, wilson):
    cm = confusion(judged, LABELS)

    _correct, _n = int(np.trace(cm)), int(cm.sum())
    _lo, _hi = wilson(_correct, _n)
    _floor = cm.sum(1).max() / _n  # always answering the majority class

    mo.hstack(
        [
            mo.stat(
                value=f"{_correct / _n:.1%}",
                label="Accuracy",
                caption=f"95% CI [{_lo:.1%}, {_hi:.1%}] · {_correct}/{_n} posts",
                bordered=True,
            ),
            mo.stat(
                value=f"{_floor:.1%}",
                label="Always-NTA baseline",
                caption="floor to beat",
                bordered=True,
            ),
        ],
        justify="start",
    )
    return (cm,)


@app.cell
def _(LABELS, cm, pd):
    cm_display = pd.DataFrame(cm, index=LABELS, columns=LABELS)
    cm_display.index.name = "reddit"
    cm_display.columns.name = "jev"
    cm_display
    return


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
def _(LABELS, alt, cm, mo, np, pd, wilson):
    _tp = np.diag(cm)
    _support = cm.sum(1)
    _ci = [wilson(int(k), int(n)) for k, n in zip(_tp, _support)]

    recall = pd.DataFrame(
        {
            "label": LABELS,
            "support": _support,
            "recall": _tp / _support,
            "lo": [c[0] for c in _ci],
            "hi": [c[1] for c in _ci],
        }
    )

    _base = alt.Chart(recall).encode(y=alt.Y("label:N", sort=LABELS, title=None))
    _bars = _base.mark_bar().encode(
        x=alt.X("recall:Q", scale=alt.Scale(domain=[0, 1]), title="recall")
    )
    _err = _base.mark_errorbar(color="#333").encode(
        x=alt.X("lo:Q", title="recall"), x2="hi:Q"
    )

    mo.vstack(
        [
            mo.md("### Recall per class"),
            mo.ui.table(
                recall.assign(
                    **{
                        "recall": recall["recall"].round(3),
                        "95% CI": [f"[{a:.2f}, {b:.2f}]" for a, b in zip(recall["lo"], recall["hi"])],
                    }
                ).drop(columns=["lo", "hi"]),
                selection=None,
            ),
            (_bars + _err).properties(width=420, height=160),
        ]
    )
    return


@app.cell
def _(judged, mo):
    # Reddit verdicts are crowd votes on self-reported stories, not ground
    # truth. Some share of these rows are jev being right, and the only way to
    # know is to read them.
    disagreements = judged[judged["jev"] != judged["reddit"]].reset_index(drop=True)
    mo.vstack(
        [
            mo.md(
                f"### {len(disagreements)} disagreements "
                f"({len(disagreements) / len(judged):.0%} of posts)"
            ),
            disagreements[["post_title", "reddit", "jev", "confidence"]],
        ]
    )
    return


if __name__ == "__main__":
    app.run()
