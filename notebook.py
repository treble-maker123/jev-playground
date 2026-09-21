import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import pandas as pd

    DATA_DIR = (mo.notebook_dir() or Path.cwd()) / "data"
    return DATA_DIR, alt, pd


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
    # NAH - no asshole here

    # excluded
    # INFO - not enough info
    # YWBTA - you would be the asshole - hypothetical
    # YWNBTA - you would not be the asshole - hypothetical

    alt.Chart(filtered["verdict"].value_counts().reset_index()).mark_bar().encode(
        x="count:Q",
        y=alt.Y("verdict:N", sort="-x"),
    )
    return


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
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
