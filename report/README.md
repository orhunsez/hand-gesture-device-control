# Report (LaTeX)

One template for both reports. It compiled on Overleaf without errors on 21 November 2026 (reported
by the group; the layout of the PDF was not checked by anyone else). LaTeX is not installed on the
development machine, so Overleaf is the easiest way to build it (upload this folder).

## Build

```
latexmk -pdf main.tex
```

or upload the folder `report/` to Overleaf and compile `main.tex`.

As long as the text contains no `\cite{...}`, BibTeX complains that it found no citation commands;
that is expected and goes away with the first citation. Only cite an entry of `literature.bib` after
it has been checked against the original paper.

The Progress Report PDF must be named after the group id (for example `G11.pdf`). Set the id in
`main.tex` (`\groupid`) and rename the PDF after the build.

## Progress Report or Final Report

`main.tex` has a switch near the top:

```
\finalfalse   % Progress Report (20 November 2026), the default
\finaltrue    % Final Report (4 January 2027)
```

With `\finalfalse` the final-only subsections (aim and objectives, contributions, comparison table,
experiments E1 to E9, discussion, conclusion) are left out and the progress-only section "Plan,
Challenges and Next Steps" (status table, Gantt chart, challenges, next steps) is added.

## Files

| File | Content |
|---|---|
| `main.tex` | Settings, title, the list of sections, bibliography |
| `sections/01_introduction.tex` to `08_contributions.tex` | The sections of the outline in `docs/REPORT_OUTLINE.md` |
| `sections/09_progress_plan.tex` | Progress Report only: status, Gantt chart, challenges, next steps |
| `sections/genai_declaration.tex` | Generative AI declaration (mandatory in both reports) |
| `literature.bib` | References. **All entries are unverified**; check each one against the paper |

Each section is its own file so that several people can write at the same time without merge
conflicts. Red `[TODO: ...]` notes mark what still has to be written; search for `TODO` before
submitting. Comments (`% ...`) in the files say what to write and list facts from the project.

## What is filled in already

- The gesture set table (Section 3), the pipeline figure (Section 4.1), the equations for the
  normalisation, the 1-euro filter and the palm-centre speed (Section 4).
- The provisional smoothing numbers in the Progress Report (Section 5), with their caveats.
- The status table and a draft Gantt chart (adjust the bars and add who does what).

Everything else is structure and instructions; the text is written by the group.
