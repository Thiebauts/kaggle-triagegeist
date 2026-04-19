"""Van Gogh Almond Blossoms palette — canonical source for the Kaggle notebook.

The first three keys (`blue`, `gold`, `sage`) are extracted directly from
`report/thiebaut-report.sty`. The remaining three are accent / background /
text values used by the in-notebook figures and are documented here so the
two surfaces stay in sync.

Provenance:

    blue  = vg-heading       (thiebaut-report.sty line 23)
    gold  = vg-blossom       (line 20)
    sage  = vg-leaf          (line 19)
    rust  = COLOR_POS        (report/latex-figures/generate_all_figures.py)
    cream = figure background (matches the Task-5 in-notebook figures)
    text  = figure body text  (matches the Task-5 in-notebook figures)

The notebook can't easily import this module on Kaggle, so cell 2 pastes the
VANGOGH dict inline. Keep this file as the single source of truth — update
both places together when the palette changes.
"""

VANGOGH = {
    'blue':  '#3D6B7E',   # vg-heading: deep navy/teal
    'gold':  '#C4A882',   # vg-blossom: warm golden ochre
    'sage':  '#6B8E6B',   # vg-leaf: muted sage green
    'rust':  '#E8525A',   # warm rust accent (used for severe / under-triage)
    'cream': '#F2EBD9',   # warm cream figure background
    'text':  '#3A3530',   # dark warm body text
}

# Per-class ESI assignment for figures generated in the notebook.
# Note: this differs from the report's existing ESI palette (which runs
# vg-branch -> vg-heading -> vg-sky -> vg-leaf -> vg-blossom). The mapping
# below was specified in the Kaggle-submission handoff.
ESI_COLORS = {
    1: VANGOGH['rust'],   # resuscitation — strongest warm tone
    2: VANGOGH['gold'],   # emergent
    3: VANGOGH['sage'],   # urgent
    4: VANGOGH['blue'],   # less urgent
    5: VANGOGH['text'],   # non-urgent — darkest / coolest
}
