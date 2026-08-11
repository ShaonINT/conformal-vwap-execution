# Manuscript source

`body_clean.tex` is the LaTeX body of the article, and `media/` holds its
figures. `VWAP_RL_SOIC.tex` is the thin wrapper that sets the document class and
`\input`s the body.

These sources will **not** compile as-is. The journal's LaTeX class
(`iapress.cls`) and the publisher's logo are the property of International
Academic Press and are not redistributed here; this repository's MIT licence
covers our own work only. To rebuild the manuscript, obtain the class file and
logo from the journal's author guidelines at

  https://iapress.org/index.php/soic/about/submissions

and place them alongside `VWAP_RL_SOIC.tex`.

For reading rather than rebuilding, use the published version in the parent
directory — it is the authoritative record:

  ../Irshad-Biswas-2026-SOIC-uncertainty-aware-execution.pdf
