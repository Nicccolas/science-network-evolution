# Decision log

Record every analytic choice that could change results, **with the date and the
reason, made before seeing the outcome where possible**. This is the audit trail for a
publication-ready paper and guards against choosing specifications by their results.
Add new entries at the top.

## Open decisions

| Decision | Options | Decide by | Notes |
| --- | --- | --- | --- |
| Case field / subfield | Shortlist from the pilot | Weeks 1–2 | Criteria: annual counts, abstract coverage, consistent membership definition, feasibility. Choose **before** computing KE vs growth. |
| Field membership rule | OpenAlex primary topic/subfield/field; minimum `primary_topic_score`; journal list (backlog idea 3) | Weeks 1–2 | |
| Observation period | Complete years only; exclude the current, incomplete year | Weeks 1–2 | Watch for the abstract-coverage drop in recent years. |
| Publication types | `article` only vs article + review, etc. | Pilot | Apply consistently across years. |
| "Usable abstract" rule | Minimum length/tokens; boilerplate filters | Pilot | The current notebooks use `len > 20` characters, which is a placeholder. |
| Embedding model and revision | SPECTER2 (provisional) vs alternatives | Weeks 3–4 | Judge by cost + nearest-neighbour relevance + citation-pair validation (backlog idea 6). |
| Embedding input | Title + [SEP] + abstract (SPECTER convention) vs abstract only | Weeks 3–4 | The Week 1 notebooks embed the abstract only, with MiniLM. |
| Vector normalization for KE | Raw vs L2-normalized | Week 5 | Backlog idea 5. Possibly report both. |
| Per-year pool size | 30k planning number | Pilot | Confirm that KE is stable between pool sizes; see `data_strategy.md`. |
| KE sample size | 2,000 (or smaller common size) | After counts are known | Sensitivity: alternative fixed sizes. |
| Time unit | Annual (default) vs sliding 3-year windows | Week 5 | Backlog idea 2; windows only for robustness or figures. |
| Cluster count *k* | Evaluated on separation, stability, interpretability | Week 6 | Must not be chosen by the strength of the temporal pattern. |
| Time-series specification | Δlog KE ~ Δlog count (+ drift); dependence structure | Week 7 | Limited by the number of years. |
| Include kNN composition? | Backlog idea 1 | After core KE is done | Requires a multi-field reference sample. |

## Decisions made

| Date | Decision | Reason |
| --- | --- | --- |
| 2026-10-10 | Embedding stack: conda-forge `pytorch` 2.14.1, `transformers` 4.57.6, `sentence-transformers` 5.7.0, pip `adapters` 1.3.0. SPECTER2 base + proximity adapter verified on `mps`. | `adapters` requires `transformers~=4.57.6`; pip-installed torch crashed on import (duplicate OpenMP). |
| 2026-10-10 | Research question changed from "deep learning and cross-disciplinary citation networks" to "publication growth and semantic breadth" (`research_design.md`). Old question parked in the backlog (idea 10). | Current proposal. |
| 2026-10-10 | Default data route: OpenAlex API with random per-year samples + exact `group_by` counts, kept under ~8 GB locally; AWS only as a fallback. | ~13 GB free disk; limited AWS experience; KE needs fixed-size samples, not full censuses (`data_strategy.md`). |
