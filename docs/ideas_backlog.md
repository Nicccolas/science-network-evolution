# Ideas backlog

Extensions and alternatives that are **not** in the core design
([`research_design.md`](research_design.md)) but should not be forgotten. Each idea
lists where it came from, what it would measure, how it connects to the research
question, and known pitfalls. Promote an idea into the design only through an
entry in [`decision_log.md`](decision_log.md).

Status values: `idea` (not started), `planned` (agreed, not built), `in progress`,
`done`, `dropped` (keep the entry and give the reason).

| # | Idea | Source | Status |
| --- | --- | --- | --- |
| 1 | Nearest-neighbour composition of each paper | Motohashi et al. (2024); Week 1 review | idea, high priority |
| 2 | Sliding 3-year windows instead of single years | Project notes | idea (optional) |
| 3 | Journal-based "ground truth" field labels | Alasehir & Acarturk (2022); Week 1 review and reflection | idea |
| 4 | Field-centroid projection / field affinity | Alasehir & Acarturk (2022) | idea |
| 5 | Pair the distance metric with the matching centroid and clustering method | Ng et al. (2024); Week 1 review | idea, affects core design |
| 6 | Validate embeddings with citation pairs vs random pairs | Motohashi et al. (2024) | idea, cheap and recommended |
| 7 | Compare representations (count vectors, TF-IDF, other embedders) | Week 1 notebook; `exploratory_analysis.ipynb` | idea (robustness) |
| 8 | Embeddings to track change over time | Kozlowski et al. (class reading); Week 1 reflection | idea |
| 9 | Local density / crowding alongside KE | Suggested extension | idea |
| 10 | Earlier question: deep learning and cross-disciplinary citation networks | Earlier version of this repo | idea (parked) |
| 11 | Second case study | Research design | idea (only if core work is finished) |

---

## 1. Nearest-neighbour composition (Motohashi et al., 2024)

**Method in the source.** Motohashi, Koshiba, and Ikeuchi turned papers and patents
into 300-dimensional document vectors, found each document's 200 nearest neighbours
by cosine similarity, and used neighbour composition as a measure. A patent with 180
of 200 neighbours that are papers has *science intensity* 180/200 = 0.9. They checked
that cosine similarity is meaningful by showing that citing–cited pairs are more
similar than random pairs. Over time, science intensity rose for patents while
technology intensity fell for papers, which they read as technology drawing on
science without feeding much back. Their setting was pharmaceuticals and the life
sciences.

**Adaptation.** For each focal paper *i* in year *t*, find its *k* nearest neighbours
in a reference set *R*, then summarize the distribution of some attribute among those
neighbours. The attribute is the key choice, and each choice measures something
different:

| Neighbour attribute | What the composition represents | Link to the research question |
| --- | --- | --- |
| **OpenAlex field / subfield** (outside the focal field) | *Semantic interdisciplinarity* or boundary position: how much of the paper's local neighbourhood belongs to other fields. Unlike reference-based measures, it uses content, not citing behaviour. | Shows whether KE growth comes from the field **moving into territory other fields already occupy** (convergence, rising out-of-field share) or **into unoccupied territory** (expansion without a rising out-of-field share). This is the most direct complement to KE. |
| **Field, focal field only** (share of neighbours from the paper's own field) | *Semantic coherence* or the "self-share" of the field. | A falling self-share during growth means the field's boundaries are blurring. |
| **Publication year of neighbours** (earlier vs later than *t*) | Backward vs forward similarity. Many earlier neighbours suggest conventional or derivative work; many later neighbours suggest anticipatory work that later papers resemble. This follows Kelly, Papanikolaou, Seru & Taddy (2021), who score patent importance as forward similarity relative to backward similarity. | Tests whether growth years produce more derivative work (consistent with Chu & Evans, 2021) or more anticipatory work. Gives paper-level novelty, which the core design explicitly does not measure. |
| **Shared clusters** (from the core clustering) | Local thematic mixing: whether a paper sits inside one cluster or at the junction of several. | Separates "more evenly spread clusters" from "more papers bridging clusters". |
| **Institution type** (education vs company; OpenAlex institution `type`) | An analogue of Motohashi's science/technology intensity: how applied or industry-adjacent a paper's neighbourhood is. | Asks whether growth is driven by applied or industry-adjacent work. |
| **Document type or source type** (journal vs repository/preprint; later, patents) | The closest analogue of the original paper–patent design. | Asks whether preprint expansion accounts for growth. |
| **Citation tier of neighbours** (highly cited vs not, citations normalized for age) | Whether a paper sits near the established canon. | Links directly to Chu & Evans: does growth pull work toward highly cited cores? |
| **AI-related flag** (keyword or classifier) | Whether AI-related work fills the neighbourhood. | Links to Hao et al. (2026), where AI is associated with narrower KE. |
| **Country / region of authors** | Geographic composition of semantic neighbourhoods. | Optional; relates to who is contributing the growth. |

**Summaries of a composition vector *p*.** These include out-of-field share (1 − *p*<sub>focal</sub>),
Shannon entropy, Gini–Simpson (1 − Σ*p*²), and Rao–Stirling diversity, which also
weights how different the fields are (field distances can come from field-centroid
distances in the same embedding space). Aggregate to years with medians or means,
or report the full distribution.

**Design requirements and pitfalls.**

- **The reference set must include other fields.** If *R* contains only the focal
  field, field composition is trivially 100% focal. Use a multi-field reference
  sample, such as a stratified random sample of all OpenAlex fields (or adjacent
  fields) per year. This is the main extra data cost; see
  [`data_strategy.md`](data_strategy.md).
- **Control for base rates.** Neighbour composition reflects how common each field is
  in *R*. Divide by the expected share or compare against a permutation null (shuffle
  labels within *R*). If other fields grow faster than the focal field, the raw
  out-of-field share rises mechanically.
- **Hold reference size constant across years.** Neighbourhood radius shrinks as *R*
  grows, which is the same mechanical issue the fixed 2,000-paper KE sample avoids.
  Use either a fixed-size *R* per year or one fixed, year-balanced *R*. Record which
  one and why.
- **Same-year vs all-years *R*.** A same-year *R* measures contemporaneous composition.
  An all-years *R* is required for the backward/forward version.
- **Choice of *k*.** Motohashi used 200. Report sensitivity across a few values of *k*
  (for example 20, 50, 200). The right *k* depends on the size of *R*.
- **Hubness.** In high-dimensional spaces some points are neighbours of many others
  (Radovanović et al., 2010). Check the distribution of how often each paper appears
  as a neighbour.
- **Label noise.** The Week 1 review notes that OpenAlex topic labels come from a
  classifier with reported accuracy of about 81.7% (verify this figure and its source
  before citing it). Consider a minimum `primary_topic_score`, subfield rather than
  topic level, or journal-based labels (idea 3).
- **Metric consistency.** Motohashi used cosine; KE uses Euclidean. On L2-normalized
  embeddings, cosine and Euclidean give the same neighbour ranking (see idea 5).
- **Compute.** Exact kNN with `sklearn.neighbors.NearestNeighbors` is fine up to a few
  hundred thousand reference points. Beyond that, use FAISS (`faiss-cpu`), which is
  not in the environment yet.

**Validation for free.** A paper's majority neighbour field vs its assigned OpenAlex
field is a diagnostic of label noise in its own right.

---

## 2. Sliding 3-year windows (optional)

Instead of one KE distribution per calendar year, pool papers from years
*t*−1, *t*, *t*+1 (or *t*−2..*t*) and sample 2,000 from the pool.

**Advantages.** Smoother series; larger pools for small fields or early years; less
sensitivity to publication-date noise (online-first vs issue year, OpenAlex
`publication_year` quirks); and clusters and kNN composition get more papers per
window.

**Costs and pitfalls.**

- Overlapping windows share papers, so adjacent values are correlated **by
  construction**. Do not feed overlapping-window values into the Δlog regression as if
  they were independent observations. Either use non-overlapping 3-year blocks (fewer
  observations) or use windows only for descriptive figures while keeping the
  regression annual.
- Publication counts must be aggregated over the same windows.
- Windows blur the timing of sharp booms, so report the annual series alongside.

**Decision.** Default to annual. Treat windows as a robustness or visualization option.

---

## 3. Journal-based "ground truth" field labels (Alasehir & Acarturk, 2022)

Alasehir & Acarturk used papers from discipline-specific journals (for example
*Journal of Personality and Social Psychology*) as labelled examples for each
discipline. Journal membership does not guarantee discipline, but it is arguably
closer than a text classifier.

**Use here.** (a) An alternative field definition for robustness: field = papers in a
curated list of core journals (OpenAlex `primary_location.source.id`). (b) Validate
OpenAlex labels by cross-referencing the topic-based corpus against core-journal
papers. (c) Cleaner labels for the kNN field composition (idea 1). The Week 1
reflection notes that adding some form of ground truth would strengthen the study.

---

## 4. Field-centroid projection / field affinity (Alasehir & Acarturk, 2022)

They built document embeddings for adjacent fields (psychology, linguistics,
anthropology, and so on), projected cognitive-science papers into that space, and
used cosine similarity to field representations to judge which adjacent fields are
most present in cognitive science.

**Use here.** For each focal paper, compute similarity to the centroid of each
reference field. This is a cheaper, smoother alternative to kNN composition (no
neighbour search) and a robustness check for idea 1. Year-by-year field affinities
show which adjacent fields the focal field is drifting toward.

---

## 5. Pair the metric with the centroid and clustering method (Ng et al., 2024)

Ng et al. used Manhattan distance, citing poor Euclidean performance in high
dimensions, and therefore k-medians: the coordinate-wise median minimizes summed
Manhattan distance, while k-means minimizes squared Euclidean distance with the mean.

**Implications for the core design.**

- KE uses Euclidean distance to the **mean**, which is consistent with k-means.
- If Manhattan distance is tested, the centroid should be the **median** and
  clustering should use k-medians.
- For cosine, L2-normalize embeddings: Euclidean distance on unit vectors is a
  monotone function of cosine, and k-means on normalized vectors approximates
  spherical k-means.
- **Normalization decision needed.** SPECTER2 vectors are not unit-length. If vector
  norms drift with year, abstract length, or terminology, raw Euclidean KE can change
  for reasons unrelated to direction. Compute KE on both raw and normalized vectors,
  or choose one and justify it. Log the choice in the decision log.

---

## 6. Citation-pair validation (Motohashi et al., 2024)

Show that citing–cited pairs are more similar than random pairs (and than same-field
random pairs). This uses the citation data `api_test/extract.py` already collects, is
cheap, and supports the claim that embedding distance tracks substantive relatedness.
It can also be used to compare candidate embedding models in the model-selection
pilot.

---

## 7. Compare representations (robustness)

`exploratory_analysis.ipynb` found that count vectors and TF-IDF give similar
neighbours, while MiniLM embeddings give quite different ones (about 10-neighbour
overlap). Possible robustness checks: KE on TF-IDF reduced with truncated SVD or LSA,
and KE with a second embedder (MiniLM or a general-purpose model) to show that the
trend does not depend on SPECTER2. Also consider title-only vs title+abstract input.

---

## 8. Embeddings to track change over time

The Week 1 reflection cites the class discussion of Kozlowski and Evans (likely
Kozlowski, Taddy & Evans, 2019, *The Geometry of Culture*; confirm which reading was
meant). That discussion motivated the year-by-year design. Possible extensions:
centroid drift between years, how far each year's centroid moves from the
year-balanced reference centroid, and semantic change of key terms.

---

## 9. Local density / crowding (suggested extension)

KE captures the extremes of the distribution. A complementary measure is the mean
distance to the *k*-th nearest neighbour within a fixed-size annual sample. Shrinking
kNN distances with stable KE would mean growth **fills in** existing territory
(crowding). Growing KE with stable density would mean growth **extends** the frontier.
This reuses the same saved vectors and the same fixed-sample logic.

---

## 10. Earlier question: deep learning and cross-disciplinary citation networks (parked)

> As deep-learning methods spread across scientific disciplines, do scientific research
> communities become increasingly cross-disciplinary, or do they remain separated along
> disciplinary boundaries?

The existing infrastructure supports this: `api_test/extract.py` builds directed
citation edge lists, and the environment includes networkx, igraph, and leidenalg.
Partial reuse within the current project:

- Compare **reference-based** interdisciplinarity (field diversity of cited works,
  Rao–Stirling) with **semantic** interdisciplinarity (idea 1).
- An AI or deep-learning flag as a neighbour attribute (idea 1 table) links this
  question to Hao et al. (2026).

---

## 11. Second case study

Only if the core single-field analysis is complete within the course schedule. Prefer a
field with a contrasting growth trajectory, such as steady vs boom.
