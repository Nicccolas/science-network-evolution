# Literature notes

How each work is used in the project. **Full bibliographic details (venue, volume,
DOI) are still to be verified.** Do not cite from this file in the paper until each
entry is checked and moved into a bibliography file such as `paper/references.bib`.

## Core motivation (from the research design)

| Work | Key point | Role in this project |
| --- | --- | --- |
| Hanson, Gómez Barreiro, Crosetto & Brockington (2024), *The strain on scientific publishing* | Scopus/WoS-indexed article totals were about 47% higher in 2022 than in 2016; strain on publishing and peer review. | Shows that publication growth matters. Does not show that breadth grows with it. |
| Milojević (2015) | Cognitive extent from title vocabulary over 20M articles (physics, astronomy, biomedicine), holding the amount of text constant; expansion does not track publication growth. | Closest conceptual precedent: breadth measured independently of volume. Motivates the fixed-sample design. |
| Chu & Evans (2021) | Larger fields show more persistence of established, highly cited work and less disruption. | Motivates asking whether growth means concentration. Citation persistence ≠ semantic breadth. Also motivates the "citation tier" neighbour attribute (backlog idea 1). |
| Hao, Xu, Li & Evans (2026) | 41.3M natural-science papers; AI-augmented research is associated with narrower collective knowledge extent; KE = maximum distance to the centroid of sampled paper embeddings. | **Direct methodological foundation** for KE. Their published implementation is the reference to restore. |

## Week 1 literature review (`Week1_notebook.ipynb`)

| Work | Key point | Role in this project |
| --- | --- | --- |
| Ng et al. (2024), cognitive neuroscience | Clustered 360-region brain-activation vectors into 6 patterns using **Manhattan distance + k-medians**, citing poorer Euclidean performance in high dimensions. The median minimizes Manhattan distance; the mean minimizes squared Euclidean distance. | The metric, centroid, and clustering method must be chosen together. KE (Euclidean, mean) is consistent with k-means. Testing Manhattan would require medians and k-medians. Backlog idea 5. |
| Alasehir & Acarturk (2022) | Doc2Vec on titles and abstracts of papers from discipline-specific journals (psychology, linguistics, anthropology, …); projected cognitive-science papers into the space; cosine similarity showed which adjacent fields are most present. | (a) Journal-based labels as quasi-ground truth versus the OpenAlex classifier; (b) field-centroid affinity. Backlog ideas 3 and 4. |
| Motohashi, Koshiba & Ikeuchi (2024) | 300-d document vectors for papers and patents; **200 nearest neighbours by cosine**; neighbour composition = science/technology intensity; validated by showing citing pairs are more similar than random pairs; patents' science intensity rose while papers' technology intensity fell (pharma / life sciences). | **Nearest-neighbour composition** method (backlog idea 1) and **citation-pair validation** (backlog idea 6). |
| OpenAlex topic classifier (cited in the Week 1 review as ~81.7% accurate) | OpenAlex topic labels are classifier output, not ground truth. | Motivates label-noise checks and journal-based alternatives. Find and cite the OpenAlex topic-classification documentation for the exact figure. |
| Kozlowski & Evans (class discussion; Week 1 reflection) | Embeddings can track cultural or semantic change over time. Probably Kozlowski, Taddy & Evans (2019), *The Geometry of Culture*; confirm which reading. | Motivated the year-by-year design. Backlog idea 8. |

## Methods and tools

| Work | Role |
| --- | --- |
| SPECTER (Cohan et al., 2020) and SPECTER2 / SciRepEval (Singh et al., 2023) | Provisional embedding model; cite the model card and paper, and record the model revision. |
| OpenAlex documentation (works object, `abstract_inverted_index`, topics, API paging and `group_by`) | Data source and field definitions. |
| Clustering evaluation (silhouette, stability across seeds) | Cluster-count selection; restore the link from the original proposal. |

## Suggested additions (not from the user's reading list; verify before relying on them)

| Work | Why it may help |
| --- | --- |
| Kelly, Papanikolaou, Seru & Taddy (2021), "Measuring technological innovation over the long run" | Backward vs forward text similarity as novelty and impact. Basis for the "neighbour publication year" composition (backlog idea 1). |
| Stirling (2007) and Rao–Stirling diversity; Leinster & Cobbold (2012) | Diversity indices that account for field dissimilarity, for summarizing neighbour composition. |
| Porter & Rafols (2009); Leahey et al. (2017) | Reference-based interdisciplinarity, as a comparison to semantic interdisciplinarity. |
| Radovanović, Nanopoulos & Ivanović (2010), hubs in high-dimensional data | The hubness caveat for kNN in embedding spaces. |
| Park, Leahey & Funk (2023), papers becoming less disruptive | Context for growth vs disruption alongside Chu & Evans. |
