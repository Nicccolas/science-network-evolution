# Research design (canonical proposal)

> **Status:** This is the active proposal as of 2026-10-10. It replaces the earlier
> "deep learning and cross-disciplinary citation networks" question; that idea is
> kept in [`ideas_backlog.md`](ideas_backlog.md). Record changes to this design in
> [`decision_log.md`](decision_log.md) instead of editing the proposal silently.
> The original text contained hyperlinks (Hanson et al., Milojević, Chu & Evans,
> Hao et al., OpenAlex work attributes, SPECTER2 documentation, the published KE
> implementation, and clustering references) that were lost when it was pasted.
> Restore them in [`literature.md`](literature.md).

**Proposed title:** Publication Growth and the Semantic Breadth of Science: A Longitudinal Case Study Using Document Embeddings and Clustering

## Research question

> How does the semantic breadth of research change as publication volume grows within a scientific field, and how do shifts in thematic composition accompany this relationship?

Supporting questions:

- Does annual knowledge extent (KE) increase, remain stable, or decline as publication volume grows, and how do their relative rates of change compare?
- Does publication growth coincide with a more even distribution of research across thematic clusters, or with greater concentration in particular themes?

## Background and prior research

Scientific publication volume has increased substantially, raising questions about whether greater output corresponds to a broader range of research. Hanson, Gómez Barreiro, Crosetto, and Brockington (2024), *The strain on scientific publishing*, report that the number of articles indexed in Scopus and Web of Science was about 47% higher in 2022 than in 2016. They examine the resulting pressure on publishing and peer review. This shows that publication growth matters, but not that semantic breadth grows at the same rate.

Milojević (2015) separates publication volume from cognitive extent. Using title vocabulary from 20 million articles in physics, astronomy, and biomedicine, the study measures cognitive breadth while holding the amount of compared text constant. It finds that periods of cognitive expansion do not necessarily coincide with publication growth. This is the closest conceptual precedent for measuring breadth independently of the number of publications.

Chu and Evans (2021) examine how the size of a scientific field relates to citation dynamics. They find that larger fields show greater persistence of established, highly cited literature and less disruption. This motivates asking whether growth comes with concentration around established research, although citation persistence and semantic breadth are different outcomes.

Hao, Xu, Li, and Evans (2026) provide the methodological foundation. Their analysis of 41.3 million natural-science papers links AI-augmented research to narrower collective knowledge extent. They use document embeddings to represent research content and compare the spatial dispersion of sampled papers. This project adapts their approach to annual comparisons within one field and adds clustering to interpret changes in thematic composition.

Together, these studies motivate asking how publication growth relates to the structure of research content. The contribution is a longitudinal case study that combines standardized annual KE estimates, growth comparisons, and thematic clustering. It does not assume that publication growth produces either expansion or contraction.

## Case selection and scope

The field and observation period will be chosen in an initial exploratory phase. Candidate fields or subfields will be assessed on annual publication counts, abstract availability, a consistent definition of field membership, and computational feasibility.

A pronounced boom may make a case especially informative, but it is not required; gradual growth also allows the question to be studied. The case will be selected from publication trajectories and data availability *before* examining the relationship between growth and KE, and the selection process will be documented.

The study prioritizes one case with a long sequence of complete publication years. A second case will be considered only if the core analysis is finished within the course schedule. Findings are evidence about the selected case, not about all of science.

## Data and preparation

OpenAlex will provide publication metadata and abstracts. The analytical corpus is English-language publications with usable titles and abstracts. Final publication-type rules will be set during the pilot and applied consistently across years.

Collected fields: OpenAlex ID, DOI where available, title, abstract, publication year and date, language, publication type, and topic classifications. Abstracts are reconstructed from OpenAlex's inverted index. Duplicate records and identifiable duplicate versions are checked before analysis.

The study distinguishes two counts:

- The annual number of publications meeting the field, language, and publication-type criteria.
- The annual number, and proportion, of those that also have usable abstracts.

KE is calculated from the second group. Both series are reported so that changes in abstract availability are not mistaken for changes in publication activity.

## Semantic representation

Each paper's title and abstract are converted into a vector with a pretrained contextual document-embedding model, to capture semantic relationships beyond word-frequency overlap.

The final model is chosen after a pilot that assesses computational cost and the relevance of nearest-neighbour papers. **SPECTER2 is the provisional candidate**: it is designed for scientific titles and abstracts and provides continuity with the methodological foundation. Its training on scientific-document relationships does not guarantee that every scientific distinction is captured accurately.

One fixed model, model version, and preprocessing procedure is used for all years. Input-length limits and truncation are documented. Each unique paper is embedded once; sampling and clustering reuse the stored vectors.

## Annual knowledge extent

The primary measure follows the maximum-distance-to-centroid approach implemented by Hao and colleagues. For each annual sample, KE is the greatest Euclidean distance between a paper embedding and that sample's mean embedding.

Procedure:

1. Randomly select 2,000 distinct papers from a year's eligible corpus, without replacement.
2. Calculate the centroid of their embeddings.
3. Calculate each sampled paper's distance from that centroid.
4. Record the largest distance as the sample's KE.
5. Repeat 1,000 times; papers may reappear across repetitions.
6. Summarize the KE values with their median and quartiles.

The centroid is recalculated for every sample, so the annual median summarizes 1,000 sample-level maximum distances.

Sample size is constant across years. If any year has fewer than 2,000 eligible papers, a smaller common sample size is adopted. Annual pools should ideally be much larger than the sample.

Holding sample size constant removes the mechanical advantage that larger pools would have if maximum distance were calculated over every paper. The quantity describes **the breadth encountered in a fixed-size sample of annual research**, not the full extent of the annual corpus.

Robustness: the 95th percentile of distances from each sample centroid is also recorded. Comparing it with the maximum distance shows whether changes involve a broad part of the literature or mainly a few extreme papers.

## Publication growth and time-series analysis

Descriptive figures: annual publication counts and annual KE box plots in vertically aligned panels sharing a time axis, plus a figure expressing counts and median KE relative to a common baseline year so their percentage changes are comparable.

Formal analysis: a parsimonious time-series regression of annual changes in log KE on annual changes in log publication counts. This asks whether years with faster publication growth also show faster KE growth, rather than relying on correlations between two trending series.

Residual temporal dependence is assessed and modelled where the series supports it; the number of annual observations limits model complexity.

A publication-growth coefficient of one is an equal-percentage-change benchmark, conditional on the model's other terms. Strict proportionality also requires no separate drift. These are descriptive scaling benchmarks: a 10% increase in this KE measure is not 10% more scientific knowledge.

The 1,000 sampling repetitions are **not** independent historical observations. Annual box-plot quartiles describe variation across sampled paper sets; they are not confidence intervals for the annual median or tests of the relationship with publication growth.

## Thematic clustering

Clustering gives an interpretable account of changes in research composition.

One clustering model is fitted to a pooled set of embeddings with equal representation from each year. A single model keeps categories comparable over time, and balancing the fitting sample stops high-volume years from dominating the cluster definitions.

K-means is the provisional starting point. Several cluster counts are evaluated on separation, stability across initializations, and interpretability of representative papers. The final choice is documented and is **not** chosen by which produces the strongest temporal pattern.

Papers from all years are then assigned to the same clusters. The analysis examines:

- Annual publication counts and proportions within each cluster.
- Changes in the share held by the largest clusters.
- Evenness of the annual distribution across clusters, summarized by entropy.
- Representative papers and descriptive terms that interpret each cluster.
- Whether clusters gaining share are central or peripheral relative to a fixed, year-balanced reference centroid.

Cluster proportions are visualized over time and read alongside KE. For example, rising KE with growth in peripheral themes is a different compositional pattern from stable KE with growing concentration in a dominant theme.

The clusters are retrospective analytical categories. Their growth or decline is not automatically read as the creation or disappearance of scientific topics.

## Validation and limitations

Validation: inspect papers responsible for extreme distances, check sensitivity to alternative fixed sample sizes, and compare maximum-distance results with the 95th-percentile measure. Assess cluster stability and the plausibility of thematic labels.

Limitations: abstracts summarize research imperfectly, and changing terminology can affect embeddings. Field boundaries, database coverage, and missing abstracts can influence trends. Maximum distance is sensitive to outliers and centroid shifts and does not measure the volume of occupied semantic space.

The project evaluates changing semantic dispersion and composition. It does not directly measure paper-level novelty, scientific quality, or cumulative progress. Publication growth and KE may respond to shared influences, so their association does not establish a causal effect.

## Course methods and feasibility

| Course method | Application |
| --- | --- |
| Measuring distance in high-dimensional space | Annual KE and the 95th-percentile dispersion measure |
| Clustering in Euclidean/non-Euclidean spaces | Shared thematic categories and their changing annual composition |
| Time-series models | Relationships between annual publication growth and KE growth |

One embedding dataset is reused for all three components. A pilot determines the feasible corpus size and embedding runtime. Repeated sampling is numerical work on saved vectors, not repeated model inference.

| Period | Planned work |
| --- | --- |
| Weeks 1–2 | Explore candidate fields, select the case, establish inclusion rules, audit coverage |
| Weeks 3–4 | Prepare text, evaluate the embedding model, generate embeddings |
| Week 5 | Calculate annual KE distributions and robustness measures |
| Week 6 | Fit and interpret shared thematic clusters |
| Week 7 | Complete growth comparisons and time-series analysis |
| Week 8 | Sensitivity checks; inspect influential observations |
| Week 9 | Finalize figures, interpretation, and the written report |

Deliverables: a reproducible analysis, annual publication and KE trajectories, thematic-composition figures, and a case-study report explaining where publication growth coincides with broader semantic dispersion and where the two diverge.
