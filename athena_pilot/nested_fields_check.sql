SELECT
  id,
  cardinality(referenced_works) AS reference_count,
  cardinality(topics) AS topic_count,
  cardinality(authorships) AS authorship_count,
  abstract_inverted_index IS NOT NULL AS has_abstract_index
FROM macs40123_pilot.openalex_works_source
LIMIT 5;
