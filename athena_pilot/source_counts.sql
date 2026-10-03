SELECT
  count(*) AS source_records,
  count_if(publication_year = 2015) AS records_from_2015,
  count_if(primary_topic.field.id = 'https://openalex.org/fields/32') AS primary_psychology
FROM macs40123_pilot.openalex_works_source;
