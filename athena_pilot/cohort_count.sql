SELECT count(*) AS matching_works
FROM macs40123_pilot.openalex_works_source
WHERE publication_year = 2015
  AND primary_topic.field.id = 'https://openalex.org/fields/32'
  AND is_xpac = false;
