-- OpenAlex Parquet snapshot 2026-09-23. Only the five listed partitions are registered.
-- This is a projection of columns needed for the bounded technical pilot.
CREATE EXTERNAL TABLE IF NOT EXISTS macs40123_pilot.openalex_works_source (
  id string,
  doi string,
  title string,
  publication_date date,
  publication_year int,
  language string,
  type string,
  authorships array<struct<author:struct<display_name:string,id:string>,author_position:string>>,
  primary_topic struct<id:string,display_name:string,score:float,subfield:struct<id:string,display_name:string>,field:struct<id:string,display_name:string>,domain:struct<id:string,display_name:string>>,
  topics array<struct<id:string,display_name:string,score:float,subfield:struct<id:string,display_name:string>,field:struct<id:string,display_name:string>,domain:struct<id:string,display_name:string>>>,
  is_xpac boolean,
  referenced_works array<string>,
  abstract_inverted_index string
)
PARTITIONED BY (updated_date string)
STORED AS PARQUET
LOCATION 's3://openalex-macs40123-bucket/raw/openalex/snapshot=2026-09-23/';
