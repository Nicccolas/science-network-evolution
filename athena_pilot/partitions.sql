ALTER TABLE macs40123_pilot.openalex_works_source ADD IF NOT EXISTS
  PARTITION (updated_date='2025-10-30') LOCATION 's3://openalex-macs40123-bucket/raw/openalex/snapshot=2026-09-23/updated_date=2025-10-30/'
  PARTITION (updated_date='2025-10-15') LOCATION 's3://openalex-macs40123-bucket/raw/openalex/snapshot=2026-09-23/updated_date=2025-10-15/'
  PARTITION (updated_date='2018-10-12') LOCATION 's3://openalex-macs40123-bucket/raw/openalex/snapshot=2026-09-23/updated_date=2018-10-12/'
  PARTITION (updated_date='2017-09-15') LOCATION 's3://openalex-macs40123-bucket/raw/openalex/snapshot=2026-09-23/updated_date=2017-09-15/'
  PARTITION (updated_date='2019-08-22') LOCATION 's3://openalex-macs40123-bucket/raw/openalex/snapshot=2026-09-23/updated_date=2019-08-22/';
