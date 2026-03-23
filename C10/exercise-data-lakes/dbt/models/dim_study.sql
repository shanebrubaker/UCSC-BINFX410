-- models/dim_study.sql
-- Dimension table: research studies (BioProjects)
-- Depends on: raw_runs (source table)

{{ config(
    materialized='table',
    file_format='parquet',
    write_compression='snappy',
    external_location="s3://{{ var('s3_bucket') }}/gold/star_schema/dim_study/"
) }}

SELECT
    study_acc                    AS study_key,
    bioproject,
    center_name,
    release_year,

    -- Derived: is this a recent study?
    CASE
        WHEN release_year >= 2022 THEN TRUE
        ELSE FALSE
    END                          AS is_recent_study,

    -- Center categorization
    CASE
        WHEN center_name IN ('Broad Institute', 'Sanger', 'NIH') THEN 'Academic'
        WHEN center_name IN ('BGI', 'Illumina') THEN 'Commercial'
        ELSE 'Other'
    END                          AS center_type

FROM (
    SELECT DISTINCT
        study_acc,
        bioproject,
        center_name,
        release_year
    FROM {{ ref('raw_runs') }}
    WHERE study_acc IS NOT NULL
) t
