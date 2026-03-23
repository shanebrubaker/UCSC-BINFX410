-- models/dim_platform.sql
-- Dimension table: sequencing platform (instrument + strategy + source)
-- Depends on: raw_runs (source table)

{{ config(
    materialized='table',
    file_format='parquet',
    write_compression='snappy',
    external_location="s3://{{ var('s3_bucket') }}/gold/star_schema/dim_platform/"
) }}

SELECT
    -- Natural key: combination of platform properties
    CONCAT(instrument_model, '|', library_strategy, '|', library_source)
                                 AS platform_key,
    instrument_model,
    library_strategy,
    library_source,

    -- Derived: assay category (useful for filtering)
    CASE
        WHEN instrument_model LIKE '%10x Genomics%'
            THEN 'Single-cell RNA-seq'
        WHEN library_strategy = 'RNA-Seq' AND library_source LIKE '%TRANSCRIPTOMIC%'
            THEN 'Bulk RNA-seq'
        WHEN library_strategy IN ('WGS', 'WES') AND library_source = 'GENOMIC'
            THEN 'DNA-seq'
        WHEN library_strategy IN ('ChIP-Seq', 'ATAC-seq', 'CUT&RUN')
            THEN 'Epigenomics'
        ELSE 'Other'
    END                          AS assay_category,

    -- Instrument generation tier
    CASE
        WHEN instrument_model LIKE '%NovaSeq X%' THEN 'Current generation'
        WHEN instrument_model LIKE '%NovaSeq%'   THEN 'Previous generation'
        WHEN instrument_model LIKE '%HiSeq%'     THEN 'Legacy'
        WHEN instrument_model LIKE '%PacBio%'    THEN 'Long-read'
        WHEN instrument_model LIKE '%Nanopore%'  THEN 'Long-read'
        ELSE 'Other'
    END                          AS instrument_tier

FROM (
    SELECT DISTINCT
        instrument_model,
        library_strategy,
        library_source
    FROM {{ ref('raw_runs') }}
    WHERE instrument_model IS NOT NULL
) t
