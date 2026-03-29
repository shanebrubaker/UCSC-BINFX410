
-- models/fact_sequencing_run.sql
-- dbt model: fact table for sequencing runs
-- depends on dim_sample and dim_study via ref()

{{ config(materialized='table', file_format='parquet') }}

SELECT
    r.run_acc AS run_key,
    s.sample_key AS sample_fk,
    st.study_key AS study_fk,
    r.release_year,
    r.bases,
    r.spots,
    r.size_mb,
    r.instrument_model,
    r.library_strategy
FROM {{ ref('raw_runs') }} r
JOIN {{ ref('dim_sample') }} s ON r.sample_acc = s.sample_key
JOIN {{ ref('dim_study') }} st ON r.study_acc = st.study_key
