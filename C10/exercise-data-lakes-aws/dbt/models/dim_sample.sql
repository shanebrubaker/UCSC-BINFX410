
-- models/dim_sample.sql
-- dbt model: dimension table for biological samples
-- ref() creates a dependency on the raw_runs source table

{{ config(materialized='table', file_format='parquet') }}

SELECT
    sample_acc AS sample_key,
    organism,
    tax_id,
    tissue,
    disease,
    sex,
    CASE
        WHEN organism = 'Homo sapiens' THEN 'Human'
        WHEN organism = 'Mus musculus' THEN 'Mouse'
        ELSE 'Other'
    END AS organism_group
FROM (
    SELECT DISTINCT sample_acc, organism, tax_id, tissue, disease, sex
    FROM {{ ref('raw_runs') }}  -- dbt ref() creates tracked dependency
) t
