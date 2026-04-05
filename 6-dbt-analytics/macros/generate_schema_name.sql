-- ---------------------------------------------------------------------------
-- Macro: generate_schema_name
--
-- Problem this solves: by default dbt names schemas as {target_schema}_{custom_schema}
-- (e.g., "main_staging", "main_marts"). This is safe for multi-developer setups
-- but ugly for a single-env project and non-standard for production warehouses.
--
-- This override makes schemas match their layer name exactly:
--   staging → staging
--   intermediate → intermediate
--   marts → marts
--   raw → raw  (seeds)
--
-- Portfolio signal: overriding this macro is a standard pattern in production
-- dbt projects. Seeing it tells reviewers you've worked in real environments.
-- ---------------------------------------------------------------------------
{% macro generate_schema_name(custom_schema_name, node) -%}
    {%- set default_schema = target.schema -%}
    {%- if custom_schema_name is none -%}
        {{ default_schema }}
    {%- else -%}
        {{ custom_schema_name | trim }}
    {%- endif -%}
{%- endmacro %}
