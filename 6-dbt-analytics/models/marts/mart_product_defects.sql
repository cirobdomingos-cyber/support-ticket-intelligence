/*
  Model   : mart_product_defects
  Grain   : One row per product_family + component_name + fault_code combination
  Source  : int_tickets_enriched

  Purpose : Identify which product families, components, and fault patterns
            drive the most tickets, critical issues, and warranty costs. Answers:
              - Which fault codes appear most often on which product families?
              - Where are warranty claims concentrated by component?
              - Which defect patterns affect the most distinct vehicles?

  Use case: Quality engineering and product reliability teams use this to
            prioritise design improvements and field service campaigns.
*/

with tickets as (

    select * from {{ ref('int_tickets_enriched') }}

),

aggregated as (

    select
        -- ── Dimensions ────────────────────────────────────────────────────────
        product_family,
        product_code,
        component_name,
        fault_code,
        fault_mode,

        -- ── Volume ────────────────────────────────────────────────────────────
        count(*)                                            as total_tickets,
        count(*) filter (where is_closed)                  as closed_tickets,
        count(*) filter (where is_critical)                as critical_tickets,
        count(*) filter (where is_warranty_claim)          as warranty_claims,
        count(*) filter (where is_sla_breach = true)       as sla_breached_tickets,
        count(distinct vehicle_chassis_id)                 as affected_vehicles,
        count(distinct dealer_code)                        as affected_dealers,

        -- ── Rates ─────────────────────────────────────────────────────────────
        round(
            100.0
            * count(*) filter (where is_critical)
            / nullif(count(*), 0),
            2
        )                                                   as critical_rate_pct,

        round(
            100.0
            * count(*) filter (where is_warranty_claim)
            / nullif(count(*), 0),
            2
        )                                                   as warranty_rate_pct,

        round(
            100.0
            * count(*) filter (where is_sla_breach = true)
            / nullif(count(*) filter (where is_closed), 0),
            2
        )                                                   as sla_breach_rate_pct,

        -- ── Resolution time ───────────────────────────────────────────────────
        round(avg(resolution_hours)    filter (where is_closed), 2) as avg_resolution_hours,
        round(median(resolution_hours) filter (where is_closed), 2) as median_resolution_hours

    from tickets
    group by
        product_family,
        product_code,
        component_name,
        fault_code,
        fault_mode

)

select * from aggregated
order by total_tickets desc
