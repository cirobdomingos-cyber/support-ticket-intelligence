/*
  Model   : mart_dealer_performance
  Grain   : One row per dealer_code per calendar month
  Source  : int_tickets_enriched

  Purpose : Surface dealer-level support quality metrics over time. Answers:
              - Which dealers generate the most critical/warranty tickets?
              - Which dealers have the worst SLA compliance?
              - Are dealer resolution times improving or worsening by month?

  Use case: Quality engineering and dealer network management teams use this
            to identify underperforming dealers and target improvement actions.
*/

with tickets as (

    select * from {{ ref('int_tickets_enriched') }}

),

aggregated as (

    select
        -- ── Dimensions ────────────────────────────────────────────────────────
        dealer_code,
        dealer_label,
        dealer_city,
        dealer_state,
        dealer_country,
        geo_region,
        created_month,
        created_year,
        created_month_num,

        -- ── Volume ────────────────────────────────────────────────────────────
        count(*)                                            as total_tickets,
        count(*) filter (where is_closed)                  as closed_tickets,
        count(*) filter (where is_open)                    as open_tickets,
        count(*) filter (where is_critical)                as critical_tickets,
        count(*) filter (where is_warranty_claim)          as warranty_claims,
        count(*) filter (where is_sla_breach = true)       as sla_breached_tickets,
        count(distinct vehicle_chassis_id)                 as distinct_vehicles,

        -- ── SLA breach rate ───────────────────────────────────────────────────
        round(
            100.0
            * count(*) filter (where is_sla_breach = true)
            / nullif(count(*) filter (where is_closed), 0),
            2
        )                                                   as sla_breach_rate_pct,

        -- ── Resolution time ───────────────────────────────────────────────────
        round(avg(resolution_hours)    filter (where is_closed), 2) as avg_resolution_hours,
        round(median(resolution_hours) filter (where is_closed), 2) as median_resolution_hours,
        round(max(resolution_hours)    filter (where is_closed), 2) as max_resolution_hours,

        -- ── Queue time ────────────────────────────────────────────────────────
        round(avg(first_queue_hours), 2)                    as avg_first_queue_hours,

        -- ── Warranty rate ─────────────────────────────────────────────────────
        round(
            100.0
            * count(*) filter (where is_warranty_claim)
            / nullif(count(*), 0),
            2
        )                                                   as warranty_rate_pct

    from tickets
    group by
        dealer_code,
        dealer_label,
        dealer_city,
        dealer_state,
        dealer_country,
        geo_region,
        created_month,
        created_year,
        created_month_num

)

select * from aggregated
