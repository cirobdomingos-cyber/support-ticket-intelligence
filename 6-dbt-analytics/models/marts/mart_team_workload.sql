/*
  Model   : mart_team_workload
  Grain   : One row per assigned_team per calendar month
  Source  : int_tickets_enriched

  Purpose : Pre-aggregate team-level performance metrics by month. Answers:
              - Which teams are breaching SLA and in which months?
              - Is resolution speed improving or degrading over time?
              - Where is the critical-ticket workload concentrated?

  Design  : Aggregating by month (not day) keeps the table manageable and
            matches the natural reporting cadence for team performance reviews.
            The weighted avg_resolution_hours calculation uses closed_tickets
            as the weight to avoid skew from months with few closures.
*/

with tickets as (

    select * from {{ ref('int_tickets_enriched') }}

),

aggregated as (

    select
        -- ── Dimensions ────────────────────────────────────────────────────────
        assigned_team,
        created_month,
        created_year,
        created_month_num,
        created_quarter,

        -- ── Volume ────────────────────────────────────────────────────────────
        count(*)                                            as total_tickets,
        count(*) filter (where is_closed)                  as closed_tickets,
        count(*) filter (where is_open)                    as open_tickets,
        count(*) filter (where is_critical)                as critical_tickets,
        count(*) filter (where is_high_severity)           as high_severity_tickets,
        count(*) filter (where is_warranty_claim)          as warranty_claims,
        count(*) filter (where is_sla_breach = true)       as sla_breached_tickets,

        -- ── SLA breach rate (closed tickets only) ─────────────────────────────
        round(
            100.0
            * count(*) filter (where is_sla_breach = true)
            / nullif(count(*) filter (where is_closed), 0),
            2
        )                                                   as sla_breach_rate_pct,

        -- ── Resolution time (closed tickets only) ─────────────────────────────
        round(avg(resolution_hours)    filter (where is_closed), 2) as avg_resolution_hours,
        round(median(resolution_hours) filter (where is_closed), 2) as median_resolution_hours,
        round(min(resolution_hours)    filter (where is_closed), 2) as min_resolution_hours,
        round(max(resolution_hours)    filter (where is_closed), 2) as max_resolution_hours,

        -- ── Queue time ────────────────────────────────────────────────────────
        round(avg(first_queue_hours), 2)                    as avg_first_queue_hours,

        -- ── Workload mix (% of total within team-month) ───────────────────────
        round(
            100.0
            * count(*) filter (where is_warranty_claim)
            / nullif(count(*), 0),
            2
        )                                                   as warranty_pct,

        round(
            100.0
            * count(*) filter (where is_critical)
            / nullif(count(*), 0),
            2
        )                                                   as critical_pct

    from tickets
    group by
        assigned_team,
        created_month,
        created_year,
        created_month_num,
        created_quarter

)

select * from aggregated
