/*
  Model   : int_tickets_enriched
  Grain   : One row per support ticket (ticket_id)
  Source  : stg_support_tickets

  Purpose : Centralises all derived business logic that is shared across
            multiple mart models. Every derived flag, metric, and calendar
            key lives here so mart models never duplicate calculations.

  Key additions:
    - is_sla_breach    : whether the ticket violated its severity SLA threshold
    - resolution_hours : resolution_seconds converted to hours
    - resolution_bucket: human-readable duration bucket for distribution charts
    - created_month    : truncated to month for aggregation joins
    - days_open        : elapsed days (for open tickets, uses current_date)

  SLA thresholds come from dbt_project.yml vars.sla_hours — edit there to
  change them project-wide without touching SQL.

  Design note: intermediate models are materialised as views. They add zero
  storage cost and are always consistent with their source. If query performance
  becomes a concern, switching to `incremental` is a one-line config change.
*/

with tickets as (

    select * from {{ ref('stg_support_tickets') }}

),

enriched as (

    select
        -- ── Pass-through all staging columns ───────────────────────────────────
        *,

        -- ── Status flags ───────────────────────────────────────────────────────
        ticket_status = 'Closed'                        as is_closed,
        ticket_status != 'Closed'                       as is_open,
        lower(service_request_type) = 'warranty claim'  as is_warranty_claim,

        -- ── Severity flags ─────────────────────────────────────────────────────
        severity_level = 'Critical'                     as is_critical,
        severity_level in ('Critical', 'High')          as is_high_severity,

        -- ── Resolution time dimensions ─────────────────────────────────────────
        resolution_seconds / 3600.0                     as resolution_hours,
        resolution_seconds / 86400.0                    as resolution_days,
        first_queue_seconds / 3600.0                    as first_queue_hours,

        -- ── SLA breach ─────────────────────────────────────────────────────────
        -- For closed tickets: compare resolution_seconds to severity threshold.
        -- For open tickets:   compare elapsed time since creation to threshold.
        -- null when no creation timestamp exists (data quality guard).
        -- Thresholds are defined in dbt_project.yml vars.sla_hours.
        {%- set sla = var('sla_hours') %}
        case
            when created_at is null
                then null

            -- Closed tickets — use actual resolution time
            when ticket_status = 'Closed' and resolution_seconds is not null then
                case
                    when severity_level = 'Critical' and resolution_seconds > {{ sla.Critical }} * 3600  then true
                    when severity_level = 'High'     and resolution_seconds > {{ sla.High }}     * 3600  then true
                    when severity_level = 'Medium'   and resolution_seconds > {{ sla.Medium }}   * 3600  then true
                    when severity_level = 'Low'      and resolution_seconds > {{ sla.Low }}      * 3600  then true
                    else false
                end

            -- Open tickets — use elapsed time from creation to now
            when ticket_status != 'Closed' then
                case
                    when severity_level = 'Critical'
                        and epoch(current_timestamp - created_at) > {{ sla.Critical }} * 3600  then true
                    when severity_level = 'High'
                        and epoch(current_timestamp - created_at) > {{ sla.High }}     * 3600  then true
                    when severity_level = 'Medium'
                        and epoch(current_timestamp - created_at) > {{ sla.Medium }}   * 3600  then true
                    when severity_level = 'Low'
                        and epoch(current_timestamp - created_at) > {{ sla.Low }}      * 3600  then true
                    else false
                end

            else false
        end                                             as is_sla_breach,

        -- ── Resolution bucket ──────────────────────────────────────────────────
        case
            when resolution_seconds is null             then 'Open'
            when resolution_seconds <= 4  * 3600        then '0–4h'
            when resolution_seconds <= 24 * 3600        then '4–24h'
            when resolution_seconds <= 72 * 3600        then '24–72h'
            when resolution_seconds <= 168 * 3600       then '3–7 days'
            else '7+ days'
        end                                             as resolution_bucket,

        -- ── Calendar keys ──────────────────────────────────────────────────────
        date_trunc('month',   created_date)::date       as created_month,
        date_part('year',     created_date)::integer    as created_year,
        date_part('month',    created_date)::integer    as created_month_num,
        date_part('quarter',  created_date)::integer    as created_quarter,
        date_part('dow',      created_date)::integer    as created_day_of_week,

        -- ── Days open ──────────────────────────────────────────────────────────
        -- Closed tickets: difference between created and closed date.
        -- Open tickets:   difference between created date and today.
        datediff('day', created_date, coalesce(closed_date, current_date))
                                                        as days_open

    from tickets

)

select * from enriched
