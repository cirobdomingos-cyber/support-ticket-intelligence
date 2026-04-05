/*
  Model   : mart_ticket_kpis
  Grain   : One row per support ticket (ticket_id)
  Source  : int_tickets_enriched

  Purpose : Wide fact table for ticket-level drill-through and dashboard
            filtering. Contains every classification, timing, and SLA field
            in a single denormalised table. BI tools query this for raw
            record browsing, pivot tables, and dynamic filters.

  Note    : This is intentionally wide — it trades normalisation for
            query simplicity. Aggregation marts (dealer, team, product)
            exist for pre-computed rollups.
*/

select
    -- ── Keys ──────────────────────────────────────────────────────────────────
    ticket_id,
    vehicle_chassis_id,
    vehicle_vin,
    chassis_series,

    -- ── Product & component ───────────────────────────────────────────────────
    product_family,
    product_code,
    component_name,
    fault_code,
    fault_mode,

    -- ── Ticket metadata ───────────────────────────────────────────────────────
    severity_level,
    ticket_status,
    ticket_substatus,
    service_request_type,
    service_request_area,
    assigned_team,
    source_channel,
    customer_segment,
    geo_region,

    -- ── Geography ─────────────────────────────────────────────────────────────
    dealer_code,
    dealer_label,
    dealer_city,
    dealer_state,
    dealer_country,
    handler_country,
    handler_department,
    reporter_country,
    reporter_department,

    -- ── Dates ─────────────────────────────────────────────────────────────────
    created_at,
    created_date,
    closed_at,
    closed_date,
    created_month,
    created_year,
    created_quarter,

    -- ── Metrics ───────────────────────────────────────────────────────────────
    first_queue_seconds,
    first_queue_hours,
    resolution_seconds,
    resolution_hours,
    resolution_days,
    days_open,
    odometer_km,

    -- ── Flags & buckets ───────────────────────────────────────────────────────
    is_closed,
    is_open,
    is_warranty_claim,
    is_critical,
    is_high_severity,
    is_sla_breach,
    resolution_bucket

from {{ ref('int_tickets_enriched') }}
