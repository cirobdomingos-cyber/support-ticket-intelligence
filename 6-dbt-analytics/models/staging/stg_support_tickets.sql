/*
  Model   : stg_support_tickets
  Grain   : One row per support ticket (ticket_id)
  Source  : seeds.raw.support_tickets (50,000 synthetic Volvo-style tickets)

  Purpose : Single point of entry for all raw ticket data. Renames every
            column to a consistent snake_case convention, casts all types
            explicitly, and normalises odometer readings to km.

  Rules   :
    - No business logic lives here — only rename, cast, and normalise.
    - All downstream models MUST reference this model, never the seed directly.
    - Column aliases intentionally avoid the API's internal naming convention
      (e.g. 'severity_level' not 'severity') — dbt owns its own vocabulary.
*/

with source as (

    select * from {{ ref('support_tickets') }}

),

renamed as (

    select
        -- ── Primary key ────────────────────────────────────────────────────────
        ticket_id,

        -- ── Vehicle identity ───────────────────────────────────────────────────
        vehicle_chassis                             as vehicle_chassis_id,
        vehicle_vin,
        chassis_series,

        -- ── Product & component ────────────────────────────────────────────────
        product_code,
        product_family,
        component_name,
        fault_code,
        fault_mode,

        -- ── Ticket classification ──────────────────────────────────────────────
        ticket_status,
        ticket_substatus,
        status_state,
        severity_level,
        current_severity,
        customer_segment,
        source_channel,
        service_request_area,
        service_request_type,
        assigned_team,
        geo_region,

        -- ── Dealer geography ───────────────────────────────────────────────────
        dealer_code,
        dealer_label,
        dealer_city_name                            as dealer_city,
        dealer_state_code                           as dealer_state,
        dealer_country_code                         as dealer_country,

        -- ── Handler & reporter ─────────────────────────────────────────────────
        handler_country,
        handler_department,
        reporter_country,
        reporter_department,

        -- ── Free-text description (kept for reference, not transformed) ─────────
        description,

        -- ── Odometer — normalise to km ─────────────────────────────────────────
        -- Raw data contains mixed units (km / miles). Convert miles → km so
        -- all downstream models work with a single numeric scale.
        case
            when lower(odometer_unit) = 'miles'
                then cast(odometer_reading as double) * 1.60934
            else cast(odometer_reading as double)
        end                                         as odometer_km,

        -- ── Timestamps & dates ─────────────────────────────────────────────────
        cast(created_timestamp as timestamp)        as created_at,
        cast(created_date      as date)             as created_date,
        cast(closed_timestamp  as timestamp)        as closed_at,
        cast(closed_date       as date)             as closed_date,

        -- ── Numeric metrics ────────────────────────────────────────────────────
        cast(first_queue_seconds as integer)        as first_queue_seconds,
        cast(resolution_seconds  as double)         as resolution_seconds

    from source

)

select * from renamed
