-- Singular test: no closed ticket should have a negative resolution time.
-- A negative value would indicate a data pipeline error (closed_at < created_at).
-- This test returns rows that FAIL the assertion — dbt marks the run as failed
-- if any rows are returned.

select
    ticket_id,
    created_at,
    closed_at,
    resolution_seconds
from {{ ref('stg_support_tickets') }}
where
    ticket_status = 'Closed'
    and resolution_seconds is not null
    and resolution_seconds < 0
