# Support Analytics — dbt + DuckDB

A production-grade analytics engineering project built on top of the
[Support Ticket Intelligence](../README.md) dataset. Demonstrates layered data
modeling, data quality testing, and self-documenting SQL using dbt Core and DuckDB.

---

## Business Questions Answered

| Question | Mart model |
|---|---|
| Which teams breach SLA most often, and in which months? | `mart_team_workload` |
| Which dealers have the slowest resolution times? | `mart_dealer_performance` |
| What product families generate the most warranty claims? | `mart_product_defects` |
| What is the monthly ticket volume trend by severity? | `mart_ticket_kpis` |
| How does resolution time distribute across severity levels? | `mart_ticket_kpis` |

---

## Architecture

```
  seeds/support_tickets.csv   ← 50,000 rows, 38 columns
  (simulates Fivetran/Airbyte raw load)
              │
              ▼
  ┌───────────────────────────────────────────┐
  │  staging.stg_support_tickets              │
  │  • Rename columns to snake_case           │
  │  • Explicit casts for all types           │
  │  • Normalize odometer to km               │
  │  Materialized: VIEW                       │
  └─────────────────────┬─────────────────────┘
                        │
                        ▼
  ┌───────────────────────────────────────────┐
  │  intermediate.int_tickets_enriched        │
  │  • SLA breach flag (per severity tier)    │
  │  • Resolution buckets (0–4h … 7d+)        │
  │  • Calendar keys (month, quarter, DOW)    │
  │  • Boolean flags (is_closed, is_warranty) │
  │  Materialized: VIEW                       │
  └──────┬────────────────────┬───────────────┘
         │                    │
         ▼                    ▼
  ┌─────────────┐   ┌──────────────────────┐
  │mart_ticket_ │   │mart_dealer_           │
  │kpis         │   │performance            │
  │(grain:      │   │(grain: dealer×month)  │
  │ ticket)     │   └──────────────────────┘
  └─────────────┘
         │
         ▼
  ┌─────────────────────┐   ┌──────────────────────┐
  │mart_team_workload   │   │mart_product_defects   │
  │(grain: team×month)  │   │(grain: product×fault) │
  └─────────────────────┘   └──────────────────────┘
```

---

## Project Structure

```
6-dbt-analytics/
├── dbt_project.yml          # Project config, materialization strategy
├── profiles.yml             # DuckDB connection (local dev)
├── packages.yml             # dbt-utils, dbt-expectations
├── seeds/
│   ├── support_tickets.csv  # Raw data (50k rows)
│   └── schema.yml           # Seed documentation + source tests
├── models/
│   ├── staging/             # Rename, cast, normalize — no business logic
│   ├── intermediate/        # Business logic + derived fields
│   └── marts/               # Aggregated, BI-ready tables
├── tests/                   # Custom singular tests
├── macros/
│   └── generate_schema_name.sql  # Override schema naming convention
└── analyses/                # Ad-hoc SQL (not materialized)
```

---

## How to Run

### 1. Environment setup

```bash
cd 6-dbt-analytics

# Create and activate virtual environment
python -m venv .venv
source .venv/Scripts/activate   # Windows bash
# source .venv/bin/activate     # Mac/Linux

# Install dbt with DuckDB adapter
pip install dbt-duckdb

# Install dbt packages (dbt-utils, dbt-expectations)
dbt deps
```

### 2. Load raw data and build all models

```bash
# Load CSV into DuckDB as the raw layer
dbt seed

# Run all models (staging → intermediate → marts)
dbt run

# Execute all tests
dbt test

# Build = seed + run + test in one command
dbt build
```

### 3. Generate and view documentation

```bash
dbt docs generate
dbt docs serve   # opens browser at localhost:8080
```

### 4. Inspect the DuckDB database directly

```bash
# Requires: pip install duckdb
python -c "
import duckdb
con = duckdb.connect('dev.duckdb')
con.execute(\"SHOW ALL TABLES\").fetchdf()
"
```

---

## Data Quality

Every model has schema-level tests. Key invariants:

- `ticket_id` is unique and not null across all layers
- `ticket_status` only contains `['Open', 'Closed']`
- `severity_level` only contains `['Low', 'Medium', 'High', 'Critical']`
- `resolution_seconds` is never negative
- Closed tickets always have a `closed_date`

---

## Tech Stack

| Tool | Role |
|---|---|
| dbt Core | Transformation framework |
| DuckDB | Local analytical warehouse |
| dbt-utils | Generic tests + utility macros |
| dbt-expectations | Statistical data quality tests |
| GitHub Pages | Hosts `dbt docs` artifact |
