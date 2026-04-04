FROM python:3.11-slim

WORKDIR /analytics

# Install dbt with DuckDB adapter — pinned for reproducibility
RUN pip install --no-cache-dir dbt-duckdb==1.8.1

# Copy project files (seed CSV is large — copy last to exploit layer caching)
COPY packages.yml      packages.yml
COPY dbt_project.yml   dbt_project.yml
COPY profiles.yml      profiles.yml
COPY macros/           macros/
COPY models/           models/
COPY tests/            tests/
COPY analyses/         analyses/
COPY seeds/schema.yml  seeds/schema.yml
COPY seeds/support_tickets.csv seeds/support_tickets.csv

# Install dbt packages (dbt-utils, dbt-expectations)
RUN dbt deps --profiles-dir .

# Build: seed → run → test. Produces dev.duckdb in /analytics/
RUN dbt build --profiles-dir .

# The output artifact is /analytics/dev.duckdb
# In multi-stage builds, copy it with:
#   COPY --from=dbt-builder /analytics/dev.duckdb /app/dev.duckdb
