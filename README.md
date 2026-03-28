# Support Ticket Intelligence

This monorepo contains an end-to-end support ticket intelligence ecosystem built around synthetic ticket generation, automated routing, and semantic search.

## Architecture

- `support-ticket-dataset/`
  - Generates a synthetic support ticket dataset.
  - Includes ticket metadata, descriptions, lifecycle timestamps, and routing labels.

- `support-ticket-routing-ml/`
  - Implements machine learning baselines for automatic ticket routing.
  - Includes text classification and metadata-driven routing models.

- `support-ticket-semantic-search/`
  - Implements semantic similarity search for support tickets.
  - Builds embeddings and a FAISS index over ticket descriptions.

## Purpose

This repository is designed as a single GitHub project containing the full support ticket intelligence stack. The root README gives the architecture and high-level flow, while each subfolder has its own detailed README and code.

## Quick start

1. Clone the repository.
2. Inspect each subproject:
   - `support-ticket-dataset/`
   - `support-ticket-routing-ml/`
   - `support-ticket-semantic-search/`
3. Follow the individual README instructions in each folder.

## Workflow

1. Generate or inspect data in `support-ticket-dataset/`.
2. Train and evaluate routing models in `support-ticket-routing-ml/`.
3. Build the semantic search index in `support-ticket-semantic-search/`.

## Notes

This repository is intended to be pushed to a new GitHub repo named `support-ticket-intelligence` so the full ecosystem is available from one remote link.
