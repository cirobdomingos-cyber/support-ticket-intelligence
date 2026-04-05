# Support Ticket Intelligence — Claude Code Guidelines

## Quem é o dono deste projeto
Ciro Beduschi Domingos — Senior Data Analyst, 15 anos Volvo Group, Python/SQL/Power BI/Databricks.
Objetivo: portfólio para posições remote de Analytics Engineer / Senior Data Analyst ($80k–$130k USD).
Este projeto precisa parecer trabalho de produção real, não tutorial.

## O que é este projeto
End-to-end AI prototype que automatiza triagem de tickets de suporte técnico:
- Roteamento automático de tickets via ML (>99% accuracy)
- Busca semântica em casos históricos (FAISS + SentenceTransformers)
- Sugestão de resposta via LLM (HuggingFace Mistral — free tier)
- Dashboard de KPIs operacionais (Streamlit)

**Live demo:** lavish-connection-production-5385.up.railway.app
**API docs:** support-ticket-intelligence-production-795d.up.railway.app/docs

## Estrutura do projeto
```
1-support-ticket-dataset/     # Gerador de dados sintéticos (~50k tickets, 38 colunas)
2-support-ticket-routing-ml/  # Treino do modelo TF-IDF + LogisticRegression
3-support-ticket-semantic-search/ # Embeddings + FAISS index
4-support-ticket-api/         # FastAPI — serve tudo via HTTP
5-support-ticket-dashboard/   # Streamlit — UI completa
```

## Tech stack
- **Linguagem:** Python 3.12 (NÃO usar Python 3.14 — muitos pacotes sem wheel)
- **API:** FastAPI + Uvicorn
- **Dashboard:** Streamlit
- **ML:** scikit-learn (TF-IDF + LogReg), SentenceTransformers, FAISS
- **LLM:** HuggingFace Inference API (Qwen/Qwen2.5-7B-Instruct-1M por padrão)
- **Deploy:** Railway (2 serviços separados: API e Dashboard)
- **CI:** GitHub Actions

## Como rodar localmente (Windows PowerShell)

### API (porta 8000)
```powershell
cd 4-support-ticket-api
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m uvicorn main:app --reload --port 8000
```

### Dashboard (porta 8501)
```powershell
cd 5-support-ticket-dashboard
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:API_URL = "http://localhost:8000"
.venv\Scripts\python.exe -m streamlit run app.py
```

## Variáveis de ambiente
- `API_URL` — URL da API (default: http://localhost:8000)
- `HUGGINGFACEHUB_API_TOKEN` — token HuggingFace para AI suggestions (opcional; sem token usa fallback local)
- `HUGGINGFACE_REPO_ID` — modelo LLM (default: Qwen/Qwen2.5-7B-Instruct-1M)
- `DUCKDB_PATH` — caminho do arquivo DuckDB para analytics (módulo 6)

## Deploy (Railway)
- 2 serviços: `4-support-ticket-api` e `5-support-ticket-dashboard`
- Cada um tem seu próprio `Dockerfile` e `railway.json`
- API tem healthcheck em `/health`
- Push para `main` = deploy automático em produção
- Existe environment "dev" no Railway para testar antes de ir pra produção

## Fluxo de trabalho com branches
- `main` = produção (Railway deploy automático)
- Nunca commitar direto na main sem PR
- Usar worktrees: `claude --worktree dev/nome-da-tarefa`
- Branch de desenvolvimento mapeia para Railway environment "dev"

## Próximo módulo planejado: dbt + DuckDB (módulo 6)
Analytics layer sobre os dados de tickets:
- dbt Core + DuckDB adapter, Python 3.12
- Seeds: support_tickets.csv (50k rows, 38 cols)
- Modelos: staging → intermediate (SLA breach logic) → marts
- Marts: ticket_kpis, team_workload, dealer_performance, product_defects
- Dashboard KPI page lê de marts DuckDB (substitui pandas transforms em runtime)
- Deploy: multi-stage Dockerfile — dbt-builder stage produz dev.duckdb, dashboard bake no image
- **Sempre desenvolver em branch separada, validar em Railway dev, só então PR para main**

## Decisões já tomadas (não reverter sem motivo)
- Python 3.12 para venvs locais (3.14 quebra pandas/pydantic sem Visual Studio instalado)
- `python -m <module>` em vez de chamar .exe diretamente (mais confiável no Windows)
- LLM via HuggingFace free tier (zero custo, funciona em Railway sem config extra)
- DuckDB local (zero infra) para analytics layer em vez de Postgres
- `requirements.txt` com `>=` para pacotes com extensões nativas (evita build failures)

## Padrões de código
- Commits em inglês, mensagem explica o "porquê" não o "o quê"
- PRs sempre a partir de branch
- Sem push direto na main
- Scripts PowerShell usam `py -3.12` explicitamente

## O que este projeto demonstra (contexto de entrevista)
- End-to-end ML pipeline: data → model → API → UI
- Analytics engineering com dbt layered modeling
- Deploy containerizado em cloud real (Railway, não só localhost)
- Separação de concerns: API / UI / analytics em camadas independentes
- Qualidade de produção: testes, CI, documentação, sem TODOs
