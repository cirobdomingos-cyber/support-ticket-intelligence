from __future__ import annotations

from contextlib import asynccontextmanager
from io import BytesIO

import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware

import services
from models import (
    BuildIndexResponse,
    GenerateDatasetRequest,
    GenerateDatasetResponse,
    HealthResponse,
    RouteRequest,
    RouteResponse,
    SearchRequest,
    SearchResult,
    StatusResponse,
    TrainResponse,
    UploadDatasetResponse,
)


def _normalize_dataset_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        col: services.PUBLIC_TO_INTERNAL_COLUMNS[col]
        for col in df.columns
        if col in services.PUBLIC_TO_INTERNAL_COLUMNS
    }
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.models_loaded = False
    app.state.routing_models_loaded = False
    app.state.semantic_search_loaded = False

    try:
        services.load_models()
        state = services.get_model_status()
        app.state.models_loaded = state["all_loaded"]
        app.state.routing_models_loaded = state["routing_loaded"]
        app.state.semantic_search_loaded = state["semantic_loaded"]
    except Exception as exc:
        # The app will still start, but routes will return 503 until models are fixed.
        print(f"[startup] model loading failed: {exc}")
        app.state.models_loaded = False
        state = services.get_model_status()
        app.state.routing_models_loaded = state["routing_loaded"]
        app.state.semantic_search_loaded = state["semantic_loaded"]
    yield


app = FastAPI(title="Support Ticket API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    return HealthResponse(status="ok", models_loaded=bool(app.state.models_loaded))


@app.post("/route", response_model=RouteResponse)
async def route_ticket(request: RouteRequest) -> RouteResponse:
    if not app.state.routing_models_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Routing models are not loaded yet",
        )
    try:
        assigned_team, confidence, all_scores = services.predict_route(request.description)
        return RouteResponse(
            assigned_team=assigned_team,
            confidence=confidence,
            all_scores=all_scores,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Routing prediction failed: {exc}",
        )


@app.post("/search", response_model=list[SearchResult])
async def search_tickets(request: SearchRequest) -> list[SearchResult]:
    if not app.state.semantic_search_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Semantic search resources are not loaded yet",
        )
    try:
        results = services.search_similar_tickets(request.description, request.top_k)
        return [SearchResult(**item) for item in results]
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {exc}",
        )


@app.get("/status", response_model=StatusResponse)
def status_check() -> StatusResponse:
    dataset_status = services.get_dataset_status()
    model_state = services.get_model_status()
    models_status = {
        "loaded": bool(model_state["routing_loaded"]),
        "available": services.get_routing_model_files(),
    }
    faiss_status = services.get_faiss_index_status()
    return StatusResponse(
        dataset=dataset_status,
        models=models_status,
        faiss_index=faiss_status,
    )


@app.get("/dataset")
def get_dataset(limit: int = Query(5000, ge=1, le=5000)) -> list[dict[str, object]]:
    try:
        return services.get_dataset_rows(limit=limit)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Dataset retrieval failed: {exc}",
        )


@app.post("/upload-dataset", response_model=UploadDatasetResponse)
async def upload_dataset(file: UploadFile = File(...)) -> UploadDatasetResponse:
    content = await file.read()
    if not content:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty.")

    try:
        df = pd.read_csv(BytesIO(content))
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid CSV upload: {exc}")

    df = _normalize_dataset_columns(df)
    required_columns = set(services.REQUIRED_INTERNAL_COLUMNS)
    missing = required_columns - set(df.columns)
    if missing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Missing required columns: {sorted(missing)}",
        )

    try:
        _, row_count, columns = services.save_dataset_file(content)
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Could not save dataset: {exc}")

    return UploadDatasetResponse(success=True, row_count=row_count, columns=columns)


@app.post("/generate-dataset", response_model=GenerateDatasetResponse)
def generate_dataset(request: GenerateDatasetRequest | None = None) -> GenerateDatasetResponse:
    try:
        include_columns = None if request is None else request.include_columns
        _, row_count = services.generate_synthetic_dataset(include_columns=include_columns)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Dataset generation failed: {exc}",
        )
    return GenerateDatasetResponse(success=True, row_count=row_count)


def _run_training_pipeline() -> TrainResponse:
    try:
        print("[train] Starting training pipeline...")
        dataset_path = services.get_dataset_path()
        dataset_generated = False
        if not dataset_path.exists():
            print("[train] No dataset found, generating synthetic dataset...")
            dataset_path, row_count = services.generate_synthetic_dataset()
            dataset_generated = True
            print(f"[train] Generated dataset with {row_count} rows at {dataset_path}")
        else:
            row_count = int(pd.read_csv(dataset_path).shape[0])
            print(f"[train] Using existing dataset with {row_count} rows at {dataset_path}")

        print("[train] Training routing models...")
        model_dir = services.train_routing_models(dataset_path=dataset_path)
        print(f"[train] Models saved to {model_dir}")

        print("[train] Building FAISS index...")
        vector_count = services.build_faiss_index()
        print(f"[train] FAISS index built with {vector_count} vectors")

        print("[train] Loading routing resources from disk...")
        services.load_routing_resources()
        print("[train] Routing resources loaded successfully")
        
        print("[train] Getting model status...")
        model_state = services.get_model_status()
        print(f"[train] Model state: {model_state}")
        
        print("[train] Updating app.state with loaded models...")
        app.state.routing_models_loaded = model_state["routing_loaded"]
        app.state.semantic_search_loaded = model_state["semantic_loaded"]
        app.state.models_loaded = model_state["all_loaded"]
        print(f"[train] App state updated: routing_loaded={app.state.routing_models_loaded}")

        response = TrainResponse(
            success=True,
            status=(
                "Training finished and models reloaded. "
                f"Routing models loaded: {model_state['routing_loaded']}."
            ),
            dataset_generated=dataset_generated,
            row_count=row_count,
            vector_count=vector_count,
            artifacts_path=str(model_dir.resolve()),
        )
        print(f"[train] Training complete, returning response: success={response.success}")
        return response
    except Exception as exc:
        print(f"[train] ERROR during training: {exc}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Training failed: {exc}",
        )


@app.post("/train", response_model=TrainResponse)
def train_models() -> TrainResponse:
    return _run_training_pipeline()


@app.post("/verify-models")
def verify_models() -> dict[str, object]:
    """Verify that models have actually been loaded into memory (useful for Railway timeout recovery)."""
    try:
        model_state = services.get_model_status()
        routing_loaded = model_state["routing_loaded"]
        semantic_loaded = model_state["semantic_loaded"]
        all_loaded = model_state["all_loaded"]
        
        print(f"[verify] Current model state: routing={routing_loaded} semantic={semantic_loaded} all={all_loaded}")
        
        # If models aren't loaded but files exist, try to load them now
        if not routing_loaded:
            print("[verify] Routing models not loaded, attempting to load...")
            try:
                services.load_routing_resources()
                model_state = services.get_model_status()
                routing_loaded = model_state["routing_loaded"]
                print(f"[verify] Routing load attempt result: {routing_loaded}")
            except Exception as e:
                print(f"[verify] Routing load failed: {e}")
        
        if not semantic_loaded:
            print("[verify] Semantic search not loaded, attempting to load...")
            try:
                services.load_semantic_search_resources()
                model_state = services.get_model_status()
                semantic_loaded = model_state["semantic_loaded"]
                print(f"[verify] Semantic load attempt result: {semantic_loaded}")
            except Exception as e:
                print(f"[verify] Semantic load failed: {e}")
        
        # Update app.state to reflect actual loaded state
        app.state.routing_models_loaded = routing_loaded
        app.state.semantic_search_loaded = semantic_loaded
        app.state.models_loaded = routing_loaded and semantic_loaded
        
        return {
            "routing_models_loaded": routing_loaded,
            "semantic_search_loaded": semantic_loaded,
            "all_models_loaded": routing_loaded and semantic_loaded,
        }
    except Exception as exc:
        print(f"[verify] ERROR: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Verification failed: {exc}",
        )


@app.post("/build-index", response_model=BuildIndexResponse)
def build_index() -> BuildIndexResponse:
    try:
        vector_count = services.build_faiss_index()
        services.SEMANTIC_SEARCH_LOADED = True
        model_state = services.get_model_status()
        app.state.routing_models_loaded = model_state["routing_loaded"]
        app.state.semantic_search_loaded = model_state["semantic_loaded"]
        app.state.models_loaded = model_state["all_loaded"]
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Index build failed: {exc}")
    return BuildIndexResponse(success=True, vector_count=vector_count)
