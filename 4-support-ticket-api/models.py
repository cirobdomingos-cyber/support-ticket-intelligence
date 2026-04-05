from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class RouteRequest(BaseModel):
    description: str = Field(..., description="Ticket description for routing")


class RouteResponse(BaseModel):
    assigned_team: str = Field(..., description="Predicted team assignment")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score for the assigned team")
    all_scores: dict[str, float] = Field(..., description="Probability scores for every possible team")


class SearchRequest(BaseModel):
    description: str = Field(..., description="Ticket description to search for similar tickets")
    top_k: int = Field(5, ge=1, description="Number of similar tickets to return")


class SearchResult(BaseModel):
    ticket_id: str = Field(..., description="Ticket identifier")
    description: str = Field(..., description="Ticket description")
    assigned_team: str = Field(..., description="Assigned team for the historical ticket")
    similarity_score: float = Field(..., ge=0.0, description="Similarity score from the FAISS search")


class SuggestRequest(BaseModel):
    description: str = Field(..., description="Ticket description for AI response suggestion")
    hf_token: str | None = Field(None, description="Optional HuggingFace API token; overrides the server env var for this request")


class SuggestResponse(BaseModel):
    suggested_response: str = Field(..., description="Suggested response for the support agent")
    context_tickets: list[SearchResult] = Field(
        default_factory=list,
        description="Top similar tickets used as context for generation",
    )
    llm_available: bool = Field(..., description="Whether a configured LLM provider was available")
    llm_error: str | None = Field(None, description="Error detail when LLM is unavailable or failed")


class DatasetStatus(BaseModel):
    exists: bool = Field(..., description="Whether the dataset file exists")
    row_count: int = Field(..., ge=0, description="Number of rows in the dataset")
    path: str = Field(..., description="Resolved dataset path")
    routing_capable: bool = Field(False, description="Whether the dataset has an assigned_team column enabling routing")
    dataset_name: str | None = Field(None, description="Optional human-friendly name for the active dataset")
    dataset_source: str | None = Field(None, description="How the active dataset was created (upload/synthetic)")


class ModelsStatus(BaseModel):
    loaded: bool = Field(..., description="Whether routing models are loaded")
    available: list[str] = Field(..., description="Available routing model artifact names")
    training_dataset: dict[str, Any] | None = Field(
        default=None,
        description="Metadata about the dataset used to train the current routing model",
    )


class FaissIndexStatus(BaseModel):
    exists: bool = Field(..., description="Whether a FAISS index is available")
    vector_count: int = Field(..., ge=0, description="Number of vectors in the FAISS index")


class StatusResponse(BaseModel):
    dataset: DatasetStatus = Field(..., description="Dataset module status")
    models: ModelsStatus = Field(..., description="Routing model status")
    faiss_index: FaissIndexStatus = Field(..., description="FAISS index status")


class UploadDatasetResponse(BaseModel):
    success: bool = Field(..., description="Whether dataset upload succeeded")
    row_count: int = Field(..., ge=0, description="Number of rows in the uploaded dataset")
    columns: list[str] = Field(..., description="Columns present in the uploaded dataset")


class GenerateDatasetResponse(BaseModel):
    success: bool = Field(..., description="Whether dataset generation succeeded")
    row_count: int = Field(..., ge=0, description="Number of rows generated")


class NamedSnapshotInfo(BaseModel):
    name: str = Field(..., description="Snapshot name (sanitized filename stem)")
    path: str = Field(..., description="Absolute path to the snapshot CSV")
    row_count: int = Field(..., ge=0, description="Number of rows in the snapshot")


class LoadSnapshotRequest(BaseModel):
    name: str = Field(..., description="Name of the snapshot to activate")


class LoadSnapshotResponse(BaseModel):
    success: bool = Field(..., description="Whether the snapshot was activated")
    row_count: int = Field(..., ge=0, description="Row count of the loaded dataset")


class GenerateDatasetRequest(BaseModel):
    size: int = Field(
        default=50000,
        ge=100,
        le=1000000,
        description="Number of rows to generate.",
    )
    include_columns: list[str] = Field(
        default_factory=list,
        description="Optional list of output columns (internal or public aliases)",
    )
    description_column: str | None = Field(
        default=None,
        description="Column to map to description for training. Null keeps the default synthetic mapping.",
    )
    assigned_team_column: str | None = Field(
        default=None,
        description="Column to map to assigned_team for routing. Empty string disables team mapping.",
    )
    ticket_id_column: str | None = Field(
        default=None,
        description="Column to map to ticket_id. Empty string auto-generates ticket ids.",
    )
    dataset_name: str | None = Field(
        default=None,
        description="Optional human-friendly name to label the generated synthetic dataset.",
    )


class BuildIndexResponse(BaseModel):
    success: bool = Field(..., description="Whether index build succeeded")
    vector_count: int = Field(..., ge=0, description="Number of vectors indexed")


class TrainResponse(BaseModel):
    success: bool = Field(..., description="Whether training and reload succeeded")
    status: str = Field(..., description="Human-readable training status")
    dataset_generated: bool = Field(..., description="Whether a dataset was auto-generated before training")
    row_count: int = Field(..., ge=0, description="Row count used for training")
    vector_count: int = Field(..., ge=0, description="Vector count in the rebuilt FAISS index")
    artifacts_path: str = Field(..., description="Path where routing model artifacts were written")


class SqlQueryRequest(BaseModel):
    sql: str = Field(..., description="SELECT query to execute against the tickets table")
    limit: int = Field(500, ge=1, le=5000, description="Max rows to return if LIMIT not in query")


class SqlQueryResponse(BaseModel):
    columns: list[str] = Field(..., description="Column names in result")
    rows: list[list] = Field(..., description="Result rows")
    row_count: int = Field(..., description="Number of rows returned")


class FeatureImportanceItem(BaseModel):
    word: str
    importance: float


class ModelPerformanceResponse(BaseModel):
    accuracy: float = Field(..., description="Hold-out accuracy of the routing model")
    class_names: list[str] = Field(..., description="Team class names")
    confusion_matrix: list[list[int]] = Field(..., description="Confusion matrix (rows=actual, cols=predicted)")
    feature_importance: list[FeatureImportanceItem] = Field(..., description="Top 30 TF-IDF features by importance")


class HealthResponse(BaseModel):
    status: str = Field(..., description="Health status")
    models_loaded: bool = Field(..., description="Whether models were successfully loaded")
