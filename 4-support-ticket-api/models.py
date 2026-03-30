from __future__ import annotations

from typing import List

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


class DatasetStatus(BaseModel):
    exists: bool = Field(..., description="Whether the dataset file exists")
    row_count: int = Field(..., ge=0, description="Number of rows in the dataset")
    path: str = Field(..., description="Resolved dataset path")


class ModelsStatus(BaseModel):
    loaded: bool = Field(..., description="Whether routing models are loaded")
    available: list[str] = Field(..., description="Available routing model artifact names")


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


class GenerateDatasetRequest(BaseModel):
    include_columns: list[str] = Field(
        default_factory=list,
        description="Optional list of output columns (internal or public aliases)",
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


class HealthResponse(BaseModel):
    status: str = Field(..., description="Health status")
    models_loaded: bool = Field(..., description="Whether models were successfully loaded")
