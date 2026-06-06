"""
X-RAG · Configuration loader (single source of truth).

USAGE (everywhere in the code):

    from config.settings import settings
    print(settings.generator.model_id)

Override anything from the environment, e.g.:

    XRAG_GENERATOR__TEMPERATURE=0.0 python -m app.api.main

The `__` separator turns nested keys into env variables (pydantic-settings convention).
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# ---------------------------------------------------------------------------
# Sub-section models (one per logical group in config.yaml)
# ---------------------------------------------------------------------------
class BudgetCfg(BaseModel):
    total_sec: float = 25.0


class ApiCfg(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    rate_limit_per_min: int = 60


class SafetyCfg(BaseModel):
    enabled: bool = True
    guard_model_id: str = "meta-llama/Llama-Guard-3-1B"
    fallback_rule_based: bool = True
    output_side_check: bool = True


class RewriterCfg(BaseModel):
    model_id: str = "unsloth/Llama-3.2-3B-Instruct"
    adapter_path: Optional[str] = None
    max_sub_queries: int = 3


class SearchCfg(BaseModel):
    provider: Literal["ddgs", "searxng"] = "ddgs"
    searxng_url: str = "http://localhost:8080"
    k_per_query: int = 10


class ScrapeCfg(BaseModel):
    concurrency: int = 8
    timeout_sec: float = 10.0
    user_agent: str = "xrag-bot/0.1"
    cache_ttl_sec: int = 600


class ChunkingCfg(BaseModel):
    size_tokens: int = 612
    overlap_ratio: float = 0.15
    link_density_drop: float = 0.55
    text_density_floor: float = 25.0


class EmbeddingCfg(BaseModel):
    model_id: str = "BAAI/bge-m3"
    top_n_candidates: int = 50
    rrf_k: int = 60


class RerankingCfg(BaseModel):
    model_id: str = "BAAI/bge-reranker-v2-m3"
    top_k_keep: int = 10
    # Relevance gate: drop any reranked chunk whose sigmoid(CE) relevance is below
    # this floor BEFORE it reaches the generator, so off-topic pages can't pollute
    # the context. 0.0 disables the gate; 0.5 keeps only chunks the cross-encoder
    # scores as net-relevant (CE > 0).
    min_relevance: float = 0.0


class GeneratorCfg(BaseModel):
    model_id: str = "NousResearch/Meta-Llama-3.1-8B-Instruct"
    adapter_path: Optional[str] = None
    temperature: float = 0.2
    max_new_tokens: int = 1024
    max_input_tokens: int = 4096                  # prompt token budget


class AttributionCfg(BaseModel):
    model_id: str = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
    head_adapter_path: Optional[str] = None
    threshold: float = 0.75
    formula: Literal["product", "min", "geomean"] = "product"
    ce_normalize: Literal["sigmoid", "minmax"] = "sigmoid"


class OrchestratorCfg(BaseModel):
    # Corrective / iterative RAG: if the answer has no green (supported) claim,
    # re-rewrite (8B few-shot rewriter, different sub-queries) + re-search, up to
    # max_attempts. Runs the engine.run path through the LangGraph state machine.
    corrective_rag: bool = False
    max_attempts: int = 3


class StorageCfg(BaseModel):
    redis_url: str = "redis://localhost:6379/0"
    qdrant_path: str = "./data/qdrant"
    faiss_index_type: Literal["flat", "hnsw"] = "flat"


class PathsCfg(BaseModel):
    data_dir: str = "./data"
    models_dir: str = "./models"


# ---------------------------------------------------------------------------
# Top-level Settings (loads config.yaml, allows env overrides)
# ---------------------------------------------------------------------------
class Settings(BaseSettings):
    """Top-level settings; loads from config.yaml and env overrides."""

    project_name: str = "xrag"
    mode: Literal["live_web", "persistent_kb"] = "live_web"

    budget: BudgetCfg = Field(default_factory=BudgetCfg)
    api: ApiCfg = Field(default_factory=ApiCfg)
    safety: SafetyCfg = Field(default_factory=SafetyCfg)
    rewriter: RewriterCfg = Field(default_factory=RewriterCfg)
    search: SearchCfg = Field(default_factory=SearchCfg)
    scrape: ScrapeCfg = Field(default_factory=ScrapeCfg)
    chunking: ChunkingCfg = Field(default_factory=ChunkingCfg)
    embedding: EmbeddingCfg = Field(default_factory=EmbeddingCfg)
    reranking: RerankingCfg = Field(default_factory=RerankingCfg)
    generator: GeneratorCfg = Field(default_factory=GeneratorCfg)
    attribution: AttributionCfg = Field(default_factory=AttributionCfg)
    orchestrator: OrchestratorCfg = Field(default_factory=OrchestratorCfg)
    storage: StorageCfg = Field(default_factory=StorageCfg)
    paths: PathsCfg = Field(default_factory=PathsCfg)

    model_config = SettingsConfigDict(
        env_prefix="XRAG_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_settings(yaml_path: str | Path = "config/config.yaml") -> Settings:
    """Load settings: yaml file first, then env-var overrides on top."""
    data = _load_yaml(Path(yaml_path))
    return Settings(**data)


# Module-level singleton — import this everywhere.
settings: Settings = load_settings()


if __name__ == "__main__":
    # `python config/settings.py` prints the effective config — handy for debugging.
    import json
    print(json.dumps(settings.model_dump(), indent=2))
