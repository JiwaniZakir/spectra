"""pgvector (PostgreSQL) vector store backend.

Uses the pgvector extension for PostgreSQL to store and search dense
vectors alongside relational metadata.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray
from pydantic import Field

from spectra.vectorstores.base import BaseVectorStore, SearchResult, VectorStoreConfig

logger = logging.getLogger(__name__)


class PgVectorConfig(VectorStoreConfig):
    """Configuration for the pgvector store."""

    connection_string: str = "postgresql://localhost:5432/spectra"
    table_name: str = "spectra_vectors"
    create_extension: bool = True
    create_table: bool = True
    index_type: str = Field(
        default="ivfflat",
        pattern="^(ivfflat|hnsw|none)$",
        description="Index type: ivfflat, hnsw, or none.",
    )
    ivfflat_lists: int = Field(default=100, ge=1)
    hnsw_m: int = Field(default=16, ge=4)
    hnsw_ef_construction: int = Field(default=64, ge=16)


class PgVectorStore(BaseVectorStore):
    """PostgreSQL + pgvector backed vector store.

    Stores vectors in a PostgreSQL table with the pgvector extension.
    Supports IVFFlat and HNSW indexing, cosine/L2/inner-product distance,
    and metadata filtering via JSON columns.

    Parameters
    ----------
    config:
        pgvector store configuration.

    Note
    ----
    Requires ``psycopg`` and ``pgvector``:
    ``pip install "psycopg[binary]" pgvector``.
    """

    def __init__(self, config: PgVectorConfig | None = None) -> None:
        self.config: PgVectorConfig = config or PgVectorConfig()
        super().__init__(self.config)
        self._conn: Any = None
        self._connect()

    def _connect(self) -> None:
        """Establish connection and set up tables."""
        try:
            import psycopg
        except ImportError as exc:
            raise ImportError(
                "psycopg is required. Install with: pip install 'psycopg[binary]'"
            ) from exc

        self._conn = psycopg.connect(self.config.connection_string, autocommit=True)

        if self.config.create_extension:
            self._conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

        if self.config.create_table:
            self._conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.config.table_name} (
                    id TEXT PRIMARY KEY,
                    embedding vector({self.config.dimension}),
                    metadata JSONB DEFAULT '{{}}'::jsonb
                )
            """)

            # Create index
            if self.config.index_type == "ivfflat":
                op_class = self._operator_class()
                self._conn.execute(f"""
                    CREATE INDEX IF NOT EXISTS {self.config.table_name}_embedding_idx
                    ON {self.config.table_name}
                    USING ivfflat (embedding {op_class})
                    WITH (lists = {self.config.ivfflat_lists})
                """)
            elif self.config.index_type == "hnsw":
                op_class = self._operator_class()
                self._conn.execute(f"""
                    CREATE INDEX IF NOT EXISTS {self.config.table_name}_embedding_idx
                    ON {self.config.table_name}
                    USING hnsw (embedding {op_class})
                    WITH (m = {self.config.hnsw_m}, ef_construction = {self.config.hnsw_ef_construction})
                """)

    def _operator_class(self) -> str:
        """Return the pgvector operator class for the configured metric."""
        return {
            "cosine": "vector_cosine_ops",
            "l2": "vector_l2_ops",
            "dot": "vector_ip_ops",
        }.get(self.config.metric, "vector_cosine_ops")

    def _distance_operator(self) -> str:
        """Return the pgvector distance operator."""
        return {
            "cosine": "<=>",
            "l2": "<->",
            "dot": "<#>",
        }.get(self.config.metric, "<=>")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(
        self,
        ids: Sequence[str],
        vectors: NDArray[np.float32],
        metadatas: Sequence[dict[str, Any]] | None = None,
    ) -> None:
        """Insert vectors into the PostgreSQL table."""
        table = self.config.table_name
        for i, doc_id in enumerate(ids):
            vec_list = vectors[i].tolist()
            meta = metadatas[i] if metadatas and i < len(metadatas) else {}
            meta_json = json.dumps(meta)

            self._conn.execute(
                f"""
                INSERT INTO {table} (id, embedding, metadata)
                VALUES (%s, %s::vector, %s::jsonb)
                ON CONFLICT (id) DO UPDATE SET
                    embedding = EXCLUDED.embedding,
                    metadata = EXCLUDED.metadata
                """,
                (doc_id, str(vec_list), meta_json),
            )

    def search(
        self,
        query_vector: NDArray[np.float32],
        top_k: int = 10,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Search for nearest neighbours using pgvector."""
        table = self.config.table_name
        op = self._distance_operator()
        vec_str = str(query_vector.tolist())

        where_clause = ""
        params: list[Any] = [vec_str]

        if filter_metadata:
            conditions = []
            for key, value in filter_metadata.items():
                conditions.append(f"metadata->>'{key}' = %s")
                params.append(str(value))
            where_clause = "WHERE " + " AND ".join(conditions)

        params.append(top_k)
        query = f"""
            SELECT id, embedding {op} %s::vector AS distance, metadata
            FROM {table}
            {where_clause}
            ORDER BY embedding {op} %s::vector
            LIMIT %s
        """
        # For ORDER BY we need the vector again
        params_full = [vec_str] + params[1:-1] + [vec_str, top_k]

        # Simplified query
        query = f"""
            SELECT id, embedding {op} %s::vector AS distance, metadata
            FROM {table}
            {where_clause}
            ORDER BY distance
            LIMIT %s
        """

        cursor = self._conn.execute(query, params)
        rows = cursor.fetchall()

        results: list[SearchResult] = []
        for row in rows:
            doc_id, distance, metadata = row
            # Convert distance to similarity score
            if self.config.metric == "cosine":
                score = 1.0 - float(distance)
            elif self.config.metric == "dot":
                score = -float(distance)  # pgvector returns negative IP
            else:
                score = 1.0 / (1.0 + float(distance))

            results.append(
                SearchResult(
                    id=doc_id,
                    score=score,
                    metadata=metadata if isinstance(metadata, dict) else {},
                )
            )

        return results

    def delete(self, ids: Sequence[str]) -> None:
        """Delete vectors by ID."""
        if not ids:
            return
        placeholders = ",".join(["%s"] * len(ids))
        self._conn.execute(
            f"DELETE FROM {self.config.table_name} WHERE id IN ({placeholders})",
            list(ids),
        )

    def count(self) -> int:
        """Return the number of vectors in the table."""
        cursor = self._conn.execute(f"SELECT COUNT(*) FROM {self.config.table_name}")
        row = cursor.fetchone()
        return int(row[0]) if row else 0

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
