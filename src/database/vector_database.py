"""
Vector Database Operations with pgvector Extension Support

This module provides vector similarity search capabilities using PostgreSQL's pgvector extension
for Story 3.0: Semantic Infrastructure Foundation.

Integrates with the existing ProductionDatabaseManager for database connections.
"""

import json
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np

try:
    import psycopg2
    from psycopg2.extras import Json
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False

from .production_database import get_database_manager, ProductionDatabaseManager


@dataclass
class SemanticTerm:
    """Data class for semantic terms with vector embeddings."""
    id: Optional[int] = None
    term: str = ""
    domain: str = ""  # 'spiritual', 'philosophical', 'scriptural', 'general'
    embedding: Optional[List[float]] = None
    transliteration: Optional[str] = None
    is_proper_noun: bool = False
    confidence_score: float = 1.0
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database operations."""
        return asdict(self)


@dataclass
class TermRelationship:
    """Data class for relationships between semantic terms."""
    id: Optional[int] = None
    source_term_id: int = 0
    target_term_id: int = 0
    relationship_type: str = ""  # 'synonym', 'related', 'variant', 'translation'
    strength: float = 0.0  # 0.0 to 1.0
    context: Optional[str] = None
    created_at: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database operations."""
        return asdict(self)


@dataclass
class SemanticContext:
    """Data class for semantic processing context."""
    text: str
    domain: str
    identified_terms: List[SemanticTerm]
    relationships: List[TermRelationship]
    confidence_score: float
    processing_metadata: Dict[str, Any]


class VectorDatabaseManager:
    """
    Manager for vector similarity operations using PostgreSQL pgvector extension.
    
    Provides semantic term storage, similarity search, and relationship management
    for the semantic infrastructure foundation.
    """

    def __init__(self, db_manager: Optional[ProductionDatabaseManager] = None):
        """Initialize with existing database manager."""
        self.db_manager = db_manager or get_database_manager()
        self.logger = logging.getLogger(__name__)
        
        # Check if pgvector extension is available
        self._pgvector_available = self._check_pgvector_extension()
        
        if not self._pgvector_available:
            self.logger.warning("pgvector extension not available - falling back to basic operations")

    def _check_pgvector_extension(self) -> bool:
        """Check if pgvector extension is installed and available."""
        try:
            if self.db_manager is None:
                self.logger.error("Error checking pgvector extension: 'NoneType' object has no attribute 'execute_read_query'")
                return False
            
            query = "SELECT 1 FROM pg_extension WHERE extname = 'vector'"
            result = self.db_manager.execute_read_query(query)
            
            if not result:
                # Try to create the extension
                create_query = "CREATE EXTENSION IF NOT EXISTS vector"
                self.db_manager.execute_write_query(create_query)
                
                # Check again
                result = self.db_manager.execute_read_query(query)
            
            return len(result) > 0
            
        except Exception as e:
            self.logger.error(f"Error checking pgvector extension: {e}")
            return False

    def initialize_schema(self) -> bool:
        """
        Initialize database schema for semantic terms and relationships.
        
        Creates the semantic_terms and term_relationships tables with vector support.
        """
        try:
            # Create semantic_terms table
            semantic_terms_query = """
            CREATE TABLE IF NOT EXISTS semantic_terms (
                id SERIAL PRIMARY KEY,
                term VARCHAR(255) NOT NULL,
                domain VARCHAR(50) NOT NULL DEFAULT 'general',
                embedding vector(384),  -- 384-dimensional embeddings for sentence-transformers
                transliteration VARCHAR(255),
                is_proper_noun BOOLEAN DEFAULT FALSE,
                confidence_score FLOAT DEFAULT 1.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata JSONB
            )
            """
            
            # Create term_relationships table
            term_relationships_query = """
            CREATE TABLE IF NOT EXISTS term_relationships (
                id SERIAL PRIMARY KEY,
                source_term_id INTEGER REFERENCES semantic_terms(id) ON DELETE CASCADE,
                target_term_id INTEGER REFERENCES semantic_terms(id) ON DELETE CASCADE,
                relationship_type VARCHAR(50) NOT NULL,
                strength FLOAT DEFAULT 0.0,
                context TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata JSONB,
                UNIQUE(source_term_id, target_term_id, relationship_type)
            )
            """
            
            # Create indexes for performance
            indexes = [
                "CREATE INDEX IF NOT EXISTS idx_semantic_terms_term ON semantic_terms(term)",
                "CREATE INDEX IF NOT EXISTS idx_semantic_terms_domain ON semantic_terms(domain)",
                "CREATE INDEX IF NOT EXISTS idx_term_relationships_source ON term_relationships(source_term_id)",
                "CREATE INDEX IF NOT EXISTS idx_term_relationships_target ON term_relationships(target_term_id)",
                "CREATE INDEX IF NOT EXISTS idx_term_relationships_type ON term_relationships(relationship_type)"
            ]
            
            # Create vector similarity index if pgvector is available
            if self._pgvector_available:
                vector_index = "CREATE INDEX IF NOT EXISTS idx_semantic_terms_embedding ON semantic_terms USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)"
                indexes.append(vector_index)
            
            # Execute schema creation
            self.db_manager.execute_write_query(semantic_terms_query)
            self.db_manager.execute_write_query(term_relationships_query)
            
            for index_query in indexes:
                self.db_manager.execute_write_query(index_query)
            
            self.logger.info("Semantic database schema initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing semantic schema: {e}")
            return False

    def store_semantic_term(self, term: SemanticTerm) -> Optional[int]:
        """
        Store a semantic term with its embedding.
        
        Args:
            term: SemanticTerm object to store
            
        Returns:
            Term ID if successful, None otherwise
        """
        try:
            if self.db_manager is None:
                self.logger.error("Error storing semantic term: 'NoneType' object has no attribute 'execute_write_query'")
                return None
            # Convert embedding to vector format if available
            embedding_value = None
            if term.embedding and self._pgvector_available:
                embedding_value = f"[{','.join(map(str, term.embedding))}]"
            
            query = """
            INSERT INTO semantic_terms (
                term, domain, embedding, transliteration, is_proper_noun, 
                confidence_score, metadata
            ) VALUES (%s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """
            
            # Handle Json import properly - use json.dumps if psycopg2.Json not available
            metadata_value = None
            if term.metadata:
                if PSYCOPG2_AVAILABLE:
                    from psycopg2.extras import Json
                    metadata_value = Json(term.metadata)
                else:
                    metadata_value = json.dumps(term.metadata)
            
            params = (
                term.term,
                term.domain,
                embedding_value,
                term.transliteration,
                term.is_proper_noun,
                term.confidence_score,
                metadata_value
            )
            
            result = self.db_manager.execute_write_query(query, params)
            
            if result and len(result) > 0:
                term_id = result[0][0]
                self.logger.debug(f"Stored semantic term '{term.term}' with ID {term_id}")
                return term_id
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error storing semantic term: {e}")
            return None

    def find_similar_terms(self, embedding: List[float], domain: Optional[str] = None, 
                          limit: int = 10, similarity_threshold: float = 0.7) -> List[Tuple[SemanticTerm, float]]:
        """
        Find semantically similar terms using vector similarity.
        
        Args:
            embedding: Query embedding vector
            domain: Optional domain filter
            limit: Maximum number of results
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of (SemanticTerm, similarity_score) tuples
        """
        try:
            if not self._pgvector_available or not embedding:
                # Fallback to text-based search
                return self._fallback_text_search(domain, limit)
            
            query_embedding = f"[{','.join(map(str, embedding))}]"
            
            # Build query with optional domain filter
            base_query = """
            SELECT id, term, domain, embedding, transliteration, is_proper_noun, 
                   confidence_score, created_at, updated_at, metadata,
                   (1 - (embedding <=> %s::vector)) as similarity
            FROM semantic_terms
            WHERE (1 - (embedding <=> %s::vector)) >= %s
            """
            
            params = [query_embedding, query_embedding, similarity_threshold]
            
            if domain:
                base_query += " AND domain = %s"
                params.append(domain)
            
            base_query += " ORDER BY embedding <=> %s::vector LIMIT %s"
            params.extend([query_embedding, limit])
            
            results = self.db_manager.execute_read_query(base_query, params)
            
            similar_terms = []
            for row in results:
                term = SemanticTerm(
                    id=row[0],
                    term=row[1],
                    domain=row[2],
                    embedding=list(row[3]) if row[3] else None,
                    transliteration=row[4],
                    is_proper_noun=row[5],
                    confidence_score=row[6],
                    created_at=row[7],
                    updated_at=row[8],
                    metadata=row[9]
                )
                similarity = row[10]
                similar_terms.append((term, similarity))
            
            return similar_terms
            
        except Exception as e:
            self.logger.error(f"Error finding similar terms: {e}")
            return []

    def _fallback_text_search(self, domain: Optional[str] = None, limit: int = 10) -> List[Tuple[SemanticTerm, float]]:
        """Fallback text-based search when pgvector is unavailable."""
        try:
            query = "SELECT id, term, domain, transliteration, is_proper_noun, confidence_score, created_at, updated_at, metadata FROM semantic_terms"
            params = []
            
            if domain:
                query += " WHERE domain = %s"
                params.append(domain)
            
            query += f" LIMIT {limit}"
            
            results = self.db_manager.execute_read_query(query, params)
            
            terms = []
            for row in results:
                term = SemanticTerm(
                    id=row[0],
                    term=row[1],
                    domain=row[2],
                    transliteration=row[3],
                    is_proper_noun=row[4],
                    confidence_score=row[5],
                    created_at=row[6],
                    updated_at=row[7],
                    metadata=row[8]
                )
                # Default similarity score for fallback
                terms.append((term, 0.8))
            
            return terms
            
        except Exception as e:
            self.logger.error(f"Error in fallback search: {e}")
            return []

    def store_term_relationship(self, relationship: TermRelationship) -> Optional[int]:
        """
        Store a relationship between semantic terms.
        
        Args:
            relationship: TermRelationship object to store
            
        Returns:
            Relationship ID if successful, None otherwise
        """
        try:
            query = """
            INSERT INTO term_relationships (
                source_term_id, target_term_id, relationship_type, strength, context, metadata
            ) VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (source_term_id, target_term_id, relationship_type) 
            DO UPDATE SET strength = EXCLUDED.strength, metadata = EXCLUDED.metadata
            RETURNING id
            """
            
            # Handle Json import properly - use json.dumps if psycopg2.Json not available
            metadata_value = None
            if relationship.metadata:
                if PSYCOPG2_AVAILABLE:
                    from psycopg2.extras import Json
                    metadata_value = Json(relationship.metadata)
                else:
                    metadata_value = json.dumps(relationship.metadata)
            
            params = (
                relationship.source_term_id,
                relationship.target_term_id,
                relationship.relationship_type,
                relationship.strength,
                relationship.context,
                metadata_value
            )
            
            result = self.db_manager.execute_write_query(query, params)
            
            if result and len(result) > 0:
                relationship_id = result[0][0]
                self.logger.debug(f"Stored term relationship with ID {relationship_id}")
                return relationship_id
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error storing term relationship: {e}")
            return None

    def get_term_relationships(self, term_id: int, relationship_types: Optional[List[str]] = None) -> List[TermRelationship]:
        """
        Get relationships for a specific term.
        
        Args:
            term_id: ID of the term to get relationships for
            relationship_types: Optional filter for relationship types
            
        Returns:
            List of TermRelationship objects
        """
        try:
            query = """
            SELECT id, source_term_id, target_term_id, relationship_type, strength, context, created_at, metadata
            FROM term_relationships
            WHERE source_term_id = %s OR target_term_id = %s
            """
            
            params = [term_id, term_id]
            
            if relationship_types:
                placeholders = ','.join(['%s'] * len(relationship_types))
                query += f" AND relationship_type IN ({placeholders})"
                params.extend(relationship_types)
            
            query += " ORDER BY strength DESC"
            
            results = self.db_manager.execute_read_query(query, params)
            
            relationships = []
            for row in results:
                relationship = TermRelationship(
                    id=row[0],
                    source_term_id=row[1],
                    target_term_id=row[2],
                    relationship_type=row[3],
                    strength=row[4],
                    context=row[5],
                    created_at=row[6],
                    metadata=row[7]
                )
                relationships.append(relationship)
            
            return relationships
            
        except Exception as e:
            self.logger.error(f"Error getting term relationships: {e}")
            return []

    async def store_term_analysis(self, term: str, analysis_result: Dict[str, Any], 
                                 embedding: Optional[List[float]] = None, 
                                 cache_key: str = None, ttl_seconds: int = 3600) -> bool:
        """
        Store semantic analysis result for caching purposes.
        
        Args:
            term: The term that was analyzed
            analysis_result: Complete analysis result data
            embedding: Optional embedding vector
            cache_key: Optional cache key for retrieval
            ttl_seconds: Time to live for cache entry
            
        Returns:
            True if stored successfully, False otherwise
        """
        try:
            # Create SemanticTerm from analysis result
            semantic_term = SemanticTerm(
                term=term,
                domain=analysis_result.get('domain', 'general'),
                embedding=embedding,
                transliteration=analysis_result.get('transliteration'),
                confidence_score=analysis_result.get('confidence_score', 0.0),
                metadata={
                    'cache_key': cache_key,
                    'analysis_result': analysis_result,
                    'cached_at': analysis_result.get('analysis_timestamp'),
                    'ttl_seconds': ttl_seconds,
                    'cache_version': analysis_result.get('cache_version', '1.0')
                }
            )
            
            # Store the semantic term (this handles upsert automatically)
            term_id = self.store_semantic_term(semantic_term)
            
            if term_id:
                self.logger.debug(f"Successfully cached analysis for term '{term}' with ID {term_id}")
                return True
            else:
                self.logger.debug(f"Database caching unavailable for term '{term}' - continuing with file-based fallback")
                return False
                
        except Exception as e:
            self.logger.error(f"Error storing term analysis cache for '{term}': {e}")
            return False

    async def get_cached_analysis(self, term: str, context: str, cache_key: str = None) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached semantic analysis result.
        
        Args:
            term: The term to look up
            context: Context string (for validation)
            cache_key: Optional specific cache key to lookup
            
        Returns:
            Cached analysis result or None if not found/expired
        """
        try:
            # Look up term by name and optionally by cache key
            query = """
            SELECT term, domain, embedding, transliteration, confidence_score, 
                   created_at, updated_at, metadata
            FROM semantic_terms 
            WHERE term = %s
            """
            params = [term]
            
            if cache_key:
                query += " AND metadata->>'cache_key' = %s"
                params.append(cache_key)
            
            query += " ORDER BY updated_at DESC LIMIT 1"
            
            results = self.db_manager.execute_read_query(query, params)
            
            if not results:
                return None
                
            row = results[0]
            metadata = row[7] or {}
            
            # Check if cache entry has expired
            cached_at = metadata.get('cached_at')
            ttl_seconds = metadata.get('ttl_seconds', 3600)
            
            if cached_at:
                import time
                if time.time() - cached_at > ttl_seconds:
                    self.logger.debug(f"Cache entry for '{term}' has expired")
                    return None
            
            # Return the cached analysis result
            analysis_result = metadata.get('analysis_result', {})
            
            # Ensure we have the embedding from the database if not in metadata
            if 'embedding' not in analysis_result and row[2]:
                analysis_result['embedding'] = list(row[2])
            
            # Mark as cache hit
            analysis_result['cache_hit'] = True
            
            self.logger.debug(f"Cache hit for term '{term}'")
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"Error retrieving cached analysis for '{term}': {e}")
            return None

    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of the vector database system."""
        status = {
            'pgvector_available': self._pgvector_available,
            'database_connected': False,
            'schema_initialized': False,
            'performance_metrics': {}
        }
        
        try:
            # Test database connection
            test_query = "SELECT 1"
            result = self.db_manager.execute_read_query(test_query)
            status['database_connected'] = len(result) > 0
            
            # Check if schema exists
            schema_query = """
            SELECT COUNT(*) FROM information_schema.tables 
            WHERE table_name IN ('semantic_terms', 'term_relationships')
            """
            result = self.db_manager.execute_read_query(schema_query)
            status['schema_initialized'] = result[0][0] == 2 if result else False
            
            # Get performance metrics
            if status['schema_initialized']:
                metrics_query = """
                SELECT 
                    (SELECT COUNT(*) FROM semantic_terms) as terms_count,
                    (SELECT COUNT(*) FROM term_relationships) as relationships_count
                """
                result = self.db_manager.execute_read_query(metrics_query)
                if result:
                    status['performance_metrics'] = {
                        'total_terms': result[0][0],
                        'total_relationships': result[0][1]
                    }
            
        except Exception as e:
            self.logger.error(f"Error getting health status: {e}")
        
        return status


# Factory function for easy initialization
def create_vector_database_manager() -> VectorDatabaseManager:
    """Create a VectorDatabaseManager instance with default configuration."""
    return VectorDatabaseManager()


# Singleton instance for global access
_vector_db_manager = None


def get_vector_database_manager() -> VectorDatabaseManager:
    """Get or create the global VectorDatabaseManager instance."""
    global _vector_db_manager
    if _vector_db_manager is None:
        _vector_db_manager = create_vector_database_manager()
    return _vector_db_manager