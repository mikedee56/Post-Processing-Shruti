"""
Knowledge Capture System - Story 3.3.1

This module implements the core knowledge capture functionality for learning
from expert decisions and improving automatic processing over time.

The system captures expert decisions, extracts patterns, and stores them
for future application to reduce human review requirements.
"""

import json
import sqlite3
import hashlib
import uuid
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import logging
from dataclasses import asdict

from .validation_models import (
    ValidationCase, ExpertDecision, LearningPattern, 
    KnowledgeCaptureMetrics, DecisionType, ConfidenceLevel, ValidationStatus,
    ExpertRecommendationType
)

logger = logging.getLogger(__name__)


class KnowledgeCapture:
    """
    Core knowledge capture system for expert decision learning
    
    This system:
    1. Stores expert decisions in a structured format
    2. Extracts patterns from decision history
    3. Tracks metrics on learning effectiveness
    4. Provides API for integration with processing pipeline
    """
    
    def __init__(self, database_path: Optional[str] = None, lexicon_manager: Optional['LexiconManager'] = None):
        """
        Initialize knowledge capture system
        
        Args:
            database_path: Path to SQLite database for storing decisions
            lexicon_manager: Optional LexiconManager for automatic vocabulary updates
        """
        self.database_path = database_path or "data/knowledge_capture/decisions.db"
        self.db_path = Path(self.database_path)
        self.lexicon_manager = lexicon_manager
        
        # Ensure directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._initialize_database()
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize metrics
        self.metrics = KnowledgeCaptureMetrics()
        
        logger.info(f"Knowledge capture system initialized with database: {self.database_path}")
        if self.lexicon_manager:
            logger.info("Knowledge capture system integrated with lexicon manager for automatic updates")
    
    def _initialize_database(self):
        """Initialize SQLite database with required tables"""
        try:
            with sqlite3.connect(self.database_path) as conn:
                # Create validation_cases table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS validation_cases (
                        case_id TEXT PRIMARY KEY,
                        original_text TEXT NOT NULL,
                        processed_text TEXT NOT NULL,
                        processing_context TEXT NOT NULL,
                        flagging_reasons TEXT NOT NULL,
                        confidence_scores TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        status TEXT NOT NULL,
                        assigned_expert TEXT,
                        priority_score REAL DEFAULT 0.0,
                        metadata TEXT DEFAULT '{}'
                    )
                """)
                
                # Create expert_decisions table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS expert_decisions (
                        decision_id TEXT PRIMARY KEY,
                        case_id TEXT NOT NULL,
                        expert_id TEXT NOT NULL,
                        decision_type TEXT NOT NULL,
                        approved_text TEXT NOT NULL,
                        reasoning TEXT NOT NULL,
                        confidence TEXT NOT NULL,
                        decision_timestamp TEXT NOT NULL,
                        processing_time_seconds REAL NOT NULL,
                        tags TEXT DEFAULT '[]',
                        pattern_hints TEXT DEFAULT '{}',
                        metadata TEXT DEFAULT '{}',
                        FOREIGN KEY (case_id) REFERENCES validation_cases (case_id)
                    )
                """)
                
                # Create learning_patterns table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS learning_patterns (
                        pattern_id TEXT PRIMARY KEY,
                        pattern_type TEXT NOT NULL,
                        condition_data TEXT NOT NULL,
                        action_data TEXT NOT NULL,
                        confidence_score REAL NOT NULL,
                        support_count INTEGER NOT NULL,
                        success_rate REAL NOT NULL,
                        created_timestamp TEXT NOT NULL,
                        last_updated TEXT NOT NULL,
                        version INTEGER DEFAULT 1,
                        is_active BOOLEAN DEFAULT TRUE,
                        expert_feedback TEXT DEFAULT '[]'
                    )
                """)
                
                # Create indexes for performance
                conn.execute("CREATE INDEX IF NOT EXISTS idx_cases_status ON validation_cases (status)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_cases_expert ON validation_cases (assigned_expert)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_decisions_expert ON expert_decisions (expert_id)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_decisions_type ON expert_decisions (decision_type)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_patterns_type ON learning_patterns (pattern_type)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_patterns_active ON learning_patterns (is_active)")
                
                conn.commit()
                logger.info("Database schema initialized successfully")
                
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration for knowledge capture system"""
        default_config = {
            'pattern_extraction': {
                'min_support_count': 3,
                'min_confidence_score': 0.7,
                'max_patterns_per_type': 100
            },
            'decision_capture': {
                'auto_extract_patterns': True,
                'require_reasoning': True,
                'min_processing_time_seconds': 5.0
            },
            'learning': {
                'pattern_application_threshold': 0.8,
                'feedback_learning_rate': 0.1,
                'success_rate_threshold': 0.75
            }
        }
        
        config_path = Path("config/knowledge_capture_config.json")
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                    # Merge with defaults
                    default_config.update(user_config)
            except Exception as e:
                logger.warning(f"Failed to load config, using defaults: {e}")
        
        return default_config
    
    def store_validation_case(self, validation_case: ValidationCase) -> Optional[str]:
        """
        Store a validation case for expert review
        
        Args:
            validation_case: ValidationCase to store
            
        Returns:
            case_id if stored successfully, None otherwise
        """
        try:
            with sqlite3.connect(self.database_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO validation_cases 
                    (case_id, original_text, processed_text, processing_context, 
                     flagging_reasons, confidence_scores, timestamp, status, 
                     assigned_expert, priority_score, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    validation_case.case_id,
                    validation_case.original_text,
                    validation_case.processed_text,
                    json.dumps(validation_case.processing_context),
                    json.dumps(validation_case.flagging_reasons),
                    json.dumps(validation_case.confidence_scores),
                    validation_case.timestamp.isoformat(),
                    validation_case.status.value,
                    validation_case.assigned_expert,
                    validation_case.priority_score,
                    json.dumps(validation_case.metadata)
                ))
                conn.commit()
                
            logger.debug(f"Stored validation case: {validation_case.case_id}")
            return validation_case.case_id
            
        except Exception as e:
            logger.error(f"Failed to store validation case {validation_case.case_id}: {e}")
            return None
    
    def store_expert_decision(self, expert_decision: ExpertDecision) -> Optional[str]:
        """
        Store an expert decision and trigger pattern extraction
        
        Args:
            expert_decision: ExpertDecision to store
            
        Returns:
            decision_id if stored successfully, None otherwise
        """
        try:
            with sqlite3.connect(self.database_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO expert_decisions 
                    (decision_id, case_id, expert_id, decision_type, approved_text, 
                     reasoning, confidence, decision_timestamp, processing_time_seconds,
                     tags, pattern_hints, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    expert_decision.decision_id,
                    expert_decision.case_id,
                    expert_decision.expert_id,
                    expert_decision.decision_type.value,
                    expert_decision.approved_text,
                    expert_decision.reasoning,
                    expert_decision.confidence.value,
                    expert_decision.decision_timestamp.isoformat(),
                    expert_decision.processing_time_seconds,
                    json.dumps(expert_decision.tags),
                    json.dumps(expert_decision.pattern_hints),
                    json.dumps(expert_decision.metadata)
                ))
                conn.commit()
                
            # Update validation case status
            self._update_case_status(expert_decision.case_id, ValidationStatus.APPROVED)
            
            # Trigger pattern extraction if enabled
            if self.config['decision_capture']['auto_extract_patterns']:
                self._extract_patterns_from_decision(expert_decision)
            
            # Update lexicon with expert decision
            validation_case = self._get_validation_case(expert_decision.case_id)
            if validation_case:
                self._update_lexicon_from_decision(validation_case, expert_decision)
            
            # Update metrics
            self.metrics.total_decisions_captured += 1
            self.metrics.last_updated = datetime.now()
            
            logger.debug(f"Stored expert decision: {expert_decision.decision_id}")
            return expert_decision.decision_id
            
        except Exception as e:
            logger.error(f"Failed to store expert decision {expert_decision.decision_id}: {e}")
            return None
    
    def _update_case_status(self, case_id: str, status: ValidationStatus):
        """Update the status of a validation case"""
        try:
            with sqlite3.connect(self.database_path) as conn:
                conn.execute(
                    "UPDATE validation_cases SET status = ? WHERE case_id = ?",
                    (status.value, case_id)
                )
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to update case status for {case_id}: {e}")
    
    def get_validation_cases(self, 
                           status: Optional[ValidationStatus] = None,
                           assigned_expert: Optional[str] = None,
                           limit: int = 100) -> List[ValidationCase]:
        """
        Retrieve validation cases from database
        
        Args:
            status: Filter by status (optional)
            assigned_expert: Filter by assigned expert (optional)
            limit: Maximum number of cases to return
            
        Returns:
            List of ValidationCase objects
        """
        try:
            with sqlite3.connect(self.database_path) as conn:
                query = "SELECT * FROM validation_cases WHERE 1=1"
                params = []
                
                if status:
                    query += " AND status = ?"
                    params.append(status.value)
                
                if assigned_expert:
                    query += " AND assigned_expert = ?"
                    params.append(assigned_expert)
                
                query += " ORDER BY priority_score DESC, timestamp DESC LIMIT ?"
                params.append(limit)
                
                cursor = conn.execute(query, params)
                rows = cursor.fetchall()
                
                cases = []
                for row in rows:
                    case = ValidationCase(
                        case_id=row[0],
                        original_text=row[1],
                        processed_text=row[2],
                        processing_context=json.loads(row[3]),
                        flagging_reasons=json.loads(row[4]),
                        confidence_scores=json.loads(row[5]),
                        timestamp=datetime.fromisoformat(row[6]),
                        status=ValidationStatus(row[7]),
                        assigned_expert=row[8],
                        priority_score=row[9],
                        metadata=json.loads(row[10]) if row[10] else {}
                    )
                    cases.append(case)
                
                return cases
                
        except Exception as e:
            logger.error(f"Failed to retrieve validation cases: {e}")
            return []
    
    def get_expert_decisions(self, 
                            expert_id: Optional[str] = None,
                            decision_type: Optional[DecisionType] = None,
                            limit: int = 100) -> List[ExpertDecision]:
        """
        Retrieve expert decisions from database
        
        Args:
            expert_id: Filter by expert ID (optional)
            decision_type: Filter by decision type (optional)
            limit: Maximum number of decisions to return
            
        Returns:
            List of ExpertDecision objects
        """
        try:
            with sqlite3.connect(self.database_path) as conn:
                query = "SELECT * FROM expert_decisions WHERE 1=1"
                params = []
                
                if expert_id:
                    query += " AND expert_id = ?"
                    params.append(expert_id)
                
                if decision_type:
                    query += " AND decision_type = ?"
                    params.append(decision_type.value)
                
                query += " ORDER BY decision_timestamp DESC LIMIT ?"
                params.append(limit)
                
                cursor = conn.execute(query, params)
                rows = cursor.fetchall()
                
                decisions = []
                for row in rows:
                    decision = ExpertDecision(
                        decision_id=row[0],
                        case_id=row[1],
                        expert_id=row[2],
                        decision_type=DecisionType(row[3]),
                        approved_text=row[4],
                        reasoning=row[5],
                        confidence=ConfidenceLevel(row[6]),
                        decision_timestamp=datetime.fromisoformat(row[7]),
                        processing_time_seconds=row[8],
                        tags=json.loads(row[9]) if row[9] else [],
                        pattern_hints=json.loads(row[10]) if row[10] else {},
                        metadata=json.loads(row[11]) if row[11] else {}
                    )
                    decisions.append(decision)
                
                return decisions
                
        except Exception as e:
            logger.error(f"Failed to retrieve expert decisions: {e}")
            return []
    
    def _extract_patterns_from_decision(self, expert_decision: ExpertDecision):
        """
        Extract learning patterns from an expert decision
        
        This is a simplified pattern extraction. In production, this would
        use more sophisticated ML techniques.
        """
        try:
            # Get the original validation case
            case = self.get_validation_case(expert_decision.case_id)
            if not case:
                return
            
            # Extract simple patterns based on decision type
            if expert_decision.decision_type == DecisionType.LEXICON_CORRECTION:
                self._extract_lexicon_pattern(case, expert_decision)
            elif expert_decision.decision_type == DecisionType.CAPITALIZATION_CHANGE:
                self._extract_capitalization_pattern(case, expert_decision)
            elif expert_decision.decision_type == DecisionType.TRANSLITERATION_FIX:
                self._extract_transliteration_pattern(case, expert_decision)
            
        except Exception as e:
            logger.error(f"Failed to extract patterns from decision {expert_decision.decision_id}: {e}")
    
    def _extract_lexicon_pattern(self, case: ValidationCase, decision: ExpertDecision):
        """Extract pattern for lexicon corrections"""
        # Simple pattern: if text contains X, change to Y
        pattern_condition = {
            'contains': case.original_text.lower(),
            'context_type': case.processing_context.get('context_type', 'general')
        }
        
        pattern_action = {
            'replace': {
                'from': case.original_text,
                'to': decision.approved_text
            }
        }
        
        self._create_or_update_pattern(
            DecisionType.LEXICON_CORRECTION,
            pattern_condition,
            pattern_action,
            decision.confidence
        )

    
    def _update_lexicon_from_decision(self, case: ValidationCase, decision: ExpertDecision):
        """Update lexicon manager with expert decision patterns"""
        if not self.lexicon_manager:
            logger.debug("No lexicon manager available for automatic updates")
            return
        
        try:
            # Extract lexicon updates for corrections
            if decision.decision_type == DecisionType.LEXICON_CORRECTION:
                original_term = case.original_text.strip().lower()
                corrected_term = decision.approved_text.strip()
                
                # Check if this is a new lexicon entry or update to existing
                existing_entries = self.lexicon_manager.get_all_entries()
                
                # Look for existing entry that might match
                matching_entry = None
                for term, entry in existing_entries.items():
                    if term.lower() == corrected_term.lower():
                        matching_entry = (term, entry)
                        break
                    # Check variations
                    if hasattr(entry, 'variations') and original_term in [v.lower() for v in entry.variations]:
                        matching_entry = (term, entry)
                        break
                
                if matching_entry:
                    # Update existing entry with new variation
                    term, entry = matching_entry
                    if hasattr(entry, 'variations') and original_term not in [v.lower() for v in entry.variations]:
                        entry.variations.append(original_term)
                        logger.info(f"Added variation '{original_term}' to existing lexicon entry '{term}'")
                else:
                    # This could be a completely new term - log for manual review
                    logger.info(f"Expert decision suggests new lexicon entry: '{original_term}' -> '{corrected_term}'. Manual lexicon review recommended.")
                
            # Update confidence scores based on expert validation
            if decision.confidence in [ConfidenceLevel.HIGH, ConfidenceLevel.VERY_HIGH]:
                self._boost_lexicon_confidence(case.original_text, decision.approved_text)
                
        except Exception as e:
            logger.error(f"Error updating lexicon from expert decision: {e}")
    
    def _boost_lexicon_confidence(self, original_text: str, corrected_text: str):
        """Boost confidence for lexicon entries validated by experts"""
        if not self.lexicon_manager:
            return
            
        try:
            existing_entries = self.lexicon_manager.get_all_entries()
            for term, entry in existing_entries.items():
                if term.lower() == corrected_text.lower():
                    # Boost confidence for validated entries
                    if hasattr(entry, 'confidence') and entry.confidence < 1.0:
                        entry.confidence = min(1.0, entry.confidence + 0.1)
                        logger.debug(f"Boosted confidence for lexicon entry '{term}' to {entry.confidence}")
                    break
        except Exception as e:
            logger.error(f"Error boosting lexicon confidence: {e}")
    
    def get_lexicon_update_stats(self) -> Dict[str, Any]:
        """Get statistics on automatic lexicon updates"""
        try:
            cursor = self.connection.cursor()
            
            # Count decisions that led to lexicon updates
            cursor.execute("""
                SELECT decision_type, COUNT(*) as count
                FROM expert_decisions 
                WHERE decision_type = 'LEXICON_CORRECTION'
                AND status = 'APPROVED'
                GROUP BY decision_type
            """)
            
            update_counts = dict(cursor.fetchall())
            
            # Get recent lexicon-related decisions
            cursor.execute("""
                SELECT created_at, approved_text, confidence_level
                FROM expert_decisions 
                WHERE decision_type = 'LEXICON_CORRECTION'
                AND status = 'APPROVED'
                ORDER BY created_at DESC
                LIMIT 10
            """)
            
            recent_updates = [
                {
                    'timestamp': row[0],
                    'approved_text': row[1],
                    'confidence': row[2]
                }
                for row in cursor.fetchall()
            ]
            
            return {
                'lexicon_updates_count': update_counts.get('LEXICON_CORRECTION', 0),
                'recent_updates': recent_updates,
                'lexicon_manager_available': self.lexicon_manager is not None,
                'auto_update_enabled': self.lexicon_manager is not None
            }
            
        except Exception as e:
            logger.error(f"Error getting lexicon update stats: {e}")
            return {
                'lexicon_updates_count': 0,
                'recent_updates': [],
                'lexicon_manager_available': False,
                'auto_update_enabled': False,
                'error': str(e)
            }

    
    def _get_validation_case(self, case_id: str) -> Optional[ValidationCase]:
        """Retrieve validation case by ID for lexicon updates"""
        try:
            with sqlite3.connect(self.database_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT case_id, original_text, suggested_text, issue_type, 
                           confidence_level, context_data, metadata_json, expert_id,
                           created_at, status
                    FROM validation_cases 
                    WHERE case_id = ?
                """, (case_id,))
                
                row = cursor.fetchone()
                if row:
                    return ValidationCase(
                        case_id=row[0],
                        original_text=row[1],
                        suggested_text=row[2],
                        issue_type=row[3],
                        confidence_level=ConfidenceLevel(row[4]),
                        processing_context=json.loads(row[5]) if row[5] else {},
                        metadata=json.loads(row[6]) if row[6] else {},
                        expert_id=row[7],
                        created_timestamp=datetime.fromisoformat(row[8]) if row[8] else datetime.now(),
                        status=ValidationStatus(row[9])
                    )
                    
        except Exception as e:
            logger.error(f"Error retrieving validation case {case_id}: {e}")
        
        return None
    
    def _extract_capitalization_pattern(self, case: ValidationCase, decision: ExpertDecision):
        """Extract pattern for capitalization changes"""
        # Pattern: capitalize specific terms
        words_changed = []
        original_words = case.original_text.split()
        approved_words = decision.approved_text.split()
        
        for i, (orig, appr) in enumerate(zip(original_words, approved_words)):
            if orig.lower() == appr.lower() and orig != appr:
                words_changed.append(orig.lower())
        
        if words_changed:
            pattern_condition = {
                'contains_words': words_changed,
                'context_type': case.processing_context.get('context_type', 'general')
            }
            
            pattern_action = {
                'capitalize_words': words_changed
            }
            
            self._create_or_update_pattern(
                DecisionType.CAPITALIZATION_CHANGE,
                pattern_condition,
                pattern_action,
                decision.confidence
            )
    
    def _extract_transliteration_pattern(self, case: ValidationCase, decision: ExpertDecision):
        """Extract pattern for transliteration fixes"""
        pattern_condition = {
            'original_form': case.original_text,
            'context_type': case.processing_context.get('context_type', 'general')
        }
        
        pattern_action = {
            'transliterate_to': decision.approved_text,
            'transliteration_standard': 'IAST'
        }
        
        self._create_or_update_pattern(
            DecisionType.TRANSLITERATION_FIX,
            pattern_condition,
            pattern_action,
            decision.confidence
        )
    
    def _create_or_update_pattern(self, 
                                  pattern_type: DecisionType,
                                  condition: Dict[str, Any],
                                  action: Dict[str, Any],
                                  confidence: ConfidenceLevel):
        """Create or update a learning pattern"""
        try:
            # Generate pattern ID based on content hash
            content_hash = hashlib.md5(
                json.dumps(condition, sort_keys=True).encode() +
                json.dumps(action, sort_keys=True).encode()
            ).hexdigest()[:16]
            
            pattern_id = f"{pattern_type.value}_{content_hash}"
            
            # Check if pattern already exists
            existing_pattern = self.get_learning_pattern(pattern_id)
            
            if existing_pattern:
                # Update existing pattern
                existing_pattern.support_count += 1
                existing_pattern.last_updated = datetime.now()
                existing_pattern.version += 1
                
                # Update confidence based on new evidence
                confidence_score = self._confidence_to_score(confidence)
                existing_pattern.confidence_score = (
                    existing_pattern.confidence_score * 0.8 + confidence_score * 0.2
                )
                
                self.store_learning_pattern(existing_pattern)
            else:
                # Create new pattern
                new_pattern = LearningPattern(
                    pattern_id=pattern_id,
                    pattern_type=pattern_type,
                    condition=condition,
                    action=action,
                    confidence_score=self._confidence_to_score(confidence),
                    support_count=1,
                    success_rate=1.0,  # Initial success rate
                    created_timestamp=datetime.now(),
                    last_updated=datetime.now()
                )
                
                self.store_learning_pattern(new_pattern)
                self.metrics.patterns_identified += 1
            
        except Exception as e:
            logger.error(f"Failed to create/update pattern: {e}")
    
    def _confidence_to_score(self, confidence: ConfidenceLevel) -> float:
        """Convert confidence level to numerical score"""
        confidence_map = {
            ConfidenceLevel.LOW: 0.4,
            ConfidenceLevel.MEDIUM: 0.7,
            ConfidenceLevel.HIGH: 0.85,
            ConfidenceLevel.VERY_HIGH: 0.95
        }
        return confidence_map.get(confidence, 0.5)
    
    def get_validation_case(self, case_id: str) -> Optional[ValidationCase]:
        """Get a specific validation case by ID"""
        cases = self.get_validation_cases()
        for case in cases:
            if case.case_id == case_id:
                return case
        return None
    
    def get_learning_pattern(self, pattern_id: str) -> Optional[LearningPattern]:
        """Get a specific learning pattern by ID"""
        try:
            with sqlite3.connect(self.database_path) as conn:
                cursor = conn.execute(
                    "SELECT * FROM learning_patterns WHERE pattern_id = ?",
                    (pattern_id,)
                )
                row = cursor.fetchone()
                
                if row:
                    return LearningPattern(
                        pattern_id=row[0],
                        pattern_type=DecisionType(row[1]),
                        condition=json.loads(row[2]),
                        action=json.loads(row[3]),
                        confidence_score=row[4],
                        support_count=row[5],
                        success_rate=row[6],
                        created_timestamp=datetime.fromisoformat(row[7]),
                        last_updated=datetime.fromisoformat(row[8]),
                        version=row[9],
                        is_active=bool(row[10]),
                        expert_feedback=json.loads(row[11]) if row[11] else []
                    )
                
                return None
                
        except Exception as e:
            logger.error(f"Failed to retrieve learning pattern {pattern_id}: {e}")
            return None

    def get_patterns_by_type(self, pattern_type: DecisionType) -> List[LearningPattern]:
        """Get all learning patterns of a specific type"""
        try:
            with sqlite3.connect(self.database_path) as conn:
                cursor = conn.execute(
                    "SELECT * FROM learning_patterns WHERE pattern_type = ? AND is_active = 1",
                    (pattern_type.value,)
                )
                rows = cursor.fetchall()
                
                patterns = []
                for row in rows:
                    patterns.append(LearningPattern(
                        pattern_id=row[0],
                        pattern_type=DecisionType(row[1]),
                        condition=json.loads(row[2]),
                        action=json.loads(row[3]),
                        confidence_score=row[4],
                        support_count=row[5],
                        success_rate=row[6],
                        created_timestamp=datetime.fromisoformat(row[7]),
                        last_updated=datetime.fromisoformat(row[8]),
                        version=row[9],
                        is_active=bool(row[10]),
                        expert_feedback=json.loads(row[11]) if row[11] else []
                    ))
                
                return patterns
                
        except Exception as e:
            logger.error(f"Failed to retrieve patterns by type {pattern_type}: {e}")
            return []
    
    def store_learning_pattern(self, pattern: LearningPattern) -> bool:
        """Store a learning pattern in the database"""
        try:
            with sqlite3.connect(self.database_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO learning_patterns 
                    (pattern_id, pattern_type, condition_data, action_data,
                     confidence_score, support_count, success_rate,
                     created_timestamp, last_updated, version, is_active,
                     expert_feedback)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    pattern.pattern_id,
                    pattern.pattern_type.value,
                    json.dumps(pattern.condition),
                    json.dumps(pattern.action),
                    pattern.confidence_score,
                    pattern.support_count,
                    pattern.success_rate,
                    pattern.created_timestamp.isoformat(),
                    pattern.last_updated.isoformat(),
                    pattern.version,
                    pattern.is_active,
                    json.dumps(pattern.expert_feedback)
                ))
                conn.commit()
                
            logger.debug(f"Stored learning pattern: {pattern.pattern_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store learning pattern {pattern.pattern_id}: {e}")
            return False
    
    def get_applicable_patterns(self, context: Dict[str, Any]) -> List[LearningPattern]:
        """
        Get learning patterns applicable to the given context
        
        Args:
            context: Processing context to match against
            
        Returns:
            List of applicable LearningPattern objects
        """
        try:
            with sqlite3.connect(self.database_path) as conn:
                cursor = conn.execute("""
                    SELECT * FROM learning_patterns 
                    WHERE is_active = TRUE 
                    AND confidence_score >= ?
                    ORDER BY confidence_score DESC, support_count DESC
                """, (self.config['learning']['pattern_application_threshold'],))
                
                rows = cursor.fetchall()
                applicable_patterns = []
                
                for row in rows:
                    pattern = LearningPattern(
                        pattern_id=row[0],
                        pattern_type=DecisionType(row[1]),
                        condition=json.loads(row[2]),
                        action=json.loads(row[3]),
                        confidence_score=row[4],
                        support_count=row[5],
                        success_rate=row[6],
                        created_timestamp=datetime.fromisoformat(row[7]),
                        last_updated=datetime.fromisoformat(row[8]),
                        version=row[9],
                        is_active=bool(row[10]),
                        expert_feedback=json.loads(row[11]) if row[11] else []
                    )
                    
                    if pattern.matches_context(context):
                        applicable_patterns.append(pattern)
                
                return applicable_patterns
                
        except Exception as e:
            logger.error(f"Failed to get applicable patterns: {e}")
            return []
    
    def get_metrics(self) -> KnowledgeCaptureMetrics:
        """Get current knowledge capture metrics"""
        try:
            with sqlite3.connect(self.database_path) as conn:
                # Update metrics from database
                cursor = conn.execute("SELECT COUNT(*) FROM expert_decisions")
                self.metrics.total_decisions_captured = cursor.fetchone()[0]
                
                cursor = conn.execute("SELECT COUNT(*) FROM learning_patterns WHERE is_active = TRUE")
                self.metrics.patterns_identified = cursor.fetchone()[0]
                
                # Calculate average confidence
                cursor = conn.execute("""
                    SELECT AVG(
                        CASE confidence 
                            WHEN 'low' THEN 0.4 
                            WHEN 'medium' THEN 0.7 
                            WHEN 'high' THEN 0.85 
                            WHEN 'very_high' THEN 0.95 
                            ELSE 0.5 
                        END
                    ) FROM expert_decisions
                """)
                avg_confidence = cursor.fetchone()[0]
                if avg_confidence:
                    self.metrics.average_decision_confidence = avg_confidence
                
                self.metrics.last_updated = datetime.now()
                
                # Add enhanced metrics for Story 3.3.1
                if not hasattr(self.metrics, 'additional_stats'):
                    self.metrics.additional_stats = {}
                
                # Count lexicon updates
                cursor = conn.execute("""
                    SELECT COUNT(*) FROM expert_decisions 
                    WHERE decision_type = 'LEXICON_CORRECTION'
                """)
                lexicon_updates = cursor.fetchone()[0]
                self.metrics.additional_stats['lexicon_updates_applied'] = lexicon_updates
                
                # Calculate review load reduction estimate
                if self.metrics.patterns_identified > 0:
                    review_reduction = min(0.75, self.metrics.patterns_identified * 0.05)
                else:
                    review_reduction = 0.0
                self.metrics.additional_stats['review_load_reduction'] = review_reduction
                
        except Exception as e:
            logger.error(f"Failed to calculate metrics: {e}")
        
        return self.metrics
    
    def apply_lexicon_updates_from_decision(self, expert_decision: ExpertDecision) -> bool:
        """
        Apply lexicon updates based on expert decision
        
        Args:
            expert_decision: ExpertDecision containing lexicon correction information
            
        Returns:
            True if update was successful, False otherwise
        """
        try:
            if not self.lexicon_manager:
                logger.debug("No lexicon manager available for updates")
                return False
            
            if expert_decision.decision_type != DecisionType.LEXICON_CORRECTION:
                logger.debug(f"Decision type {expert_decision.decision_type} not applicable for lexicon updates")
                return False
            
            # Validate decision data
            if not expert_decision.original_text.strip() or not expert_decision.corrected_text.strip():
                logger.warning("Cannot apply lexicon update with empty original or corrected text")
                return False
            
            if not (0.0 <= expert_decision.confidence_score <= 1.0):
                logger.warning(f"Invalid confidence score: {expert_decision.confidence_score}")
                return False
            
            original_term = expert_decision.original_text.strip().lower()
            corrected_term = expert_decision.corrected_text.strip()
            
            # Get existing entries
            existing_entries = self.lexicon_manager.get_all_entries()
            
            # Look for existing entry that matches the corrected term
            matching_entry = None
            for term, entry in existing_entries.items():
                if term.lower() == corrected_term.lower():
                    matching_entry = (term, entry)
                    break
                # Check if this original term is already in variations
                if hasattr(entry, 'variations') and original_term in [v.lower() for v in entry.variations]:
                    matching_entry = (term, entry)
                    break
            
            if matching_entry:
                # Update existing entry with new variation
                term, entry = matching_entry
                if hasattr(entry, 'variations') and original_term not in [v.lower() for v in entry.variations]:
                    entry.variations.append(original_term)
                    logger.info(f"Added variation '{original_term}' to existing lexicon entry '{term}'")
                
                # Update confidence using weighted average
                if hasattr(entry, 'confidence'):
                    new_confidence = (entry.confidence + expert_decision.confidence_score) / 2
                    entry.confidence = new_confidence
                    logger.debug(f"Updated confidence for '{term}' to {new_confidence:.3f}")
                
                return True
            else:
                # Create new lexicon entry for terms not already present
                from sanskrit_hindi_identifier.lexicon_manager import LexiconEntry
                
                new_entry = LexiconEntry(
                    transliteration=corrected_term,
                    variations=[original_term],
                    is_proper_noun=self._infer_proper_noun_status(corrected_term),
                    category=self._infer_term_category(corrected_term),
                    confidence=expert_decision.confidence_score,
                    source_authority="expert_decision"
                )
                
                # Add to lexicon manager
                if hasattr(self.lexicon_manager, 'add_entry'):
                    self.lexicon_manager.add_entry(corrected_term, new_entry)
                else:
                    # Fallback: directly update internal structure
                    existing_entries[corrected_term] = new_entry
                
                logger.info(f"Created new lexicon entry: '{corrected_term}' with variation '{original_term}'")
                return True
            
        except Exception as e:
            logger.error(f"Failed to apply lexicon updates from decision: {e}")
            return False
    
    def create_knowledge_base_snapshot(self, description: str = "") -> Optional[str]:
        """
        Create a snapshot of the current knowledge base state
        
        Args:
            description: Optional description of the snapshot
            
        Returns:
            snapshot_id if successful, None otherwise
        """
        try:
            # Generate unique snapshot ID
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            snapshot_id = f"snapshot_{timestamp}_{uuid.uuid4().hex[:8]}"
            
            # Create snapshots directory
            snapshots_dir = Path(self.db_path).parent / "kb_snapshots"
            snapshots_dir.mkdir(exist_ok=True)
            
            # Copy database file
            db_snapshot_path = snapshots_dir / f"{snapshot_id}_database.db"
            shutil.copy2(self.database_path, db_snapshot_path)
            
            # Copy lexicon file if available
            lexicon_snapshot_path = None
            if self.lexicon_manager and hasattr(self.lexicon_manager, 'lexicon_file'):
                lexicon_file = Path(self.lexicon_manager.lexicon_file)
                if lexicon_file.exists():
                    lexicon_snapshot_path = snapshots_dir / f"{snapshot_id}_lexicon.json"
                    shutil.copy2(lexicon_file, lexicon_snapshot_path)
            
            # Create snapshot metadata
            snapshot_metadata = {
                "snapshot_id": snapshot_id,
                "description": description,
                "created_at": datetime.now().isoformat(),
                "database_path": str(db_snapshot_path),
                "lexicon_path": str(lexicon_snapshot_path) if lexicon_snapshot_path else None,
                "creator": "knowledge_capture_system"
            }
            
            # Save metadata
            metadata_path = snapshots_dir / f"{snapshot_id}_metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(snapshot_metadata, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Created knowledge base snapshot: {snapshot_id}")
            return snapshot_id
            
        except Exception as e:
            logger.error(f"Failed to create knowledge base snapshot: {e}")
            return None
    
    def restore_knowledge_base_from_snapshot(self, snapshot_id: str) -> bool:
        """
        Restore knowledge base from a snapshot
        
        Args:
            snapshot_id: ID of the snapshot to restore from
            
        Returns:
            True if restoration was successful, False otherwise
        """
        try:
            snapshots_dir = Path(self.db_path).parent / "kb_snapshots"
            metadata_path = snapshots_dir / f"{snapshot_id}_metadata.json"
            
            if not metadata_path.exists():
                logger.error(f"Snapshot metadata not found: {snapshot_id}")
                return False
            
            # Load snapshot metadata
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            # Verify snapshot files exist
            db_snapshot_path = Path(metadata['database_path'])
            if not db_snapshot_path.exists():
                logger.error(f"Database snapshot file not found: {db_snapshot_path}")
                return False
            
            # Close current database connection
            if hasattr(self, '_current_connection'):
                try:
                    self._current_connection.close()
                except:
                    pass
            
            # Restore database
            shutil.copy2(db_snapshot_path, self.database_path)
            logger.info(f"Restored database from snapshot: {snapshot_id}")
            
            # Restore lexicon if available
            if metadata.get('lexicon_path') and self.lexicon_manager:
                lexicon_snapshot_path = Path(metadata['lexicon_path'])
                if lexicon_snapshot_path.exists():
                    if hasattr(self.lexicon_manager, 'lexicon_file'):
                        shutil.copy2(lexicon_snapshot_path, self.lexicon_manager.lexicon_file)
                        # Reload lexicon
                        if hasattr(self.lexicon_manager, 'load_lexicon'):
                            self.lexicon_manager.load_lexicon(Path(self.lexicon_manager.lexicon_file))
                        logger.info(f"Restored lexicon from snapshot: {snapshot_id}")
            
            # Reinitialize database connection
            self._initialize_database()
            
            logger.info(f"Successfully restored knowledge base from snapshot: {snapshot_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to restore knowledge base from snapshot {snapshot_id}: {e}")
            return False
    
    def list_knowledge_base_snapshots(self) -> List[Dict[str, Any]]:
        """
        List all available knowledge base snapshots
        
        Returns:
            List of snapshot information dictionaries
        """
        try:
            snapshots_dir = Path(self.db_path).parent / "kb_snapshots"
            if not snapshots_dir.exists():
                return []
            
            snapshots = []
            
            # Find all metadata files
            for metadata_file in snapshots_dir.glob("*_metadata.json"):
                try:
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                    
                    # Verify snapshot files still exist
                    db_path = Path(metadata['database_path'])
                    snapshot_valid = db_path.exists()
                    
                    snapshot_info = {
                        'snapshot_id': metadata['snapshot_id'],
                        'description': metadata['description'],
                        'created_at': metadata['created_at'],
                        'creator': metadata.get('creator', 'unknown'),
                        'has_lexicon': metadata.get('lexicon_path') is not None,
                        'valid': snapshot_valid
                    }
                    
                    snapshots.append(snapshot_info)
                    
                except Exception as e:
                    logger.warning(f"Could not parse snapshot metadata {metadata_file}: {e}")
                    continue
            
            # Sort by creation time (newest first)
            snapshots.sort(key=lambda x: x['created_at'], reverse=True)
            
            return snapshots
            
        except Exception as e:
            logger.error(f"Failed to list knowledge base snapshots: {e}")
            return []
    
    def get_expert_review_load_metrics(self) -> Dict[str, Any]:
        """
        Calculate expert review load metrics and trends
        
        Returns:
            Dictionary containing review load metrics and trend analysis
        """
        try:
            with sqlite3.connect(self.database_path) as conn:
                # Calculate current period metrics (last 30 days)
                current_date = datetime.now()
                thirty_days_ago = current_date - timedelta(days=30)
                sixty_days_ago = current_date - timedelta(days=60)
                
                # Current period decisions
                cursor = conn.execute("""
                    SELECT COUNT(*), AVG(processing_time_seconds)
                    FROM expert_decisions 
                    WHERE decision_timestamp >= ?
                """, (thirty_days_ago.isoformat(),))
                current_decisions, avg_time_current = cursor.fetchone()
                current_decisions = current_decisions or 0
                avg_time_current = avg_time_current or 0.0
                
                # Previous period decisions (30-60 days ago)
                cursor = conn.execute("""
                    SELECT COUNT(*), AVG(processing_time_seconds)
                    FROM expert_decisions 
                    WHERE decision_timestamp >= ? AND decision_timestamp < ?
                """, (sixty_days_ago.isoformat(), thirty_days_ago.isoformat()))
                previous_decisions, avg_time_previous = cursor.fetchone()
                previous_decisions = previous_decisions or 0
                avg_time_previous = avg_time_previous or 0.0
                
                # Total decisions count
                cursor = conn.execute("SELECT COUNT(*) FROM expert_decisions")
                total_decisions = cursor.fetchone()[0]
                
                # Calculate automation rate (estimated)
                # Assumption: fewer expert decisions = higher automation rate
                if total_decisions > 0:
                    # Simple heuristic: automation improves as patterns are learned
                    cursor = conn.execute("SELECT COUNT(*) FROM learning_patterns WHERE is_active = TRUE")
                    active_patterns = cursor.fetchone()[0]
                    
                    # Estimate automation rate based on patterns and decision frequency
                    base_automation = 0.6  # Base automation rate
                    pattern_boost = min(0.3, active_patterns * 0.02)  # Up to 30% boost from patterns
                    
                    # Reduce if many recent decisions (indicates low automation)
                    if current_decisions > 50:
                        frequency_penalty = min(0.2, (current_decisions - 50) * 0.004)
                    else:
                        frequency_penalty = 0.0
                    
                    automation_rate = max(0.0, base_automation + pattern_boost - frequency_penalty)
                else:
                    automation_rate = 0.8  # High automation if no expert intervention needed
                
                # Calculate trend (percentage change)
                if previous_decisions > 0:
                    change_percentage = ((current_decisions - previous_decisions) / previous_decisions) * 100
                else:
                    change_percentage = 0.0 if current_decisions == 0 else 100.0
                
                # Get workload distribution by expert
                cursor = conn.execute("""
                    SELECT expert_id, COUNT(*) as decision_count,
                           AVG(processing_time_seconds) as avg_time
                    FROM expert_decisions 
                    WHERE decision_timestamp >= ?
                    GROUP BY expert_id
                    ORDER BY decision_count DESC
                """, (thirty_days_ago.isoformat(),))
                
                workload_distribution = []
                for row in cursor.fetchall():
                    workload_distribution.append({
                        'expert_id': row[0],
                        'decision_count': row[1],
                        'avg_processing_time': row[2]
                    })
                
                # Construct metrics response
                metrics = {
                    'review_volume': {
                        'total_decisions': total_decisions,
                        'current_period_decisions': current_decisions,
                        'previous_period_decisions': previous_decisions,
                        'avg_processing_time_current': avg_time_current,
                        'avg_processing_time_previous': avg_time_previous
                    },
                    'automation_rate': {
                        'current_period': automation_rate,
                        'estimated_improvement': max(0, (avg_time_previous - avg_time_current) / max(avg_time_previous, 1)) if avg_time_previous > 0 else 0.0
                    },
                    'trend_analysis': {
                        'period_comparison': {
                            'current_period_days': 30,
                            'previous_period_days': 30,
                            'current_decisions': current_decisions,
                            'previous_decisions': previous_decisions
                        },
                        'change_percentage': change_percentage,
                        'trend_direction': 'improving' if change_percentage < 0 else 'stable' if change_percentage < 10 else 'increasing'
                    },
                    'workload_distribution': workload_distribution,
                    'generated_at': current_date.isoformat()
                }
                
                return metrics
                
        except Exception as e:
            logger.error(f"Failed to calculate expert review load metrics: {e}")
            return {
                'review_volume': {'total_decisions': 0},
                'automation_rate': {'current_period': 0.0},
                'trend_analysis': {'change_percentage': 0.0},
                'workload_distribution': [],
                'error': str(e)
            }
    
    def _infer_proper_noun_status(self, term: str) -> bool:
        """
        Infer if a term should be treated as a proper noun
        
        Args:
            term: Term to analyze
            
        Returns:
            True if likely a proper noun, False otherwise
        """
        # Simple heuristics for proper noun detection
        if term[0].isupper():
            return True
        
        # Known categories that are typically proper nouns
        proper_noun_indicators = [
            'gita', 'krishna', 'rama', 'shiva', 'brahma', 'vishnu',
            'patanjali', 'shankaracharya', 'vedanta', 'upanishad'
        ]
        
        term_lower = term.lower()
        for indicator in proper_noun_indicators:
            if indicator in term_lower:
                return True
        
        return False
    
    def _infer_term_category(self, term: str) -> str:
        """
        Infer the category of a term based on simple heuristics
        
        Args:
            term: Term to categorize
            
        Returns:
            Inferred category string
        """
        term_lower = term.lower()
        
        # Deity names
        if any(deity in term_lower for deity in ['krishna', 'rama', 'shiva', 'brahma', 'vishnu']):
            return 'deity'
        
        # Scripture names
        if any(scripture in term_lower for scripture in ['gita', 'upanishad', 'veda', 'sutra']):
            return 'scripture'
        
        # Practices
        if any(practice in term_lower for practice in ['yoga', 'meditation', 'pranayama']):
            return 'practice'
        
        # Concepts
        if any(concept in term_lower for concept in ['dharma', 'karma', 'moksha', 'ahimsa']):
            return 'concept'
        
        # People
        if any(person in term_lower for person in ['patanjali', 'shankaracharya', 'acharya']):
            return 'person'
        
        # Mantras
        if any(mantra in term_lower for mantra in ['om', 'aum', 'mantra']):
            return 'mantra'
        
        # Default
        return 'general'
    
    def create_validation_case_from_processing(self, 
                                               original_text: str,
                                               processed_text: str,
                                               processing_context: Dict[str, Any],
                                               flagging_reasons: List[str],
                                               confidence_scores: Dict[str, float]) -> ValidationCase:
        """
        Create a validation case from processing results
        
        This is a helper method to create ValidationCase objects from
        processing pipeline results that need expert review.
        """
        case_id = str(uuid.uuid4())
        
        # Calculate priority score based on confidence and flagging reasons
        priority_score = 0.0
        
        # Lower confidence = higher priority
        avg_confidence = sum(confidence_scores.values()) / len(confidence_scores) if confidence_scores else 0.5
        priority_score += (1.0 - avg_confidence) * 50  # 0-50 points based on confidence
        
        # More flagging reasons = higher priority
        priority_score += len(flagging_reasons) * 10  # 10 points per flagging reason
        
        # Boost priority for certain context types
        if processing_context.get('context_type') == 'scriptural':
            priority_score += 20
        elif processing_context.get('context_type') == 'Sanskrit':
            priority_score += 15
        
        validation_case = ValidationCase(
            case_id=case_id,
            original_text=original_text,
            processed_text=processed_text,
            processing_context=processing_context,
            flagging_reasons=flagging_reasons,
            confidence_scores=confidence_scores,
            timestamp=datetime.now(),
            priority_score=priority_score
        )
        
        return validation_case