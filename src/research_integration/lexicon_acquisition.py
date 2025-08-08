"""
Lexicon Acquisition System

Provides multi-source lexicon building capabilities with quality assessment 
and automated validation workflows for continuous improvement.
"""

import json
import csv
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any, Union
from pathlib import Path
from enum import Enum
import time
import hashlib
import re
from urllib.parse import urlparse

from src.utils.logger_config import get_logger
from src.enhancement_integration.provenance_manager import ProvenanceManager, ProvenanceLevel

logger = get_logger(__name__)


class LexiconSourceType(Enum):
    """Types of lexicon sources"""
    SCHOLARLY_DATABASE = "scholarly_database"
    ACADEMIC_PUBLICATION = "academic_publication" 
    MANUSCRIPT_DIGITIZATION = "manuscript_digitization"
    CROWD_SOURCED = "crowd_sourced"
    LEGACY_SYSTEM = "legacy_system"
    API_ENDPOINT = "api_endpoint"


class LexiconEntryQuality(Enum):
    """Quality levels for lexicon entries"""
    VERIFIED = "verified"
    RELIABLE = "reliable"
    PROVISIONAL = "provisional"
    QUESTIONABLE = "questionable"


@dataclass
class LexiconSource:
    """Information about a lexicon data source"""
    source_id: str
    name: str
    source_type: LexiconSourceType
    authority_level: ProvenanceLevel
    url: Optional[str] = None
    description: Optional[str] = None
    last_updated: Optional[float] = None
    entry_count: int = 0
    quality_score: float = 0.0
    access_method: str = "manual"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LexiconEntry:
    """Individual lexicon entry with quality and provenance information"""
    term: str
    transliteration: str
    variations: List[str] = field(default_factory=list)
    category: str = "general"
    is_proper_noun: bool = False
    confidence: float = 1.0
    source_id: str = "unknown"
    quality_level: LexiconEntryQuality = LexiconEntryQuality.PROVISIONAL
    verification_count: int = 0
    last_verified: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_hash(self) -> str:
        """Generate unique hash for the entry"""
        content = f"{self.term}:{self.transliteration}:{self.category}"
        return hashlib.md5(content.encode('utf-8')).hexdigest()


@dataclass
class QualityAssessmentResult:
    """Result of lexicon quality assessment"""
    total_entries: int
    verified_entries: int
    reliable_entries: int
    provisional_entries: int
    questionable_entries: int
    overall_quality_score: float
    source_quality_scores: Dict[str, float] = field(default_factory=dict)
    quality_issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class AcquisitionReport:
    """Report of lexicon acquisition operation"""
    operation_type: str
    source_id: str
    entries_processed: int
    entries_added: int
    entries_updated: int
    entries_rejected: int
    processing_time: float
    quality_assessment: QualityAssessmentResult
    errors_encountered: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class LexiconAcquisition:
    """
    Multi-source lexicon acquisition system with quality assessment and validation.
    
    Provides automated lexicon building, quality classification, and continuous
    improvement workflows.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")
        
        # Initialize provenance manager for source classification
        self.provenance_manager = ProvenanceManager()
        
        # Lexicon storage
        self.sources: Dict[str, LexiconSource] = {}
        self.entries: Dict[str, LexiconEntry] = {}
        
        # Quality thresholds
        self.quality_thresholds = {
            'verified_confidence': 0.9,
            'reliable_confidence': 0.7,
            'provisional_confidence': 0.5,
            'min_verification_count': 2
        }
        
        # Load existing sources and entries
        self._load_existing_data()
    
    def _load_existing_data(self) -> None:
        """Load existing lexicon sources and entries"""
        try:
            # Load from enhancement_integration lexicon manager if available
            from src.sanskrit_hindi_identifier.lexicon_manager import LexiconManager
            lexicon_manager = LexiconManager()
            
            existing_entries = lexicon_manager.get_all_entries()
            
            for term, entry_data in existing_entries.items():
                lexicon_entry = LexiconEntry(
                    term=term,
                    transliteration=entry_data.transliteration,
                    variations=entry_data.variations,
                    category=entry_data.category,
                    is_proper_noun=entry_data.is_proper_noun,
                    confidence=entry_data.confidence,
                    source_id="existing_system",
                    quality_level=self._determine_quality_level(entry_data.confidence)
                )
                
                self.entries[lexicon_entry.get_hash()] = lexicon_entry
            
            self.logger.info(f"Loaded {len(self.entries)} existing lexicon entries")
            
        except Exception as e:
            self.logger.warning(f"Could not load existing lexicon data: {e}")
    
    def _determine_quality_level(self, confidence: float) -> LexiconEntryQuality:
        """Determine quality level based on confidence score"""
        if confidence >= self.quality_thresholds['verified_confidence']:
            return LexiconEntryQuality.VERIFIED
        elif confidence >= self.quality_thresholds['reliable_confidence']:
            return LexiconEntryQuality.RELIABLE
        elif confidence >= self.quality_thresholds['provisional_confidence']:
            return LexiconEntryQuality.PROVISIONAL
        else:
            return LexiconEntryQuality.QUESTIONABLE
    
    def register_source(self, source: LexiconSource) -> None:
        """Register a new lexicon source"""
        self.sources[source.source_id] = source
        self.logger.info(f"Registered lexicon source: {source.name} ({source.source_type.value})")
    
    def acquire_from_json_file(self, file_path: Path, source_id: str) -> AcquisitionReport:
        """
        Acquire lexicon entries from JSON file.
        
        Args:
            file_path: Path to JSON file containing lexicon data
            source_id: ID of the registered source
        
        Returns:
            Acquisition operation report
        """
        start_time = time.time()
        self.logger.info(f"Starting JSON lexicon acquisition from {file_path}")
        
        if source_id not in self.sources:
            raise ValueError(f"Source {source_id} not registered")
        
        source = self.sources[source_id]
        entries_processed = 0
        entries_added = 0
        entries_updated = 0
        entries_rejected = 0
        errors = []
        warnings = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle different JSON structures
            if isinstance(data, dict):
                if 'entries' in data:
                    lexicon_data = data['entries']
                else:
                    lexicon_data = data
            elif isinstance(data, list):
                lexicon_data = {f"entry_{i}": entry for i, entry in enumerate(data)}
            else:
                raise ValueError("Unsupported JSON structure")
            
            for key, entry_data in lexicon_data.items():
                entries_processed += 1
                
                try:
                    # Parse entry data
                    entry = self._parse_entry_data(entry_data, source_id)
                    
                    # Quality validation
                    if self._validate_entry_quality(entry):
                        entry_hash = entry.get_hash()
                        
                        if entry_hash in self.entries:
                            # Update existing entry
                            self._merge_entry_data(self.entries[entry_hash], entry)
                            entries_updated += 1
                        else:
                            # Add new entry
                            self.entries[entry_hash] = entry
                            entries_added += 1
                    else:
                        entries_rejected += 1
                        warnings.append(f"Quality validation failed for entry: {key}")
                        
                except Exception as e:
                    entries_rejected += 1
                    errors.append(f"Failed to process entry {key}: {str(e)}")
        
        except Exception as e:
            errors.append(f"Failed to read JSON file: {str(e)}")
        
        processing_time = time.time() - start_time
        
        # Update source statistics
        source.entry_count = entries_added + entries_updated
        source.last_updated = time.time()
        
        # Perform quality assessment
        quality_assessment = self.assess_lexicon_quality()
        
        return AcquisitionReport(
            operation_type="json_file_acquisition",
            source_id=source_id,
            entries_processed=entries_processed,
            entries_added=entries_added,
            entries_updated=entries_updated,
            entries_rejected=entries_rejected,
            processing_time=processing_time,
            quality_assessment=quality_assessment,
            errors_encountered=errors,
            warnings=warnings
        )
    
    def acquire_from_csv_file(self, file_path: Path, source_id: str, 
                             field_mapping: Dict[str, str]) -> AcquisitionReport:
        """
        Acquire lexicon entries from CSV file.
        
        Args:
            file_path: Path to CSV file
            source_id: ID of the registered source
            field_mapping: Mapping of CSV columns to entry fields
        
        Returns:
            Acquisition operation report
        """
        start_time = time.time()
        self.logger.info(f"Starting CSV lexicon acquisition from {file_path}")
        
        if source_id not in self.sources:
            raise ValueError(f"Source {source_id} not registered")
        
        source = self.sources[source_id]
        entries_processed = 0
        entries_added = 0
        entries_updated = 0
        entries_rejected = 0
        errors = []
        warnings = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                
                for row_num, row in enumerate(reader, 1):
                    entries_processed += 1
                    
                    try:
                        # Map CSV fields to entry data
                        entry_data = {}
                        for csv_field, entry_field in field_mapping.items():
                            if csv_field in row:
                                entry_data[entry_field] = row[csv_field]
                        
                        # Create lexicon entry
                        entry = self._parse_entry_data(entry_data, source_id)
                        
                        # Quality validation
                        if self._validate_entry_quality(entry):
                            entry_hash = entry.get_hash()
                            
                            if entry_hash in self.entries:
                                self._merge_entry_data(self.entries[entry_hash], entry)
                                entries_updated += 1
                            else:
                                self.entries[entry_hash] = entry
                                entries_added += 1
                        else:
                            entries_rejected += 1
                            warnings.append(f"Quality validation failed for row {row_num}")
                            
                    except Exception as e:
                        entries_rejected += 1
                        errors.append(f"Failed to process row {row_num}: {str(e)}")
        
        except Exception as e:
            errors.append(f"Failed to read CSV file: {str(e)}")
        
        processing_time = time.time() - start_time
        
        # Update source statistics
        source.entry_count = entries_added + entries_updated
        source.last_updated = time.time()
        
        # Perform quality assessment
        quality_assessment = self.assess_lexicon_quality()
        
        return AcquisitionReport(
            operation_type="csv_file_acquisition",
            source_id=source_id,
            entries_processed=entries_processed,
            entries_added=entries_added,
            entries_updated=entries_updated,
            entries_rejected=entries_rejected,
            processing_time=processing_time,
            quality_assessment=quality_assessment,
            errors_encountered=errors,
            warnings=warnings
        )
    
    def _parse_entry_data(self, entry_data: Dict[str, Any], source_id: str) -> LexiconEntry:
        """Parse raw entry data into LexiconEntry object"""
        # Extract required fields
        term = entry_data.get('term') or entry_data.get('word') or entry_data.get('original')
        if not term:
            raise ValueError("Missing required 'term' field")
        
        transliteration = entry_data.get('transliteration') or entry_data.get('iast') or term
        
        # Extract optional fields
        variations = entry_data.get('variations', [])
        if isinstance(variations, str):
            variations = [v.strip() for v in variations.split(',')]
        
        category = entry_data.get('category', 'general')
        is_proper_noun = entry_data.get('is_proper_noun', False)
        if isinstance(is_proper_noun, str):
            is_proper_noun = is_proper_noun.lower() in ['true', '1', 'yes']
        
        confidence = float(entry_data.get('confidence', 0.7))
        
        # Create entry
        entry = LexiconEntry(
            term=term,
            transliteration=transliteration,
            variations=variations,
            category=category,
            is_proper_noun=is_proper_noun,
            confidence=confidence,
            source_id=source_id,
            quality_level=self._determine_quality_level(confidence)
        )
        
        return entry
    
    def _validate_entry_quality(self, entry: LexiconEntry) -> bool:
        """Validate lexicon entry quality"""
        # Basic validation checks
        if not entry.term or len(entry.term.strip()) == 0:
            return False
        
        if not entry.transliteration or len(entry.transliteration.strip()) == 0:
            return False
        
        # Check for reasonable confidence score
        if entry.confidence < 0.0 or entry.confidence > 1.0:
            return False
        
        # Check for suspicious patterns
        if re.search(r'[<>{}()[\]]', entry.term):
            return False
        
        return True
    
    def _merge_entry_data(self, existing_entry: LexiconEntry, new_entry: LexiconEntry) -> None:
        """Merge new entry data into existing entry"""
        # Increase verification count
        existing_entry.verification_count += 1
        existing_entry.last_verified = time.time()
        
        # Update confidence with weighted average
        weight = 0.3  # Weight for new data
        existing_entry.confidence = (
            existing_entry.confidence * (1 - weight) + 
            new_entry.confidence * weight
        )
        
        # Merge variations
        for variation in new_entry.variations:
            if variation not in existing_entry.variations:
                existing_entry.variations.append(variation)
        
        # Update quality level
        existing_entry.quality_level = self._determine_quality_level(existing_entry.confidence)
        
        # Update metadata
        existing_entry.metadata[f'merge_{new_entry.source_id}'] = time.time()
    
    def assess_lexicon_quality(self) -> QualityAssessmentResult:
        """
        Assess overall lexicon quality and generate recommendations.
        
        Returns:
            Comprehensive quality assessment result
        """
        total_entries = len(self.entries)
        if total_entries == 0:
            return QualityAssessmentResult(
                total_entries=0,
                verified_entries=0,
                reliable_entries=0,
                provisional_entries=0,
                questionable_entries=0,
                overall_quality_score=0.0
            )
        
        # Count entries by quality level
        quality_counts = {
            LexiconEntryQuality.VERIFIED: 0,
            LexiconEntryQuality.RELIABLE: 0,
            LexiconEntryQuality.PROVISIONAL: 0,
            LexiconEntryQuality.QUESTIONABLE: 0
        }
        
        source_quality_scores = {}
        source_counts = {}
        
        for entry in self.entries.values():
            quality_counts[entry.quality_level] += 1
            
            # Track source quality
            if entry.source_id not in source_quality_scores:
                source_quality_scores[entry.source_id] = []
                source_counts[entry.source_id] = 0
            
            source_quality_scores[entry.source_id].append(entry.confidence)
            source_counts[entry.source_id] += 1
        
        # Calculate source averages
        for source_id in source_quality_scores:
            source_quality_scores[source_id] = sum(source_quality_scores[source_id]) / len(source_quality_scores[source_id])
        
        # Calculate overall quality score
        weighted_score = (
            quality_counts[LexiconEntryQuality.VERIFIED] * 1.0 +
            quality_counts[LexiconEntryQuality.RELIABLE] * 0.8 +
            quality_counts[LexiconEntryQuality.PROVISIONAL] * 0.6 +
            quality_counts[LexiconEntryQuality.QUESTIONABLE] * 0.3
        ) / total_entries
        
        # Generate quality issues
        quality_issues = []
        recommendations = []
        
        questionable_ratio = quality_counts[LexiconEntryQuality.QUESTIONABLE] / total_entries
        if questionable_ratio > 0.2:
            quality_issues.append(f"{questionable_ratio:.1%} of entries are questionable quality")
            recommendations.append("Review and improve questionable entries")
        
        verified_ratio = quality_counts[LexiconEntryQuality.VERIFIED] / total_entries
        if verified_ratio < 0.3:
            quality_issues.append(f"Only {verified_ratio:.1%} of entries are verified")
            recommendations.append("Increase verification efforts for higher quality")
        
        return QualityAssessmentResult(
            total_entries=total_entries,
            verified_entries=quality_counts[LexiconEntryQuality.VERIFIED],
            reliable_entries=quality_counts[LexiconEntryQuality.RELIABLE],
            provisional_entries=quality_counts[LexiconEntryQuality.PROVISIONAL],
            questionable_entries=quality_counts[LexiconEntryQuality.QUESTIONABLE],
            overall_quality_score=weighted_score,
            source_quality_scores=source_quality_scores,
            quality_issues=quality_issues,
            recommendations=recommendations
        )
    
    def export_lexicon(self, output_path: Path, format: str = "json") -> None:
        """
        Export lexicon to file.
        
        Args:
            output_path: Output file path
            format: Export format ('json' or 'csv')
        """
        try:
            if format.lower() == "json":
                self._export_json_lexicon(output_path)
            elif format.lower() == "csv":
                self._export_csv_lexicon(output_path)
            else:
                raise ValueError(f"Unsupported export format: {format}")
            
            self.logger.info(f"Lexicon exported to {output_path} ({format} format)")
            
        except Exception as e:
            self.logger.error(f"Failed to export lexicon: {e}")
            raise
    
    def _export_json_lexicon(self, output_path: Path) -> None:
        """Export lexicon to JSON format"""
        lexicon_data = {
            'metadata': {
                'export_timestamp': time.time(),
                'total_entries': len(self.entries),
                'sources': {sid: {
                    'name': source.name,
                    'type': source.source_type.value,
                    'authority_level': source.authority_level.value,
                    'entry_count': source.entry_count
                } for sid, source in self.sources.items()}
            },
            'entries': {}
        }
        
        for entry_hash, entry in self.entries.items():
            lexicon_data['entries'][entry.term] = {
                'transliteration': entry.transliteration,
                'variations': entry.variations,
                'category': entry.category,
                'is_proper_noun': entry.is_proper_noun,
                'confidence': entry.confidence,
                'source_id': entry.source_id,
                'quality_level': entry.quality_level.value,
                'verification_count': entry.verification_count
            }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(lexicon_data, f, indent=2, ensure_ascii=False)
    
    def _export_csv_lexicon(self, output_path: Path) -> None:
        """Export lexicon to CSV format"""
        fieldnames = [
            'term', 'transliteration', 'variations', 'category', 'is_proper_noun',
            'confidence', 'source_id', 'quality_level', 'verification_count'
        ]
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for entry in self.entries.values():
                writer.writerow({
                    'term': entry.term,
                    'transliteration': entry.transliteration,
                    'variations': ','.join(entry.variations),
                    'category': entry.category,
                    'is_proper_noun': entry.is_proper_noun,
                    'confidence': entry.confidence,
                    'source_id': entry.source_id,
                    'quality_level': entry.quality_level.value,
                    'verification_count': entry.verification_count
                })
    
    def get_acquisition_statistics(self) -> Dict[str, Any]:
        """Get comprehensive acquisition statistics"""
        quality_assessment = self.assess_lexicon_quality()
        
        return {
            'total_sources': len(self.sources),
            'total_entries': len(self.entries),
            'quality_assessment': quality_assessment,
            'source_breakdown': {
                source_id: {
                    'name': source.name,
                    'type': source.source_type.value,
                    'authority_level': source.authority_level.value,
                    'entry_count': source.entry_count,
                    'quality_score': source.quality_score
                }
                for source_id, source in self.sources.items()
            }
        }