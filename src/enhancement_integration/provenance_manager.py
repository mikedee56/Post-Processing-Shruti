"""
Provenance Management System for Story 2.4.4

This module provides Gold/Silver/Bronze classification and weighted confidence
adjustments for lexicon sources and processing components.

Key Features:
- Gold/Silver/Bronze lexicon source classification
- Provenance-weighted confidence adjustments  
- Source authority validation and reporting
- Integration with existing lexicon management
"""

from typing import Dict, List, Optional, Any, Union, Set
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json
import yaml
import logging

from utils.logger_config import get_logger


class ProvenanceLevel(Enum):
    """Provenance classification levels for sources."""
    GOLD = "gold"
    SILVER = "silver"  
    BRONZE = "bronze"
    UNVERIFIED = "unverified"


class SourceType(Enum):
    """Types of sources in the system."""
    LEXICON = "lexicon"
    SCRIPTURE = "scripture"
    ACADEMIC = "academic"
    COMMUNITY = "community"
    AUTOMATED = "automated"
    MANUAL = "manual"


@dataclass
class ProvenanceRecord:
    """Record of source provenance information."""
    source_id: str
    source_name: str
    source_type: SourceType
    provenance_level: ProvenanceLevel
    authority_score: float  # 0.0-1.0
    verification_date: Optional[str] = None
    verification_method: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate provenance record on creation."""
        self.authority_score = max(0.0, min(1.0, self.authority_score))


@dataclass  
class ProvenanceWeightedResult:
    """Result with provenance-weighted confidence adjustment."""
    original_confidence: float
    adjusted_confidence: float
    provenance_weight: float
    provenance_level: ProvenanceLevel
    source_records: List[ProvenanceRecord]
    adjustment_metadata: Dict[str, Any]


class ProvenanceManager:
    """
    Gold/Silver/Bronze provenance classification and confidence weighting system.
    
    This component implements AC6 of Story 2.4.4, providing:
    - Classification of lexicon sources into Gold/Silver/Bronze tiers
    - Provenance-weighted confidence adjustments
    - Source authority validation and reporting
    - Integration with existing lexicon and scripture systems
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the provenance manager.
        
        Args:
            config: Configuration parameters for provenance management
        """
        self.logger = get_logger(__name__)
        self.config = config or {}
        
        # Provenance weight multipliers
        self.provenance_weights = {
            ProvenanceLevel.GOLD: 1.0,      # No adjustment for gold sources
            ProvenanceLevel.SILVER: 0.85,   # Slight reduction for silver
            ProvenanceLevel.BRONZE: 0.7,    # Moderate reduction for bronze
            ProvenanceLevel.UNVERIFIED: 0.5 # Significant reduction for unverified
        }
        
        # Update weights from config
        config_weights = self.config.get('provenance_weights', {})
        for level_str, weight in config_weights.items():
            try:
                level = ProvenanceLevel(level_str)
                self.provenance_weights[level] = weight
            except ValueError:
                self.logger.warning(f"Unknown provenance level in config: {level_str}")
        
        # Source registry
        self.source_registry: Dict[str, ProvenanceRecord] = {}
        
        # Default provenance classifications
        self._initialize_default_classifications()
        
        self.logger.info("Provenance manager initialized")
    
    def _initialize_default_classifications(self) -> None:
        """Initialize default provenance classifications for known sources."""
        default_sources = [
            # Gold tier sources - highest authority
            ProvenanceRecord(
                source_id="gita_press_official",
                source_name="Gita Press Official Publications",
                source_type=SourceType.ACADEMIC,
                provenance_level=ProvenanceLevel.GOLD,
                authority_score=1.0,
                verification_method="academic_review",
                metadata={"publisher": "Gita Press", "established": 1923}
            ),
            ProvenanceRecord(
                source_id="vedic_heritage_portal",
                source_name="Vedic Heritage Portal",
                source_type=SourceType.ACADEMIC,
                provenance_level=ProvenanceLevel.GOLD,
                authority_score=0.95,
                verification_method="scholarly_consensus",
                metadata={"institution": "IIT Kanpur", "peer_reviewed": True}
            ),
            ProvenanceRecord(
                source_id="monier_williams",
                source_name="Monier-Williams Sanskrit Dictionary",
                source_type=SourceType.ACADEMIC,
                provenance_level=ProvenanceLevel.GOLD,
                authority_score=0.95,
                verification_method="classical_reference",
                metadata={"year": 1899, "classic_reference": True}
            ),
            
            # Silver tier sources - good authority
            ProvenanceRecord(
                source_id="wikisource_sanskrit",
                source_name="Wikisource Sanskrit Texts",
                source_type=SourceType.COMMUNITY,
                provenance_level=ProvenanceLevel.SILVER,
                authority_score=0.8,
                verification_method="community_review",
                metadata={"platform": "Wikimedia", "community_edited": True}
            ),
            ProvenanceRecord(
                source_id="sanskrit_web_dictionary",
                source_name="Sanskrit Web Dictionary Project",
                source_type=SourceType.COMMUNITY,
                provenance_level=ProvenanceLevel.SILVER,
                authority_score=0.75,
                verification_method="collaborative_editing",
                metadata={"open_source": True, "user_contributed": True}
            ),
            
            # Bronze tier sources - moderate authority
            ProvenanceRecord(
                source_id="generic_online_dictionary",
                source_name="Generic Online Sanskrit Dictionary",
                source_type=SourceType.AUTOMATED,
                provenance_level=ProvenanceLevel.BRONZE,
                authority_score=0.6,
                verification_method="automated_extraction",
                metadata={"automated": True, "verification_limited": True}
            ),
            
            # Unverified sources - lowest authority
            ProvenanceRecord(
                source_id="unknown_source",
                source_name="Unknown Source",
                source_type=SourceType.AUTOMATED,
                provenance_level=ProvenanceLevel.UNVERIFIED,
                authority_score=0.4,
                verification_method="none",
                metadata={"requires_verification": True}
            )
        ]
        
        for record in default_sources:
            self.source_registry[record.source_id] = record
        
        self.logger.info(f"Initialized {len(default_sources)} default provenance classifications")
    
    def register_source(self, provenance_record: ProvenanceRecord) -> None:
        """
        Register a new source with provenance information.
        
        Args:
            provenance_record: Provenance record for the source
        """
        self.source_registry[provenance_record.source_id] = provenance_record
        
        self.logger.info(
            f"Registered source: {provenance_record.source_name} "
            f"({provenance_record.provenance_level.value})"
        )
    
    def get_source_provenance(self, source_id: str) -> Optional[ProvenanceRecord]:
        """
        Get provenance record for a source.
        
        Args:
            source_id: Identifier of the source
            
        Returns:
            ProvenanceRecord if found, None otherwise
        """
        return self.source_registry.get(source_id)
    
    def classify_source_authority(self, source_id: str) -> ProvenanceLevel:
        """
        Classify the authority level of a source.
        
        Args:
            source_id: Identifier of the source
            
        Returns:
            ProvenanceLevel classification
        """
        record = self.get_source_provenance(source_id)
        if record:
            return record.provenance_level
        else:
            # Unknown sources default to unverified
            return ProvenanceLevel.UNVERIFIED
    
    def calculate_provenance_weight(
        self, 
        source_ids: Union[str, List[str]]
    ) -> float:
        """
        Calculate combined provenance weight for source(s).
        
        Args:
            source_ids: Single source ID or list of source IDs
            
        Returns:
            Combined provenance weight (0.0-1.0)
        """
        if isinstance(source_ids, str):
            source_ids = [source_ids]
        
        if not source_ids:
            return self.provenance_weights[ProvenanceLevel.UNVERIFIED]
        
        weights = []
        for source_id in source_ids:
            record = self.get_source_provenance(source_id)
            if record:
                weight = self.provenance_weights[record.provenance_level]
                # Also factor in authority score
                adjusted_weight = weight * record.authority_score
                weights.append(adjusted_weight)
            else:
                # Unknown source
                weights.append(self.provenance_weights[ProvenanceLevel.UNVERIFIED])
        
        # Use weighted average for multiple sources
        return sum(weights) / len(weights) if weights else 0.5
    
    def apply_provenance_weighting(
        self,
        original_confidence: float,
        source_ids: Union[str, List[str]],
        adjustment_method: str = "multiplicative"
    ) -> ProvenanceWeightedResult:
        """
        Apply provenance-based confidence weighting.
        
        Args:
            original_confidence: Original confidence score
            source_ids: Source ID(s) for provenance lookup
            adjustment_method: How to apply weighting ("multiplicative", "additive", "hybrid")
            
        Returns:
            ProvenanceWeightedResult with adjusted confidence
        """
        if isinstance(source_ids, str):
            source_ids = [source_ids]
        
        # Get provenance records
        source_records = []
        provenance_levels = []
        
        for source_id in source_ids:
            record = self.get_source_provenance(source_id)
            if record:
                source_records.append(record)
                provenance_levels.append(record.provenance_level)
            else:
                # Create placeholder for unknown source
                unknown_record = ProvenanceRecord(
                    source_id=source_id,
                    source_name=f"Unknown Source ({source_id})",
                    source_type=SourceType.AUTOMATED,
                    provenance_level=ProvenanceLevel.UNVERIFIED,
                    authority_score=0.4
                )
                source_records.append(unknown_record)
                provenance_levels.append(ProvenanceLevel.UNVERIFIED)
        
        # Calculate combined provenance weight
        provenance_weight = self.calculate_provenance_weight(source_ids)
        
        # Apply weighting based on method
        if adjustment_method == "multiplicative":
            adjusted_confidence = original_confidence * provenance_weight
        elif adjustment_method == "additive":
            # Add/subtract based on provenance (more conservative)
            adjustment = (provenance_weight - 0.75) * 0.2  # Scale adjustment
            adjusted_confidence = original_confidence + adjustment
        elif adjustment_method == "hybrid":
            # Combination of multiplicative and additive
            multiplicative_adj = original_confidence * provenance_weight
            additive_adj = original_confidence + ((provenance_weight - 0.75) * 0.1)
            adjusted_confidence = (multiplicative_adj + additive_adj) / 2
        else:
            self.logger.warning(f"Unknown adjustment method: {adjustment_method}")
            adjusted_confidence = original_confidence * provenance_weight
        
        # Ensure adjusted confidence stays in valid range
        adjusted_confidence = max(0.0, min(1.0, adjusted_confidence))
        
        # Determine overall provenance level
        if all(level == ProvenanceLevel.GOLD for level in provenance_levels):
            overall_level = ProvenanceLevel.GOLD
        elif all(level in [ProvenanceLevel.GOLD, ProvenanceLevel.SILVER] for level in provenance_levels):
            overall_level = ProvenanceLevel.SILVER
        elif any(level == ProvenanceLevel.UNVERIFIED for level in provenance_levels):
            overall_level = ProvenanceLevel.UNVERIFIED
        else:
            overall_level = ProvenanceLevel.BRONZE
        
        # Create adjustment metadata
        adjustment_metadata = {
            "adjustment_method": adjustment_method,
            "provenance_weight": provenance_weight,
            "confidence_change": adjusted_confidence - original_confidence,
            "source_count": len(source_records),
            "source_breakdown": {
                record.source_id: {
                    "level": record.provenance_level.value,
                    "authority_score": record.authority_score,
                    "weight": self.provenance_weights[record.provenance_level]
                }
                for record in source_records
            }
        }
        
        return ProvenanceWeightedResult(
            original_confidence=original_confidence,
            adjusted_confidence=adjusted_confidence,
            provenance_weight=provenance_weight,
            provenance_level=overall_level,
            source_records=source_records,
            adjustment_metadata=adjustment_metadata
        )
    
    def get_source_statistics(self) -> Dict[str, Any]:
        """Get statistics about registered sources."""
        level_counts = {}
        type_counts = {}
        authority_scores = []
        
        for record in self.source_registry.values():
            # Count by provenance level
            level = record.provenance_level.value
            level_counts[level] = level_counts.get(level, 0) + 1
            
            # Count by source type
            source_type = record.source_type.value
            type_counts[source_type] = type_counts.get(source_type, 0) + 1
            
            # Collect authority scores
            authority_scores.append(record.authority_score)
        
        avg_authority = sum(authority_scores) / len(authority_scores) if authority_scores else 0.0
        
        return {
            "total_sources": len(self.source_registry),
            "provenance_breakdown": level_counts,
            "source_type_breakdown": type_counts,
            "average_authority_score": avg_authority,
            "provenance_weights": {
                level.value: weight 
                for level, weight in self.provenance_weights.items()
            }
        }
    
    def export_provenance_registry(self, file_path: Path) -> None:
        """
        Export provenance registry to file.
        
        Args:
            file_path: Path to export file
        """
        try:
            registry_data = {
                "version": "2.4.4",
                "export_timestamp": "",
                "provenance_weights": {
                    level.value: weight 
                    for level, weight in self.provenance_weights.items()
                },
                "sources": {
                    source_id: {
                        "source_name": record.source_name,
                        "source_type": record.source_type.value,
                        "provenance_level": record.provenance_level.value,
                        "authority_score": record.authority_score,
                        "verification_date": record.verification_date,
                        "verification_method": record.verification_method,
                        "metadata": record.metadata
                    }
                    for source_id, record in self.source_registry.items()
                }
            }
            
            if file_path.suffix.lower() == '.json':
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(registry_data, f, indent=2, ensure_ascii=False)
            elif file_path.suffix.lower() in ['.yaml', '.yml']:
                with open(file_path, 'w', encoding='utf-8') as f:
                    yaml.dump(registry_data, f, default_flow_style=False, allow_unicode=True)
            else:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")
            
            self.logger.info(f"Exported provenance registry to {file_path}")
            
        except Exception as e:
            self.logger.error(f"Error exporting provenance registry: {e}")
            raise
    
    def import_provenance_registry(self, file_path: Path) -> None:
        """
        Import provenance registry from file.
        
        Args:
            file_path: Path to import file
        """
        try:
            if file_path.suffix.lower() == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    registry_data = json.load(f)
            elif file_path.suffix.lower() in ['.yaml', '.yml']:
                with open(file_path, 'r', encoding='utf-8') as f:
                    registry_data = yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")
            
            # Import sources
            sources_data = registry_data.get("sources", {})
            imported_count = 0
            
            for source_id, source_info in sources_data.items():
                record = ProvenanceRecord(
                    source_id=source_id,
                    source_name=source_info["source_name"],
                    source_type=SourceType(source_info["source_type"]),
                    provenance_level=ProvenanceLevel(source_info["provenance_level"]),
                    authority_score=source_info["authority_score"],
                    verification_date=source_info.get("verification_date"),
                    verification_method=source_info.get("verification_method"),
                    metadata=source_info.get("metadata", {})
                )
                
                self.source_registry[source_id] = record
                imported_count += 1
            
            # Update weights if provided
            weights_data = registry_data.get("provenance_weights", {})
            for level_str, weight in weights_data.items():
                try:
                    level = ProvenanceLevel(level_str)
                    self.provenance_weights[level] = weight
                except ValueError:
                    self.logger.warning(f"Unknown provenance level in import: {level_str}")
            
            self.logger.info(f"Imported {imported_count} sources from {file_path}")
            
        except Exception as e:
            self.logger.error(f"Error importing provenance registry: {e}")
            raise
    
    def validate_configuration(self) -> Dict[str, Any]:
        """Validate provenance manager configuration."""
        validation = {
            "is_valid": True,
            "warnings": [],
            "errors": []
        }
        
        # Check provenance weights
        for level, weight in self.provenance_weights.items():
            if weight < 0 or weight > 1:
                validation["errors"].append(f"Invalid weight for {level.value}: {weight}")
                validation["is_valid"] = False
        
        # Check source registry
        if not self.source_registry:
            validation["warnings"].append("No sources registered in provenance manager")
        
        # Validate individual source records
        for source_id, record in self.source_registry.items():
            if not (0.0 <= record.authority_score <= 1.0):
                validation["errors"].append(f"Invalid authority score for {source_id}: {record.authority_score}")
                validation["is_valid"] = False
        
        return validation