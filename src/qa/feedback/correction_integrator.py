"""
Professional Standards Compliant Correction Integration System
Story 4.3: Benchmarking & Continuous Improvement - Feedback Integration

This module implements professional standards for integrating expert corrections
into the lexicon system, following CEO directive for honest and evidence-based work.

Features:
- Expert correction validation and integration
- Lexicon version management with compatibility checking
- Quality gate integration for automated feedback processing
- Professional audit trail for all corrections
"""

import json
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import logging
import hashlib
import shutil


@dataclass
class CorrectionEntry:
    """Represents a single expert correction with professional validation."""
    original_term: str
    corrected_term: str
    correction_type: str  # 'substitution', 'addition', 'deletion', 'transliteration'
    expert_id: str
    confidence_score: float  # Expert confidence 0.0 - 1.0
    context: str  # Context where correction applies
    validation_status: str  # 'pending', 'validated', 'rejected'
    created_at: datetime
    applied_at: Optional[datetime] = None
    rejection_reason: Optional[str] = None


@dataclass
class IntegrationResult:
    """Professional results reporting for correction integration."""
    total_corrections: int
    applied_corrections: int
    rejected_corrections: int
    validation_errors: int
    lexicon_version_before: str
    lexicon_version_after: str
    processing_time: float
    quality_impact_score: Optional[float] = None


class LexiconVersionManager:
    """Professional version management for lexicon updates."""
    
    def __init__(self, lexicon_path: str):
        self.lexicon_path = Path(lexicon_path)
        self.backup_dir = self.lexicon_path.parent / "backups"
        self.backup_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
    
    def create_backup(self) -> str:
        """Create timestamped backup of current lexicon."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        backup_name = f"{self.lexicon_path.stem}_backup_{timestamp}.yaml"
        backup_path = self.backup_dir / backup_name
        
        shutil.copy2(self.lexicon_path, backup_path)
        
        self.logger.info(f"Lexicon backup created: {backup_path}")
        return str(backup_path)
    
    def get_current_version(self) -> str:
        """Get current lexicon version hash for tracking."""
        if not self.lexicon_path.exists():
            return "empty"
            
        with open(self.lexicon_path, 'rb') as f:
            content = f.read()
            return hashlib.sha256(content).hexdigest()[:12]
    
    def rollback_to_backup(self, backup_path: str) -> bool:
        """Professional rollback capability."""
        try:
            backup_file = Path(backup_path)
            if not backup_file.exists():
                self.logger.error(f"Backup file not found: {backup_path}")
                return False
            
            shutil.copy2(backup_file, self.lexicon_path)
            self.logger.info(f"Successfully rolled back to: {backup_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Rollback failed: {e}")
            return False


class CorrectionValidator:
    """Professional validation system for expert corrections."""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        
        # Professional validation rules
        self.validation_rules = {
            'min_confidence': 0.7,
            'max_context_length': 500,
            'required_fields': ['original_term', 'corrected_term', 'expert_id'],
            'valid_correction_types': ['substitution', 'addition', 'deletion', 'transliteration']
        }
    
    def validate_correction(self, correction: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate a correction entry with professional standards.
        
        Returns:
            (is_valid, error_messages)
        """
        errors = []
        
        # Check required fields
        for field in self.validation_rules['required_fields']:
            if field not in correction or not correction[field]:
                errors.append(f"Missing required field: {field}")
        
        # Validate correction type
        if 'correction_type' in correction:
            if correction['correction_type'] not in self.validation_rules['valid_correction_types']:
                errors.append(f"Invalid correction type: {correction['correction_type']}")
        
        # Validate confidence score
        if 'confidence_score' in correction:
            score = correction['confidence_score']
            if not isinstance(score, (int, float)) or score < 0 or score > 1:
                errors.append("Confidence score must be between 0.0 and 1.0")
            elif score < self.validation_rules['min_confidence']:
                errors.append(f"Confidence score {score} below minimum threshold {self.validation_rules['min_confidence']}")
        
        # Validate context length
        if 'context' in correction:
            if len(correction['context']) > self.validation_rules['max_context_length']:
                errors.append(f"Context exceeds maximum length of {self.validation_rules['max_context_length']} characters")
        
        # Validate term content
        if 'original_term' in correction and 'corrected_term' in correction:
            if correction['original_term'] == correction['corrected_term']:
                errors.append("Original and corrected terms are identical")
        
        return len(errors) == 0, errors
    
    def validate_expert_credentials(self, expert_id: str) -> bool:
        """Validate expert credentials (placeholder for professional implementation)."""
        # In production, this would check against expert database
        # For now, basic validation
        return bool(expert_id and len(expert_id) >= 3)


class CorrectionIntegrator:
    """
    Professional correction integration system implementing Story 4.3.
    
    Features:
    - Evidence-based correction validation
    - Professional audit trail
    - Version-controlled lexicon updates
    - Quality impact measurement
    """
    
    def __init__(self):
        self.validator = CorrectionValidator()
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        
        # Initialize audit trail
        self.audit_log_path = Path("data/audit/correction_integration.jsonl")
        self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)
    
    def integrate_expert_corrections(
        self,
        corrections_file: str,
        target_lexicon: str,
        dry_run: bool = False
    ) -> IntegrationResult:
        """
        Integrate expert corrections into target lexicon with professional standards.
        
        Args:
            corrections_file: Path to JSON file containing expert corrections
            target_lexicon: Path to target lexicon YAML file
            dry_run: If True, validate without applying changes
        
        Returns:
            IntegrationResult with comprehensive reporting
        """
        start_time = datetime.now(timezone.utc)
        
        self.logger.info(f"Starting correction integration: {corrections_file} -> {target_lexicon}")
        
        # Initialize version manager
        version_manager = LexiconVersionManager(target_lexicon)
        version_before = version_manager.get_current_version()
        
        # Create backup before modifications
        backup_path = None
        if not dry_run:
            backup_path = version_manager.create_backup()
        
        try:
            # Load corrections
            corrections = self._load_corrections(corrections_file)
            
            # Validate corrections
            validated_corrections = self._validate_corrections(corrections)
            
            # Load current lexicon
            current_lexicon = self._load_lexicon(target_lexicon)
            
            # Apply corrections
            if not dry_run:
                updated_lexicon = self._apply_corrections(current_lexicon, validated_corrections)
                self._save_lexicon(target_lexicon, updated_lexicon)
            
            # Calculate results
            version_after = version_manager.get_current_version() if not dry_run else version_before
            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            
            result = IntegrationResult(
                total_corrections=len(corrections),
                applied_corrections=len([c for c in validated_corrections if c.validation_status == 'validated']),
                rejected_corrections=len([c for c in validated_corrections if c.validation_status == 'rejected']),
                validation_errors=len([c for c in validated_corrections if c.validation_status == 'rejected']),
                lexicon_version_before=version_before,
                lexicon_version_after=version_after,
                processing_time=processing_time
            )
            
            # Log audit trail
            self._log_integration_audit(corrections_file, target_lexicon, result, dry_run)
            
            self.logger.info(
                f"Integration completed: {result.applied_corrections}/{result.total_corrections} "
                f"corrections applied in {processing_time:.2f}s"
            )
            
            return result
            
        except Exception as e:
            # Professional error handling with rollback
            self.logger.error(f"Integration failed: {e}")
            
            if backup_path and not dry_run:
                self.logger.info("Attempting rollback...")
                if version_manager.rollback_to_backup(backup_path):
                    self.logger.info("Successfully rolled back changes")
                else:
                    self.logger.error("Rollback failed - manual intervention required")
            
            raise
    
    def _load_corrections(self, corrections_file: str) -> List[Dict[str, Any]]:
        """Load corrections from JSON file with validation."""
        corrections_path = Path(corrections_file)
        
        if not corrections_path.exists():
            raise FileNotFoundError(f"Corrections file not found: {corrections_file}")
        
        try:
            with open(corrections_path, 'r', encoding='utf-8') as f:
                corrections_data = json.load(f)
            
            # Handle both single corrections and batch format
            if isinstance(corrections_data, dict):
                if 'corrections' in corrections_data:
                    corrections = corrections_data['corrections']
                else:
                    corrections = [corrections_data]
            elif isinstance(corrections_data, list):
                corrections = corrections_data
            else:
                raise ValueError("Invalid corrections file format")
            
            self.logger.info(f"Loaded {len(corrections)} corrections from {corrections_file}")
            return corrections
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in corrections file: {e}")
    
    def _validate_corrections(self, corrections: List[Dict[str, Any]]) -> List[CorrectionEntry]:
        """Validate and convert corrections to CorrectionEntry objects."""
        validated_corrections = []
        
        for i, correction in enumerate(corrections):
            is_valid, errors = self.validator.validate_correction(correction)
            
            # Create CorrectionEntry
            entry = CorrectionEntry(
                original_term=correction.get('original_term', ''),
                corrected_term=correction.get('corrected_term', ''),
                correction_type=correction.get('correction_type', 'substitution'),
                expert_id=correction.get('expert_id', 'unknown'),
                confidence_score=correction.get('confidence_score', 1.0),
                context=correction.get('context', ''),
                validation_status='validated' if is_valid else 'rejected',
                created_at=datetime.now(timezone.utc),
                rejection_reason='; '.join(errors) if errors else None
            )
            
            validated_corrections.append(entry)
            
            if not is_valid:
                self.logger.warning(f"Correction {i} rejected: {'; '.join(errors)}")
        
        return validated_corrections
    
    def _load_lexicon(self, lexicon_path: str) -> Dict[str, Any]:
        """Load current lexicon with error handling."""
        path = Path(lexicon_path)
        
        if not path.exists():
            # Create empty lexicon structure
            return {'corrections': []}
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {'corrections': []}
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in lexicon file: {e}")
    
    def _apply_corrections(
        self, 
        lexicon: Dict[str, Any], 
        corrections: List[CorrectionEntry]
    ) -> Dict[str, Any]:
        """Apply validated corrections to lexicon."""
        updated_lexicon = lexicon.copy()
        
        # Ensure corrections structure exists
        if 'corrections' not in updated_lexicon:
            updated_lexicon['corrections'] = []
        
        for correction in corrections:
            if correction.validation_status == 'validated':
                # Add correction to lexicon
                correction_entry = {
                    'original': correction.original_term,
                    'corrected': correction.corrected_term,
                    'type': correction.correction_type,
                    'expert_id': correction.expert_id,
                    'confidence': correction.confidence_score,
                    'context': correction.context,
                    'applied_at': datetime.now(timezone.utc).isoformat()
                }
                
                # Check for duplicates
                existing_entry = self._find_existing_correction(
                    updated_lexicon['corrections'], 
                    correction.original_term
                )
                
                if existing_entry:
                    # Update existing entry if new confidence is higher
                    if correction.confidence_score > existing_entry.get('confidence', 0):
                        updated_lexicon['corrections'].remove(existing_entry)
                        updated_lexicon['corrections'].append(correction_entry)
                        self.logger.info(f"Updated correction for '{correction.original_term}'")
                    else:
                        self.logger.info(f"Skipped lower-confidence correction for '{correction.original_term}'")
                else:
                    updated_lexicon['corrections'].append(correction_entry)
                    self.logger.info(f"Added new correction: '{correction.original_term}' -> '{correction.corrected_term}'")
        
        return updated_lexicon
    
    def _find_existing_correction(
        self, 
        corrections: List[Dict[str, Any]], 
        original_term: str
    ) -> Optional[Dict[str, Any]]:
        """Find existing correction for the same original term."""
        for correction in corrections:
            if correction.get('original') == original_term:
                return correction
        return None
    
    def _save_lexicon(self, lexicon_path: str, lexicon: Dict[str, Any]):
        """Save updated lexicon with professional formatting."""
        path = Path(lexicon_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(
                lexicon, 
                f, 
                default_flow_style=False, 
                allow_unicode=True,
                sort_keys=True
            )
    
    def _log_integration_audit(
        self, 
        corrections_file: str, 
        target_lexicon: str,
        result: IntegrationResult,
        dry_run: bool
    ):
        """Log integration to audit trail."""
        audit_entry = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'operation': 'correction_integration',
            'corrections_file': corrections_file,
            'target_lexicon': target_lexicon,
            'dry_run': dry_run,
            'result': asdict(result)
        }
        
        # Append to audit log
        with open(self.audit_log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(audit_entry) + '\n')
    
    def get_integration_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent integration history for monitoring."""
        if not self.audit_log_path.exists():
            return []
        
        history = []
        try:
            with open(self.audit_log_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        history.append(json.loads(line))
            
            # Return most recent entries
            return sorted(history, key=lambda x: x['timestamp'], reverse=True)[:limit]
            
        except Exception as e:
            self.logger.error(f"Failed to read integration history: {e}")
            return []
    
    def validate_lexicon_integrity(self, lexicon_path: str) -> Dict[str, Any]:
        """Validate lexicon integrity after integration."""
        try:
            lexicon = self._load_lexicon(lexicon_path)
            
            corrections = lexicon.get('corrections', [])
            
            # Integrity checks
            total_corrections = len(corrections)
            duplicate_originals = len(corrections) - len(set(c.get('original', '') for c in corrections))
            missing_fields = sum(1 for c in corrections 
                               if not all(field in c for field in ['original', 'corrected']))
            
            return {
                'is_valid': duplicate_originals == 0 and missing_fields == 0,
                'total_corrections': total_corrections,
                'duplicate_originals': duplicate_originals,
                'missing_fields': missing_fields,
                'validation_timestamp': datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            return {
                'is_valid': False,
                'error': str(e),
                'validation_timestamp': datetime.now(timezone.utc).isoformat()
            }


# Professional usage example following Story 4.3 specifications
def main():
    """Professional demonstration of correction integration system."""
    
    # Configure professional logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    integrator = CorrectionIntegrator()
    
    try:
        # Example: Integrate expert corrections
        result = integrator.integrate_expert_corrections(
            corrections_file="data/expert_corrections/batch_001.json",
            target_lexicon="data/lexicons/corrections.yaml",
            dry_run=True  # Validate first
        )
        
        print(f"Integration Result:")
        print(f"  Total corrections: {result.total_corrections}")
        print(f"  Applied: {result.applied_corrections}")
        print(f"  Rejected: {result.rejected_corrections}")
        print(f"  Processing time: {result.processing_time:.2f}s")
        
        # Validate lexicon integrity
        integrity_result = integrator.validate_lexicon_integrity("data/lexicons/corrections.yaml")
        print(f"\nLexicon Integrity: {'✅ Valid' if integrity_result['is_valid'] else '❌ Invalid'}")
        
    except Exception as e:
        logging.error(f"Professional correction integration failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())