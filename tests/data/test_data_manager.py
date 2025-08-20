"""
Test Data Management System for Story 5.5: Testing & Quality Assurance Framework

This module provides comprehensive test data management capabilities including:
- Golden dataset validation and management
- Synthetic test data generation for edge cases
- Test fixtures and mocking infrastructure
- Sanskrit/Hindi accuracy validation datasets
"""

import json
import yaml
import logging
import tempfile
import shutil
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import random
import string

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TestDataset:
    """Represents a test dataset with metadata."""
    name: str
    description: str
    data_type: str  # 'srt', 'json', 'yaml'
    category: str  # 'golden', 'synthetic', 'edge_case', 'regression'
    language: str  # 'sanskrit', 'hindi', 'english', 'mixed'
    size: int
    checksum: str
    created_at: str
    last_modified: str
    validation_status: str
    metadata: Dict[str, Any]


@dataclass
class GoldenDatasetEntry:
    """Represents a golden dataset entry for accuracy validation."""
    original_text: str
    expected_text: str
    transformations: List[str]  # List of transformations applied
    category: str  # 'scriptural', 'sanskrit_terms', 'numbers', 'capitalization'
    confidence_score: float
    validation_notes: str
    created_by: str
    reviewed_by: Optional[str] = None


@dataclass
class SyntheticTestCase:
    """Represents a synthetically generated test case."""
    input_text: str
    expected_output: str
    test_category: str
    edge_case_type: str
    complexity_level: str  # 'simple', 'medium', 'complex'
    metadata: Dict[str, Any]


class TestDataManager:
    """
    Comprehensive test data management system for ASR post-processing testing.
    
    Handles golden datasets, synthetic data generation, test fixtures, and 
    validation infrastructure for the complete testing framework.
    """
    
    def __init__(self, data_root: Optional[Path] = None):
        """Initialize test data manager with configurable data root."""
        self.data_root = data_root or Path("data")
        self.test_data_root = Path("tests/data")
        self.golden_dataset_path = self.data_root / "golden_dataset"
        self.synthetic_data_path = self.test_data_root / "synthetic"
        self.fixtures_path = self.test_data_root / "fixtures"
        
        # Create directories if they don't exist
        self._setup_directories()
        
        # Load configuration
        self.config = self._load_configuration()
        
        # Initialize dataset registry
        self.dataset_registry: Dict[str, TestDataset] = {}
        self._load_dataset_registry()
        
        logger.info(f"TestDataManager initialized with data root: {self.data_root}")
    
    def _setup_directories(self):
        """Create necessary directories for test data management."""
        directories = [
            self.golden_dataset_path,
            self.synthetic_data_path,
            self.fixtures_path,
            self.test_data_root / "generated",
            self.test_data_root / "temp",
            self.test_data_root / "archives"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created directory: {directory}")
    
    def _load_configuration(self) -> Dict[str, Any]:
        """Load test data management configuration."""
        config_path = self.test_data_root / "config.yaml"
        
        default_config = {
            "golden_dataset": {
                "validation_threshold": 0.95,
                "max_entries_per_category": 1000,
                "require_review": True
            },
            "synthetic_data": {
                "generation_seed": 42,
                "complexity_distribution": {
                    "simple": 0.4,
                    "medium": 0.4,
                    "complex": 0.2
                },
                "edge_case_types": [
                    "unicode_corruption",
                    "malformed_timestamps",
                    "extremely_long_segments",
                    "empty_content",
                    "special_characters",
                    "mixed_languages"
                ]
            },
            "fixtures": {
                "auto_cleanup": True,
                "temp_data_retention_days": 7
            },
            "validation": {
                "accuracy_threshold": 0.90,
                "performance_threshold_segments_per_sec": 10.0,
                "variance_threshold_percent": 10.0
            }
        }
        
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                loaded_config = yaml.safe_load(f)
                # Merge with defaults
                default_config.update(loaded_config)
        else:
            # Save default configuration
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(default_config, f, default_flow_style=False)
        
        return default_config
    
    def _load_dataset_registry(self):
        """Load the dataset registry from file."""
        registry_path = self.test_data_root / "dataset_registry.json"
        
        if registry_path.exists():
            try:
                with open(registry_path, 'r', encoding='utf-8') as f:
                    registry_data = json.load(f)
                    for name, data in registry_data.items():
                        self.dataset_registry[name] = TestDataset(**data)
                logger.info(f"Loaded {len(self.dataset_registry)} datasets from registry")
            except Exception as e:
                logger.error(f"Failed to load dataset registry: {e}")
                self.dataset_registry = {}
    
    def _save_dataset_registry(self):
        """Save the dataset registry to file."""
        registry_path = self.test_data_root / "dataset_registry.json"
        
        try:
            registry_data = {
                name: asdict(dataset) for name, dataset in self.dataset_registry.items()
            }
            with open(registry_path, 'w', encoding='utf-8') as f:
                json.dump(registry_data, f, indent=2)
            logger.debug("Saved dataset registry")
        except Exception as e:
            logger.error(f"Failed to save dataset registry: {e}")
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum for a file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def create_golden_dataset_entry(
        self, 
        original_text: str,
        expected_text: str,
        transformations: List[str],
        category: str,
        confidence_score: float = 1.0,
        validation_notes: str = "",
        created_by: str = "test_system"
    ) -> GoldenDatasetEntry:
        """Create a new golden dataset entry for accuracy validation."""
        
        entry = GoldenDatasetEntry(
            original_text=original_text,
            expected_text=expected_text,
            transformations=transformations,
            category=category,
            confidence_score=confidence_score,
            validation_notes=validation_notes,
            created_by=created_by
        )
        
        return entry
    
    def save_golden_dataset(self, entries: List[GoldenDatasetEntry], dataset_name: str):
        """Save golden dataset entries to file."""
        dataset_file = self.golden_dataset_path / f"{dataset_name}.json"
        
        try:
            entries_data = [asdict(entry) for entry in entries]
            with open(dataset_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "metadata": {
                        "name": dataset_name,
                        "created_at": datetime.now().isoformat(),
                        "entry_count": len(entries),
                        "categories": list(set(entry.category for entry in entries))
                    },
                    "entries": entries_data
                }, f, indent=2, ensure_ascii=False)
            
            # Register dataset
            checksum = self._calculate_checksum(dataset_file)
            dataset = TestDataset(
                name=dataset_name,
                description=f"Golden dataset with {len(entries)} entries",
                data_type="json",
                category="golden",
                language="mixed",
                size=len(entries),
                checksum=checksum,
                created_at=datetime.now().isoformat(),
                last_modified=datetime.now().isoformat(),
                validation_status="valid",
                metadata={"categories": list(set(entry.category for entry in entries))}
            )
            
            self.dataset_registry[dataset_name] = dataset
            self._save_dataset_registry()
            
            logger.info(f"Saved golden dataset '{dataset_name}' with {len(entries)} entries")
            
        except Exception as e:
            logger.error(f"Failed to save golden dataset '{dataset_name}': {e}")
            raise
    
    def load_golden_dataset(self, dataset_name: str) -> List[GoldenDatasetEntry]:
        """Load golden dataset entries from file."""
        dataset_file = self.golden_dataset_path / f"{dataset_name}.json"
        
        if not dataset_file.exists():
            raise FileNotFoundError(f"Golden dataset '{dataset_name}' not found")
        
        try:
            with open(dataset_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                entries = [GoldenDatasetEntry(**entry_data) for entry_data in data["entries"]]
            
            logger.info(f"Loaded golden dataset '{dataset_name}' with {len(entries)} entries")
            return entries
            
        except Exception as e:
            logger.error(f"Failed to load golden dataset '{dataset_name}': {e}")
            raise
    
    def generate_synthetic_test_data(
        self, 
        count: int = 100,
        categories: Optional[List[str]] = None,
        complexity_levels: Optional[List[str]] = None
    ) -> List[SyntheticTestCase]:
        """Generate synthetic test data for edge cases and comprehensive testing."""
        
        if categories is None:
            categories = ["scriptural", "sanskrit_terms", "numbers", "capitalization", "mixed"]
        
        if complexity_levels is None:
            complexity_levels = ["simple", "medium", "complex"]
        
        synthetic_cases = []
        random.seed(self.config["synthetic_data"]["generation_seed"])
        
        # Templates for different categories
        templates = {
            "scriptural": [
                "Today we study {scripture} chapter {number} verse {number}.",
                "The {scripture} teaches us about {concept}.",
                "In {scripture}, {character} explains {teaching}."
            ],
            "sanskrit_terms": [
                "We practice {yoga_term} and {meditation_term}.",
                "The concept of {philosophy_term} is central to {tradition}.",
                "{teacher} taught about {sanskrit_concept}."
            ],
            "numbers": [
                "In the year {year}, we began studying {subject}.",
                "Chapter {number} verse {number} explains {concept}.",
                "There are {number} types of {category}."
            ],
            "capitalization": [
                "today we study {proper_noun} and {proper_noun}.",
                "{teacher} was born in {place}.",
                "the {scripture} contains {number} chapters."
            ],
            "mixed": [
                "today {teacher} explained {scripture} chapter {number} about {concept}.",
                "in the year {year}, {character} practiced {yoga_term}.",
                "the {number} principles of {philosophy} were taught by {teacher}."
            ]
        }
        
        # Data pools for substitution
        data_pools = {
            "scripture": ["bhagavad gita", "yoga sutras", "upanishads", "ramayana", "mahabharata"],
            "character": ["arjuna", "krishna", "rama", "hanuman", "sita"],
            "teacher": ["patanjali", "shankaracharya", "swami vivekananda", "ramana maharshi"],
            "yoga_term": ["pranayama", "dharana", "dhyana", "samadhi", "asana"],
            "meditation_term": ["vipassana", "trataka", "mantra", "mindfulness"],
            "philosophy_term": ["dharma", "karma", "moksha", "samsara", "ahimsa"],
            "tradition": ["yoga", "vedanta", "advaita", "samkhya"],
            "concept": ["liberation", "consciousness", "enlightenment", "self-realization"],
            "teaching": ["non-attachment", "self-inquiry", "devotion", "service"],
            "proper_noun": ["krishna", "shiva", "vishnu", "brahma", "ganga"],
            "place": ["rishikesh", "varanasi", "haridwar", "vrindavan"],
            "category": ["yoga poses", "meditation techniques", "breathing exercises"],
            "subject": ["vedanta", "yoga philosophy", "meditation"]
        }
        
        # Generate test cases
        for i in range(count):
            category = random.choice(categories)
            complexity = random.choice(complexity_levels)
            template = random.choice(templates[category])
            
            # Generate substitution values
            substitutions = {}
            for key in ["number", "year"]:
                if "{" + key + "}" in template:
                    if key == "number":
                        substitutions[key] = str(random.randint(1, 50))
                    elif key == "year":
                        substitutions[key] = f"two thousand {random.choice(['five', 'six', 'seven', 'eight', 'nine'])}"
            
            # Generate text substitutions
            for key, values in data_pools.items():
                if "{" + key + "}" in template:
                    substitutions[key] = random.choice(values)
            
            # Create input text (with common ASR errors)
            input_text = template.format(**substitutions)
            
            # Apply ASR-style corruptions based on complexity
            if complexity == "medium":
                input_text = self._apply_medium_corruptions(input_text)
            elif complexity == "complex":
                input_text = self._apply_complex_corruptions(input_text)
            
            # Generate expected output (properly formatted)
            expected_output = self._generate_expected_output(template.format(**substitutions), category)
            
            # Determine edge case type
            edge_case_type = random.choice(self.config["synthetic_data"]["edge_case_types"])
            
            synthetic_case = SyntheticTestCase(
                input_text=input_text,
                expected_output=expected_output,
                test_category=category,
                edge_case_type=edge_case_type,
                complexity_level=complexity,
                metadata={
                    "template": template,
                    "substitutions": substitutions,
                    "generated_at": datetime.now().isoformat()
                }
            )
            
            synthetic_cases.append(synthetic_case)
        
        logger.info(f"Generated {len(synthetic_cases)} synthetic test cases")
        return synthetic_cases
    
    def _apply_medium_corruptions(self, text: str) -> str:
        """Apply medium-level ASR corruptions to text."""
        corruptions = [
            ("krishna", "krsna"),
            ("dharma", "dharama"),
            ("yoga", "yog"),
            ("chapter", "chaptor"),
            ("verse", "vers")
        ]
        
        for original, corrupted in corruptions:
            if original in text.lower():
                text = text.replace(original, corrupted)
        
        return text
    
    def _apply_complex_corruptions(self, text: str) -> str:
        """Apply complex-level ASR corruptions to text."""
        text = self._apply_medium_corruptions(text)
        
        # Add more complex corruptions
        complex_corruptions = [
            ("bhagavad gita", "bhagvad geeta"),
            ("patanjali", "patanjalee"),
            ("shankaracharya", "shankara charya"),
            ("two thousand", "2000"),
            ("twenty five", "25")
        ]
        
        for original, corrupted in complex_corruptions:
            if original in text.lower():
                text = text.replace(original, corrupted)
        
        # Add filler words occasionally
        if random.random() < 0.3:
            filler_positions = random.randint(1, 3)
            fillers = ["um", "uh", "you know", "actually"]
            
            words = text.split()
            for _ in range(filler_positions):
                pos = random.randint(0, len(words))
                words.insert(pos, random.choice(fillers) + ",")
            
            text = " ".join(words)
        
        return text
    
    def _generate_expected_output(self, clean_text: str, category: str) -> str:
        """Generate expected output with proper transformations applied."""
        # Apply transformations based on category
        
        if category == "scriptural":
            # Convert number words to digits in scriptural references
            clean_text = clean_text.replace("two", "2").replace("twenty five", "25")
            # Capitalize scripture names
            clean_text = clean_text.replace("bhagavad gita", "Bhagavad Gita")
            clean_text = clean_text.replace("yoga sutras", "Yoga Sutras")
        
        elif category == "sanskrit_terms":
            # Capitalize Sanskrit terms
            sanskrit_terms = ["Krishna", "Shiva", "Vishnu", "Dharma", "Yoga", "Pranayama"]
            for term in sanskrit_terms:
                clean_text = clean_text.replace(term.lower(), term)
        
        elif category == "numbers":
            # Convert written numbers to digits
            number_conversions = {
                "two thousand five": "2005",
                "two thousand six": "2006",
                "twenty five": "25",
                "two": "2",
                "three": "3"
            }
            for written, digit in number_conversions.items():
                clean_text = clean_text.replace(written, digit)
        
        elif category == "capitalization":
            # Capitalize proper nouns
            proper_nouns = ["Krishna", "Patanjali", "Shankaracharya", "Rishikesh", "Bhagavad Gita"]
            for noun in proper_nouns:
                clean_text = clean_text.replace(noun.lower(), noun)
            
            # Capitalize sentence beginnings
            sentences = clean_text.split(". ")
            sentences = [s.capitalize() for s in sentences]
            clean_text = ". ".join(sentences)
        
        return clean_text
    
    def save_synthetic_dataset(self, test_cases: List[SyntheticTestCase], dataset_name: str):
        """Save synthetic test cases to file."""
        dataset_file = self.synthetic_data_path / f"{dataset_name}.json"
        
        try:
            cases_data = [asdict(case) for case in test_cases]
            with open(dataset_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "metadata": {
                        "name": dataset_name,
                        "created_at": datetime.now().isoformat(),
                        "case_count": len(test_cases),
                        "categories": list(set(case.test_category for case in test_cases)),
                        "complexity_levels": list(set(case.complexity_level for case in test_cases))
                    },
                    "test_cases": cases_data
                }, f, indent=2, ensure_ascii=False)
            
            # Register dataset
            checksum = self._calculate_checksum(dataset_file)
            dataset = TestDataset(
                name=dataset_name,
                description=f"Synthetic dataset with {len(test_cases)} test cases",
                data_type="json",
                category="synthetic",
                language="mixed",
                size=len(test_cases),
                checksum=checksum,
                created_at=datetime.now().isoformat(),
                last_modified=datetime.now().isoformat(),
                validation_status="valid",
                metadata={
                    "categories": list(set(case.test_category for case in test_cases)),
                    "complexity_levels": list(set(case.complexity_level for case in test_cases))
                }
            )
            
            self.dataset_registry[dataset_name] = dataset
            self._save_dataset_registry()
            
            logger.info(f"Saved synthetic dataset '{dataset_name}' with {len(test_cases)} test cases")
            
        except Exception as e:
            logger.error(f"Failed to save synthetic dataset '{dataset_name}': {e}")
            raise
    
    def create_test_fixtures(self) -> Dict[str, Any]:
        """Create comprehensive test fixtures for the testing framework."""
        
        fixtures = {
            "sample_srt_content": {
                "basic": """1
00:00:01,000 --> 00:00:05,000
Today we study yoga and dharma.

2
00:00:06,000 --> 00:00:10,000
Krishna teaches us about moksha.""",
                
                "complex": """1
00:00:01,000 --> 00:00:05,000
today we study krsna in bhagavad geeta chapter two verse twenty five.

2
00:00:06,000 --> 00:00:10,000
um, this verse, uh, teaches about dharama and, you know, yoga practices.

3
00:00:11,000 --> 00:00:15,000
In the year two thousand five, swami vivekananda explained this.""",
                
                "scriptural": """1
00:00:01,000 --> 00:00:05,000
The bhagvad gita chapter two verse twenty five says.

2
00:00:06,000 --> 00:00:10,000
yoga sutras of patanjalee explain dharma.""",
                
                "malformed": """1
00:00:01,000 --> 00:00:05,000


2
00:00:06,000 --> 00:00:10,000
Text with no timestamps

This is malformed content."""
            },
            
            "sanskrit_terms": {
                "basic": ["krishna", "dharma", "yoga", "moksha", "karma"],
                "complex": ["pranayama", "dharana", "dhyana", "samadhi", "ahimsa"],
                "proper_nouns": ["Krishna", "Shiva", "Vishnu", "Patanjali", "Shankaracharya"],
                "scriptures": ["Bhagavad Gita", "Yoga Sutras", "Upanishads", "Ramayana"]
            },
            
            "test_configurations": {
                "performance": {
                    "target_segments_per_second": 10.0,
                    "variance_threshold_percent": 10.0,
                    "memory_limit_mb": 500
                },
                "accuracy": {
                    "minimum_accuracy_threshold": 0.90,
                    "word_error_rate_threshold": 0.1,
                    "character_error_rate_threshold": 0.05
                },
                "quality": {
                    "coverage_threshold": 0.95,
                    "complexity_score_threshold": 0.8
                }
            },
            
            "mock_responses": {
                "mcp_client": {
                    "success": {"status": "success", "confidence": 0.95},
                    "failure": {"status": "error", "message": "Processing failed"},
                    "timeout": {"status": "timeout", "message": "Request timed out"}
                }
            }
        }
        
        # Save fixtures to file
        fixtures_file = self.fixtures_path / "test_fixtures.json"
        with open(fixtures_file, 'w', encoding='utf-8') as f:
            json.dump(fixtures, f, indent=2, ensure_ascii=False)
        
        logger.info("Created comprehensive test fixtures")
        return fixtures
    
    def validate_golden_dataset_accuracy(self, dataset_name: str) -> Dict[str, Any]:
        """Validate golden dataset accuracy using the processing pipeline."""
        
        try:
            # Import processing components
            import sys
            sys.path.insert(0, str(Path("src").absolute()))
            
            from post_processors.sanskrit_post_processor import SanskritPostProcessor
            
            # Load golden dataset
            entries = self.load_golden_dataset(dataset_name)
            
            # Initialize processor
            processor = SanskritPostProcessor()
            
            validation_results = {
                "dataset_name": dataset_name,
                "total_entries": len(entries),
                "validated_entries": 0,
                "passed_entries": 0,
                "failed_entries": 0,
                "accuracy_score": 0.0,
                "category_results": {},
                "failed_cases": [],
                "validation_timestamp": datetime.now().isoformat()
            }
            
            category_stats = {}
            
            for entry in entries:
                validation_results["validated_entries"] += 1
                
                try:
                    # Process the original text
                    result = processor.text_normalizer.normalize_with_advanced_tracking(entry.original_text)
                    processed_text = result.corrected_text
                    
                    # Compare with expected output
                    if processed_text.strip() == entry.expected_text.strip():
                        validation_results["passed_entries"] += 1
                        passed = True
                    else:
                        validation_results["failed_entries"] += 1
                        validation_results["failed_cases"].append({
                            "original": entry.original_text,
                            "expected": entry.expected_text,
                            "actual": processed_text,
                            "category": entry.category
                        })
                        passed = False
                    
                    # Track category statistics
                    if entry.category not in category_stats:
                        category_stats[entry.category] = {"total": 0, "passed": 0}
                    
                    category_stats[entry.category]["total"] += 1
                    if passed:
                        category_stats[entry.category]["passed"] += 1
                
                except Exception as e:
                    logger.error(f"Failed to validate entry: {e}")
                    validation_results["failed_entries"] += 1
                    validation_results["failed_cases"].append({
                        "original": entry.original_text,
                        "expected": entry.expected_text,
                        "actual": f"ERROR: {str(e)}",
                        "category": entry.category
                    })
            
            # Calculate accuracy metrics
            if validation_results["validated_entries"] > 0:
                validation_results["accuracy_score"] = (
                    validation_results["passed_entries"] / validation_results["validated_entries"]
                )
            
            # Calculate category-specific results
            for category, stats in category_stats.items():
                validation_results["category_results"][category] = {
                    "total": stats["total"],
                    "passed": stats["passed"],
                    "accuracy": stats["passed"] / stats["total"] if stats["total"] > 0 else 0.0
                }
            
            logger.info(f"Golden dataset validation completed: {validation_results['accuracy_score']:.3f} accuracy")
            return validation_results
            
        except Exception as e:
            logger.error(f"Failed to validate golden dataset '{dataset_name}': {e}")
            return {
                "dataset_name": dataset_name,
                "error": str(e),
                "validation_timestamp": datetime.now().isoformat()
            }
    
    def cleanup_temp_data(self, retention_days: int = 7):
        """Clean up temporary test data older than retention period."""
        temp_path = self.test_data_root / "temp"
        cutoff_date = datetime.now().timestamp() - (retention_days * 24 * 3600)
        
        cleaned_files = 0
        
        for file_path in temp_path.glob("*"):
            if file_path.stat().st_mtime < cutoff_date:
                try:
                    if file_path.is_file():
                        file_path.unlink()
                        cleaned_files += 1
                    elif file_path.is_dir():
                        shutil.rmtree(file_path)
                        cleaned_files += 1
                except Exception as e:
                    logger.warning(f"Failed to clean up {file_path}: {e}")
        
        logger.info(f"Cleaned up {cleaned_files} temporary files/directories")
        return cleaned_files
    
    def get_dataset_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about all managed datasets."""
        
        stats = {
            "total_datasets": len(self.dataset_registry),
            "datasets_by_category": {},
            "datasets_by_type": {},
            "total_data_size": 0,
            "last_updated": None,
            "dataset_details": []
        }
        
        for dataset in self.dataset_registry.values():
            # Count by category
            if dataset.category not in stats["datasets_by_category"]:
                stats["datasets_by_category"][dataset.category] = 0
            stats["datasets_by_category"][dataset.category] += 1
            
            # Count by type
            if dataset.data_type not in stats["datasets_by_type"]:
                stats["datasets_by_type"][dataset.data_type] = 0
            stats["datasets_by_type"][dataset.data_type] += 1
            
            # Sum data size
            stats["total_data_size"] += dataset.size
            
            # Track latest update
            if stats["last_updated"] is None or dataset.last_modified > stats["last_updated"]:
                stats["last_updated"] = dataset.last_modified
            
            # Add dataset details
            stats["dataset_details"].append({
                "name": dataset.name,
                "category": dataset.category,
                "size": dataset.size,
                "validation_status": dataset.validation_status,
                "last_modified": dataset.last_modified
            })
        
        return stats


# Initialize default test data manager
def get_test_data_manager() -> TestDataManager:
    """Get the default test data manager instance."""
    return TestDataManager()


# Test data initialization functions
def initialize_default_golden_dataset():
    """Initialize default golden dataset for testing."""
    manager = get_test_data_manager()
    
    # Create sample golden dataset entries
    golden_entries = [
        manager.create_golden_dataset_entry(
            original_text="today we study krsna in bhagavad geeta chapter two verse twenty five",
            expected_text="Today we study Krishna in Bhagavad Gita chapter 2 verse 25",
            transformations=["capitalization", "sanskrit_correction", "number_conversion"],
            category="scriptural",
            confidence_score=1.0,
            validation_notes="Complete scriptural reference with multiple transformations"
        ),
        manager.create_golden_dataset_entry(
            original_text="dharama and yog practice help us understand moksha",
            expected_text="Dharma and yoga practice help us understand moksha",
            transformations=["sanskrit_correction", "capitalization"],
            category="sanskrit_terms",
            confidence_score=0.95,
            validation_notes="Sanskrit term corrections with proper capitalization"
        ),
        manager.create_golden_dataset_entry(
            original_text="in the year two thousand five we started meditation",
            expected_text="In the year 2005 we started meditation",
            transformations=["number_conversion", "capitalization"],
            category="numbers",
            confidence_score=1.0,
            validation_notes="Year conversion with sentence capitalization"
        ),
        manager.create_golden_dataset_entry(
            original_text="patanjalee taught the eight limbs of yog",
            expected_text="Patanjali taught the eight limbs of yoga",
            transformations=["name_correction", "sanskrit_correction"],
            category="sanskrit_terms",
            confidence_score=0.98,
            validation_notes="Proper noun and Sanskrit term corrections"
        )
    ]
    
    manager.save_golden_dataset(golden_entries, "default_golden_dataset")
    
    # Generate synthetic test data
    synthetic_cases = manager.generate_synthetic_test_data(count=50)
    manager.save_synthetic_dataset(synthetic_cases, "default_synthetic_dataset")
    
    # Create test fixtures
    manager.create_test_fixtures()
    
    logger.info("Initialized default golden dataset and test data")


if __name__ == "__main__":
    # Initialize test data management system
    initialize_default_golden_dataset()
    
    # Get statistics
    manager = get_test_data_manager()
    stats = manager.get_dataset_statistics()
    
    print("Test Data Management System Initialized")
    print(f"Total datasets: {stats['total_datasets']}")
    print(f"Datasets by category: {stats['datasets_by_category']}")
    print(f"Total data size: {stats['total_data_size']}")