# 8. Data Models

### 8.1 Core Data Structures

#### 8.1.1 Transcript Segment
**Purpose**: Represent an individual, timestamped segment of the transcript.

```yaml
TranscriptSegment:
  id: string                    # Unique segment identifier
  text: string                  # The transcribed text of the segment
  start_time: float             # Start time in seconds
  end_time: float               # End time in seconds
  confidence_score: float       # ASR confidence score (0.0 - 1.0)
  is_flagged: boolean           # Requires human review
  flag_reason: string           # Reason for flagging (low_confidence, high_oov, etc.)
  correction_history: array     # Log of all human corrections
  processing_metadata: object   # Additional processing information
```

#### 8.1.2 Lexicon Entry
**Purpose**: Represent a single entry in the domain-specific lexicon.

```yaml
LexiconEntry:
  original_term: string         # The correct term (e.g., "Dharma")
  variations: array[string]     # Common misrecognized variations
  transliteration: string       # IAST transliteration standard
  is_proper_noun: boolean       # Should be capitalized
  is_verse: boolean             # Part of scriptural verse
  canonical_text: string        # Full canonical text for verses
  category: string              # Term classification (philosophy, deity, text, etc.)
  confidence: float             # Lexicon entry reliability score
  last_updated: datetime        # Version control timestamp
  source_authority: string      # Academic source reference
```

#### 8.1.3 Processing Result
**Purpose**: Track processing outcomes and quality metrics.

```yaml
ProcessingResult:
  file_id: string               # Unique processing run identifier
  original_file: string         # Input SRT file path
  processed_file: string        # Output SRT file path
  processing_time: float        # Total processing duration
  corrections_made: integer     # Number of corrections applied
  segments_flagged: integer     # Segments requiring human review
  confidence_average: float     # Average confidence across segments
  lexicon_version: string       # Lexicon version used
  quality_metrics: object       # WER, CER, and custom metrics
  created_at: datetime          # Processing completion timestamp
```

### 8.2 Configuration Models

#### 8.2.1 Processing Configuration
```yaml
ProcessingConfig:
  lexicon_paths: array[string]  # Paths to lexicon files
  confidence_thresholds: object # Flagging thresholds by correction type
  iast_rules: object           # Transliteration rule configurations
  exclusion_patterns: array    # Terms to exclude from processing
  batch_size: integer         # Processing batch configuration
  logging_level: string       # System logging configuration
```

---
