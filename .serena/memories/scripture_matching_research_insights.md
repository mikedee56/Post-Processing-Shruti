# Digital Dharma Research Insights for ASR Scripture Matching

## Key Technologies Identified

### 1. Fuzzy and Phonetic Search
- **Levenshtein Distance**: For handling transliteration variations (Krishna/Kṛṣṇa)
- **Phonetic Encoding**: Critical for Sanskrit oral tradition matching
- **Soundex/Metaphone**: For approximate phonetic matching

### 2. Semantic Search Architecture
- **Vector Embeddings**: Using BERT/SentenceTransformers for meaning-based matching
- **FAISS Indexing**: Facebook AI Similarity Search for efficient vector retrieval
- **RAG Pipeline**: Retrieval-Augmented Generation for grounding responses

### 3. Sanskrit-Specific NLP Challenges
- **Sandhi Splitting**: Breaking compound words (95% accuracy with DD-RNN)
- **Morphological Analysis**: Handling highly inflected Sanskrit forms
- **IAST Transliteration**: Standardized representation

## Existing Implementations to Learn From

1. **VedicSage**: GitHub project using FAISS + SentenceTransformers for Upanishads
2. **Gita Life Guide**: Web app with semantic search on Bhagavad Gita verses
3. **GitaGPT**: RAG-based conversational agent with verse retrieval
4. **SanskritShala**: IIT's neural toolkit for Sanskrit NLP
5. **The Sanskrit Library**: Morphological analyzers and paradigm generators

## Critical Gaps for ASR Matching

1. **Phonetic Tolerance**: ASR outputs like "vow dharm ashya glan ir bavat ebharata" need robust phonetic matching
2. **Multi-Stage Pipeline**: Need combination of phonetic → sequence → semantic matching
3. **Confidence Scoring**: Must handle partial matches and provide confidence levels
4. **Source Authority**: Need provenance tracking (which scripture, chapter, verse)

## Implementation Strategy

### Stage 1: Phonetic Hashing
- Generate phonetic signatures for both ASR and canonical texts
- Use Sanskrit-aware phonetic encoding (not just English Soundex)

### Stage 2: Sequence Alignment
- Smith-Waterman algorithm for local sequence alignment
- Handle word order variations and missing segments

### Stage 3: Semantic Validation
- Vector similarity using pre-trained Sanskrit/multilingual models
- Contextual validation against surrounding verses

### Data Sources Needed
- Bhagavad Gita: Well-structured YAML with IAST transliterations
- Upanishads: Principal texts with variations
- Yoga Sutras: Complete with commentaries
- Puranas: Selected verses (future expansion)