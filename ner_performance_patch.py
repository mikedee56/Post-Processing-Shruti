
# Performance Optimization Patch for YogaVedantaNER
# Apply this patch to implement entity caching

# Add to __init__ method:
self._entity_cache = {}
self._cache_max_size = 1000

# Replace identify_entities method with cached version:

    def __init__(self, training_data_dir: Optional[Path] = None, lexicon_manager: Optional[LexiconManager] = None):
        """Initialize YogaVedantaNER with performance optimizations."""
        self.training_data_dir = training_data_dir or Path("data/ner_training")
        self.lexicon_manager = lexicon_manager or LexiconManager()
        
        # PERFORMANCE OPTIMIZATION: Entity result caching
        self._entity_cache = {}
        self._cache_max_size = 1000
        
        # Initialize components...
        config_path = self.training_data_dir / "config" / "ner_config.yaml"
        self.config = self._load_config(config_path)
        
        self.entity_classifier = EntityClassifier()
        
        self.logger.info("Loaded configuration for 6 categories")
        self.logger.info("EntityClassifier initialized with 6 categories")
        
        # Initialize NLP libraries
        try:
            import indicnlp
            from indicnlp import loader
            loader.load()
            self.indicnlp_available = True
            self.logger.info("IndicNLP Library initialized for Sanskrit/Hindi tokenization")
        except ImportError:
            self.indicnlp_available = False
            self.logger.warning("IndicNLP Library not available")
        
        try:
            import inltk
            self.inltk_available = True
            self.logger.info("iNLTK library available for enhanced Indic processing")
        except ImportError:
            self.inltk_available = False
            self.logger.warning("iNLTK library not available")
        
        # Load training data and patterns
        self._load_training_data()
        self._load_recognition_patterns()
        
        self.logger.info("NER model initialization complete")
        self.logger.info(f"PRD-compliant YogaVedantaNER model v2.0-PRD-Compliant initialized")
        self.logger.info(f"IndicNLP available: {self.indicnlp_available}, iNLTK available: {self.inltk_available}, ByT5-Sanskrit enabled: False")
    
    def identify_entities(self, text: str) -> NERResult:
        """PERFORMANCE OPTIMIZED: Identify entities with caching."""
        # Check cache first
        text_hash = hash(text)
        if text_hash in self._entity_cache:
            return self._entity_cache[text_hash]
        
        # Process if not in cache
        result = self._process_entities_uncached(text)
        
        # Cache result (with size limit)
        if len(self._entity_cache) >= self._cache_max_size:
            # Remove oldest entries (simple FIFO)
            oldest_keys = list(self._entity_cache.keys())[:100]
            for key in oldest_keys:
                del self._entity_cache[key]
        
        self._entity_cache[text_hash] = result
        return result
    
    def _process_entities_uncached(self, text: str) -> NERResult:
        """Original entity processing without caching."""
        entities = []
        confidence_scores = []
        
        # Use lexicon-based identification
        lexicon_entities = self._find_lexicon_entities(text)
        entities.extend(lexicon_entities)
        
        # Use pattern-based identification
        pattern_entities = self._find_pattern_entities(text)
        entities.extend(pattern_entities)
        
        # Calculate confidence scores
        for entity in entities:
            confidence_scores.append(entity.confidence)
        
        overall_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
        
        return NERResult(
            entities=entities,
            overall_confidence=overall_confidence,
            processing_time_ms=0.0,  # Simplified for performance
            model_version="2.0-PRD-Compliant"
        )

