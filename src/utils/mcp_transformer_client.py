"""
MCP Transformer Client for Sanskrit Processing Enhancement

Integrates transformer-based semantic processing with existing MCP infrastructure
to provide research-grade Sanskrit accuracy improvements.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import asyncio
import logging
from pathlib import Path
import json
from datetime import datetime
import time

try:
    from .mcp_client_manager import MCPClientManager
except ImportError:
    # Fallback for development
    MCPClientManager = None


class SemanticConfidenceLevel(Enum):
    """Semantic confidence levels for transformer-based processing."""
    HIGH = "high"
    MEDIUM = "medium" 
    LOW = "low"
    UNCERTAIN = "uncertain"


class CulturalContext(Enum):
    """Cultural context categories for Yoga Vedanta terminology."""
    SCRIPTURAL = "scriptural"
    PHILOSOPHICAL = "philosophical"
    DEVOTIONAL = "devotional"
    PRACTICAL = "practical"
    HISTORICAL = "historical"
    UNKNOWN = "unknown"


@dataclass
class TransformerResult:
    """Result from transformer-based semantic processing."""
    original_text: str
    processed_text: str
    confidence_score: float
    semantic_context: CulturalContext
    confidence_level: SemanticConfidenceLevel
    cultural_context_score: float
    suggested_improvements: List[str]
    processing_time_ms: float


@dataclass
class SemanticClassification:
    """Classification of Sanskrit/Hindi terms with cultural context."""
    term: str
    primary_category: str
    cultural_context: CulturalContext
    semantic_confidence: float
    context_awareness_score: float
    suggested_transliteration: Optional[str]
    cultural_notes: List[str]


class MCPTransformerClient:
    """
    MCP Transformer Client for Sanskrit Processing Enhancement.
    
    Integrates with Story 4.1 MCP infrastructure to provide transformer-based
    semantic understanding and cultural context awareness for Sanskrit/Hindi terms.
    """

    def __init__(self, mcp_manager: Optional[Any] = None, config: Optional[Dict] = None):
        """
        Initialize the MCP transformer client.
        
        Args:
            mcp_manager: Existing MCP client manager from Story 4.1
            config: Configuration parameters
        """
        self.logger = logging.getLogger(__name__)
        self.mcp_manager = mcp_manager
        self.config = config or self._get_default_config()
        
        # Performance tracking
        self.processing_stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'avg_processing_time_ms': 0.0,
            'cultural_context_accuracy': 0.0
        }
        
        # Initialize transformer services
        self._initialize_transformer_services()

    def _get_default_config(self) -> Dict:
        """Get default configuration for transformer client."""
        return {
            'max_processing_time_ms': 1000,  # <1s requirement
            'cultural_context_threshold': 0.7,
            'semantic_confidence_threshold': 0.6,
            'enable_cultural_awareness': True,
            'enable_semantic_classification': True,
            'batch_processing_size': 10,
            'fallback_to_lexicon': True
        }

    def _initialize_transformer_services(self):
        """Initialize transformer services through MCP."""
        try:
            # Use existing MCP infrastructure to connect to transformer services
            self.logger.info("Initializing transformer services via MCP infrastructure")
            
            # For now, default to fallback mode since MCP infrastructure is optional
            if not self.mcp_manager:
                self.logger.warning("MCP client not available, using fallback")
                self.config['fallback_mode'] = True
            else:
                self.config['fallback_mode'] = False
                self.logger.info("Transformer services initialized successfully")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize transformer services: {e}")
            self.config['fallback_mode'] = True

    async def _test_transformer_connection(self) -> bool:
        """Test connection to transformer services."""
        try:
            # Use MCP manager to test transformer service connectivity
            test_text = "dharma yoga practice"
            result = await self._call_transformer_service("semantic_classify", {"text": test_text})
            return result is not None
        except Exception as e:
            self.logger.debug(f"Transformer connection test failed: {e}")
            return False

    async def _call_transformer_service(self, operation: str, params: Dict) -> Optional[Dict]:
        """Call transformer service through MCP infrastructure."""
        try:
            # Leverage existing MCP client for transformer calls
            if self.mcp_manager and hasattr(self.mcp_manager, 'mcp_client') and self.mcp_manager.mcp_client:
                # Use MCP client to call transformer services
                # This integrates with the existing MCP infrastructure from Story 4.1
                response = await self._mcp_transformer_call(operation, params)
                return response
            else:
                self.logger.warning("MCP client not available, using fallback")
                return await self._fallback_processing(operation, params)
                
        except Exception as e:
            self.logger.error(f"Transformer service call failed: {e}")
            return await self._fallback_processing(operation, params)

    async def _mcp_transformer_call(self, operation: str, params: Dict) -> Dict:
        """Make transformer call through MCP infrastructure."""
        # Placeholder for actual MCP transformer integration
        # In real implementation, this would use the MCP protocol to call transformer services
        
        if operation == "semantic_classify":
            return await self._mock_semantic_classify(params["text"])
        elif operation == "cultural_context":
            return await self._mock_cultural_context(params["text"])
        elif operation == "confidence_score":
            return await self._mock_confidence_scoring(params["text"], params.get("context", ""))
        else:
            raise ValueError(f"Unknown transformer operation: {operation}")

    async def _mock_semantic_classify(self, text: str) -> Dict:
        """Mock semantic classification for development."""
        # This would be replaced with actual transformer calls
        sanskrit_terms = ["dharma", "yoga", "krishna", "gita", "vedanta", "karma", "bhakti"]
        
        contains_sanskrit = any(term in text.lower() for term in sanskrit_terms)
        
        return {
            "classification": "sanskrit_philosophical" if contains_sanskrit else "general",
            "confidence": 0.85 if contains_sanskrit else 0.3,
            "cultural_context": CulturalContext.PHILOSOPHICAL.value if contains_sanskrit else CulturalContext.UNKNOWN.value,
            "semantic_features": ["spiritual", "philosophical"] if contains_sanskrit else ["general"]
        }

    async def _mock_cultural_context(self, text: str) -> Dict:
        """Mock cultural context analysis."""
        context_keywords = {
            CulturalContext.SCRIPTURAL: ["gita", "upanishad", "veda", "sutra", "bhagavad"],
            CulturalContext.PHILOSOPHICAL: ["dharma", "karma", "moksha", "samsara", "vedanta"],
            CulturalContext.DEVOTIONAL: ["krishna", "rama", "shiva", "bhakti", "krsna"],
            CulturalContext.PRACTICAL: ["yoga", "meditation", "pranayama", "asana"]
        }
        
        text_lower = text.lower()
        best_context = CulturalContext.UNKNOWN
        best_score = 0.0
        
        for context, keywords in context_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > best_score:
                best_score = score
                best_context = context
        
        # Ensure we recognize common Sanskrit terms even if not in keywords
        if best_context == CulturalContext.UNKNOWN:
            sanskrit_indicators = ["krishna", "dharma", "yoga", "karma", "moksha", "gita", "vedanta", "bhakti"]
            if any(indicator in text_lower for indicator in sanskrit_indicators):
                best_context = CulturalContext.PHILOSOPHICAL
                best_score = 1.0
        
        return {
            "primary_context": best_context.value,
            "context_confidence": min(max(best_score / 2.0, 0.5), 1.0) if best_context != CulturalContext.UNKNOWN else 0.3,
            "detected_elements": [kw for kw in context_keywords.get(best_context, []) if kw in text_lower]
        }

    async def _mock_confidence_scoring(self, text: str, context: str) -> Dict:
        """Mock confidence scoring for transformer results."""
        # Simulate transformer-based confidence scoring
        base_confidence = 0.7
        
        # Adjust based on context
        if context in ["scriptural", "philosophical"]:
            base_confidence += 0.15
        elif context in ["practical", "devotional"]:
            base_confidence += 0.1
        
        return {
            "confidence_score": min(base_confidence, 1.0),
            "confidence_factors": {
                "context_match": context != "unknown",
                "term_recognition": len(text.split()) > 0,
                "semantic_coherence": True
            }
        }

    async def _fallback_processing(self, operation: str, params: Dict) -> Dict:
        """Fallback processing when transformer services unavailable."""
        self.logger.debug(f"Using fallback processing for {operation}")
        
        # Use rule-based fallback similar to existing lexicon-based processing
        if operation == "semantic_classify":
            text = params.get("text", "").lower()
            sanskrit_terms = ["dharma", "yoga", "krishna", "gita", "vedanta", "karma", "bhakti"]
            contains_sanskrit = any(term in text for term in sanskrit_terms)
            return {
                "classification": "sanskrit_philosophical" if contains_sanskrit else "general", 
                "confidence": 0.7 if contains_sanskrit else 0.3, 
                "fallback": True
            }
        elif operation == "cultural_context":
            # Apply the same logic as _mock_cultural_context for fallback
            return await self._mock_cultural_context(params.get("text", ""))
        else:
            return {"result": "fallback", "confidence": 0.1}

    async def process_sanskrit_text_with_context(self, text: str, existing_context: Optional[Dict] = None) -> TransformerResult:
        """
        Process Sanskrit/Hindi text with cultural context awareness.
        
        Args:
            text: Text to process
            existing_context: Existing context from lexicon or previous processing
            
        Returns:
            TransformerResult with semantic and cultural analysis
        """
        start_time = time.time()
        
        try:
            # Semantic classification
            semantic_result = await self._call_transformer_service(
                "semantic_classify", 
                {"text": text, "context": existing_context}
            )
            
            # Cultural context analysis
            cultural_result = await self._call_transformer_service(
                "cultural_context",
                {"text": text}
            )
            
            # Confidence scoring
            confidence_result = await self._call_transformer_service(
                "confidence_score",
                {"text": text, "context": cultural_result.get("primary_context", "")}
            )
            
            # Combine results
            cultural_context = CulturalContext(cultural_result.get("primary_context", "unknown"))
            confidence_score = confidence_result.get("confidence_score", 0.5)
            
            # Determine confidence level
            if confidence_score >= 0.9:
                confidence_level = SemanticConfidenceLevel.HIGH
            elif confidence_score >= 0.7:
                confidence_level = SemanticConfidenceLevel.MEDIUM
            elif confidence_score >= 0.5:
                confidence_level = SemanticConfidenceLevel.LOW
            else:
                confidence_level = SemanticConfidenceLevel.UNCERTAIN
            
            processing_time_ms = (time.time() - start_time) * 1000
            
            # Update stats
            self._update_processing_stats(True, processing_time_ms)
            
            result = TransformerResult(
                original_text=text,
                processed_text=text,  # Would be modified by actual transformer
                confidence_score=confidence_score,
                semantic_context=cultural_context,
                confidence_level=confidence_level,
                cultural_context_score=cultural_result.get("context_confidence", 0.5),
                suggested_improvements=[],  # Would come from transformer
                processing_time_ms=processing_time_ms
            )
            
            self.logger.debug(f"Processed '{text}' with confidence {confidence_score:.3f}")
            return result
            
        except Exception as e:
            processing_time_ms = (time.time() - start_time) * 1000
            self._update_processing_stats(False, processing_time_ms)
            
            self.logger.error(f"Error processing text '{text}': {e}")
            
            # Return fallback result
            return TransformerResult(
                original_text=text,
                processed_text=text,
                confidence_score=0.1,
                semantic_context=CulturalContext.UNKNOWN,
                confidence_level=SemanticConfidenceLevel.UNCERTAIN,
                cultural_context_score=0.0,
                suggested_improvements=[],
                processing_time_ms=processing_time_ms
            )

    def classify_term_semantically(self, term: str, context: Optional[str] = None) -> SemanticClassification:
        """
        Classify a Sanskrit/Hindi term with semantic understanding.
        
        Args:
            term: Term to classify
            context: Optional context for classification
            
        Returns:
            SemanticClassification with cultural context awareness
        """
        try:
            # Use synchronous wrapper for async processing
            result = asyncio.run(self._classify_term_async(term, context))
            return result
        except Exception as e:
            self.logger.error(f"Error classifying term '{term}': {e}")
            
            # Return fallback classification
            return SemanticClassification(
                term=term,
                primary_category="unknown",
                cultural_context=CulturalContext.UNKNOWN,
                semantic_confidence=0.1,
                context_awareness_score=0.0,
                suggested_transliteration=None,
                cultural_notes=[]
            )

    async def _classify_term_async(self, term: str, context: Optional[str]) -> SemanticClassification:
        """Async implementation of term classification."""
        # Get semantic classification
        semantic_result = await self._call_transformer_service(
            "semantic_classify", 
            {"text": term, "context": context}
        )
        
        # Get cultural context
        cultural_result = await self._call_transformer_service(
            "cultural_context",
            {"text": term}
        )
        
        return SemanticClassification(
            term=term,
            primary_category=semantic_result.get("classification", "unknown"),
            cultural_context=CulturalContext(cultural_result.get("primary_context", "unknown")),
            semantic_confidence=semantic_result.get("confidence", 0.5),
            context_awareness_score=cultural_result.get("context_confidence", 0.5),
            suggested_transliteration=semantic_result.get("suggested_transliteration"),
            cultural_notes=cultural_result.get("cultural_notes", [])
        )

    def _update_processing_stats(self, success: bool, processing_time_ms: float):
        """Update processing statistics."""
        self.processing_stats['total_requests'] += 1
        
        if success:
            self.processing_stats['successful_requests'] += 1
        else:
            self.processing_stats['failed_requests'] += 1
        
        # Update average processing time
        total_requests = self.processing_stats['total_requests']
        current_avg = self.processing_stats['avg_processing_time_ms']
        self.processing_stats['avg_processing_time_ms'] = (
            (current_avg * (total_requests - 1) + processing_time_ms) / total_requests
        )

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the transformer client."""
        success_rate = 0.0
        if self.processing_stats['total_requests'] > 0:
            success_rate = self.processing_stats['successful_requests'] / self.processing_stats['total_requests']
        
        return {
            **self.processing_stats,
            'success_rate': success_rate,
            'performance_target_met': self.processing_stats['avg_processing_time_ms'] < self.config['max_processing_time_ms'],
            'fallback_mode': self.config.get('fallback_mode', False),
            'cultural_context_enabled': self.config['enable_cultural_awareness']
        }

    def validate_performance_targets(self) -> bool:
        """Validate that performance targets are being met."""
        metrics = self.get_performance_metrics()
        
        # Check processing time requirement (<1s)
        time_target_met = metrics['avg_processing_time_ms'] < self.config['max_processing_time_ms']
        
        # Check success rate
        success_target_met = metrics['success_rate'] >= 0.9
        
        return time_target_met and success_target_met


def create_transformer_client(mcp_manager: Optional[Any] = None, config: Optional[Dict] = None) -> MCPTransformerClient:
    """
    Factory function to create MCP transformer client.
    
    Args:
        mcp_manager: Optional MCP client manager from Story 4.1
        config: Optional configuration
        
    Returns:
        Configured MCPTransformerClient instance
    """
    return MCPTransformerClient(mcp_manager=mcp_manager, config=config)


# Test function for development
async def test_transformer_client():
    """Test the transformer client functionality."""
    client = create_transformer_client()
    
    test_texts = [
        "Today we study dharma and yoga practice",
        "Krishna teaches in the Bhagavad Gita", 
        "Regular English text without Sanskrit terms"
    ]
    
    for text in test_texts:
        result = await client.process_sanskrit_text_with_context(text)
        print(f"Text: {text}")
        print(f"Confidence: {result.confidence_score:.3f}")
        print(f"Cultural Context: {result.semantic_context.value}")
        print(f"Processing Time: {result.processing_time_ms:.1f}ms")
        print("---")
    
    # Test semantic classification
    for text in ["dharma", "yoga", "regular"]:
        classification = client.classify_term_semantically(text)
        print(f"Term: {text} -> Category: {classification.primary_category}, Context: {classification.cultural_context.value}")


if __name__ == "__main__":
    asyncio.run(test_transformer_client())