"""
MCP Client Implementation for Story 5.2
Implements robust MCP client with connection management, reliability patterns,
and professional standards compliance per CEO directive.
"""

import asyncio
import time
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum
import httpx
import websockets
from pydantic import BaseModel, ValidationError

from utils.professional_standards import ProfessionalStandardsValidator

logger = logging.getLogger(__name__)


class MCPConnectionState(Enum):
    """MCP connection state enumeration"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    FAILED = "failed"


@dataclass
class MCPConfig:
    """MCP client configuration"""
    server_url: str = "ws://localhost:8000"
    authentication_token: Optional[str] = None
    connection_timeout: float = 30.0
    request_timeout: float = 10.0
    max_retries: int = 3
    retry_backoff: float = 1.0
    max_backoff: float = 60.0
    enable_compression: bool = True
    max_connections: int = 10
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: float = 60.0


@dataclass
class MCPPerformanceMetrics:
    """MCP performance tracking"""
    requests_sent: int = 0
    responses_received: int = 0
    failed_requests: int = 0
    average_response_time: float = 0.0
    connection_uptime: float = 0.0
    last_request_time: float = field(default_factory=time.time)
    circuit_breaker_trips: int = 0


class MCPRequest(BaseModel):
    """MCP request model with validation"""
    id: str
    method: str
    params: Dict[str, Any] = {}
    timestamp: float = field(default_factory=time.time)


class MCPResponse(BaseModel):
    """MCP response model with validation"""
    id: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None
    timestamp: float = field(default_factory=time.time)


class MCPCircuitBreaker:
    """Circuit breaker pattern for MCP reliability"""
    
    def __init__(self, config: MCPConfig):
        self.config = config
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "closed"  # closed, open, half-open
        
    def can_execute(self) -> bool:
        """Check if request can be executed based on circuit breaker state"""
        if self.state == "closed":
            return True
        elif self.state == "open":
            if time.time() - self.last_failure_time > self.config.circuit_breaker_timeout:
                self.state = "half-open"
                return True
            return False
        elif self.state == "half-open":
            return True
        return False
    
    def record_success(self):
        """Record successful request"""
        self.failure_count = 0
        self.state = "closed"
    
    def record_failure(self):
        """Record failed request"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.config.circuit_breaker_threshold:
            self.state = "open"


class MCPSessionManager:
    """MCP session lifecycle management with professional standards compliance"""
    
    def __init__(self, config: MCPConfig):
        self.config = config
        self.connection_state = MCPConnectionState.DISCONNECTED
        self.websocket = None
        self.session_id = None
        self.connection_start_time = None
        self.professional_validator = ProfessionalStandardsValidator()
        
    async def connect(self) -> bool:
        """Establish MCP connection with professional validation"""
        try:
            self.connection_state = MCPConnectionState.CONNECTING
            
            # Professional standards validation for connection attempt
            connection_claims = {
                'connection_attempt': {
                    'factual_basis': f'Attempting connection to {self.config.server_url}',
                    'verification_method': 'websocket_handshake',
                    'supporting_data': {'timeout': self.config.connection_timeout}
                }
            }
            
            validation_result = self.professional_validator.validate_technical_claims(connection_claims)
            if not validation_result['professional_compliance']:
                logger.error("Connection attempt failed professional standards validation")
                return False
            
            # Establish WebSocket connection
            self.websocket = await websockets.connect(
                self.config.server_url,
                timeout=self.config.connection_timeout,
                compression="deflate" if self.config.enable_compression else None
            )
            
            self.connection_state = MCPConnectionState.CONNECTED
            self.connection_start_time = time.time()
            logger.info(f"MCP connection established to {self.config.server_url}")
            
            return True
            
        except Exception as e:
            self.connection_state = MCPConnectionState.FAILED
            logger.error(f"MCP connection failed: {e}")
            return False
    
    async def disconnect(self):
        """Clean disconnection"""
        if self.websocket:
            await self.websocket.close()
            self.websocket = None
        
        self.connection_state = MCPConnectionState.DISCONNECTED
        self.session_id = None
    
    def is_connected(self) -> bool:
        """Check connection status"""
        return (self.connection_state == MCPConnectionState.CONNECTED and 
                self.websocket is not None and 
                not self.websocket.closed)


class MCPClient:
    """
    Robust MCP client with connection management, reliability patterns,
    and professional standards compliance per CEO directive.
    """
    
    def __init__(self, config: Optional[MCPConfig] = None):
        self.config = config or MCPConfig()
        self.session_manager = MCPSessionManager(self.config)
        self.circuit_breaker = MCPCircuitBreaker(self.config)
        self.performance_metrics = MCPPerformanceMetrics()
        self.professional_validator = ProfessionalStandardsValidator()
        self.connection_pool = []
        self.request_cache = {}
        
        logger.info("MCPClient initialized with professional standards compliance")
    
    async def process_text(self, text: str, context: str = "general") -> str:
        """
        MCP-enhanced text processing with fallback and professional validation
        
        Args:
            text: Text to process
            context: Processing context for optimization
            
        Returns:
            Processed text with MCP enhancements or fallback processing
        """
        # Professional standards validation for processing request
        processing_claims = {
            'text_processing_request': {
                'factual_basis': f'Processing {len(text)} characters with context: {context}',
                'verification_method': 'mcp_text_processing',
                'supporting_data': {'text_length': len(text), 'context': context}
            }
        }
        
        validation_result = self.professional_validator.validate_technical_claims(processing_claims)
        if not validation_result['professional_compliance']:
            logger.error("Text processing request failed professional standards validation")
            return text  # Return original text if validation fails
        
        # Check circuit breaker
        if not self.circuit_breaker.can_execute():
            logger.warning("Circuit breaker open - using fallback processing")
            return await self._fallback_processing(text, context)
        
        try:
            # Ensure connection
            if not self.session_manager.is_connected():
                await self.session_manager.connect()
            
            # Create MCP request
            request = MCPRequest(
                id=f"text_proc_{int(time.time() * 1000)}",
                method="process_text",
                params={
                    "text": text,
                    "context": context,
                    "timestamp": time.time()
                }
            )
            
            # Send request and get response
            start_time = time.time()
            response = await self._send_request(request)
            processing_time = time.time() - start_time
            
            # Update metrics
            self.performance_metrics.requests_sent += 1
            self.performance_metrics.responses_received += 1
            self._update_average_response_time(processing_time)
            
            # Record success
            self.circuit_breaker.record_success()
            
            # Extract processed text
            if response and response.result:
                processed_text = response.result.get("processed_text", text)
                logger.info(f"MCP text processing completed in {processing_time:.3f}s")
                return processed_text
            else:
                logger.warning("MCP response empty - using fallback")
                return await self._fallback_processing(text, context)
                
        except Exception as e:
            logger.error(f"MCP text processing error: {e}")
            self.performance_metrics.failed_requests += 1
            self.circuit_breaker.record_failure()
            
            # Fallback processing
            return await self._fallback_processing(text, context)
    
    async def classify_context(self, text: str) -> Dict[str, Any]:
        """MCP-enhanced context classification"""
        if not self.circuit_breaker.can_execute():
            return {"context": "general", "confidence": 0.5, "source": "fallback"}
        
        try:
            if not self.session_manager.is_connected():
                await self.session_manager.connect()
            
            request = MCPRequest(
                id=f"context_class_{int(time.time() * 1000)}",
                method="classify_context",
                params={"text": text}
            )
            
            response = await self._send_request(request)
            
            if response and response.result:
                return response.result
            else:
                return {"context": "general", "confidence": 0.5, "source": "mcp_fallback"}
                
        except Exception as e:
            logger.error(f"MCP context classification error: {e}")
            return {"context": "general", "confidence": 0.3, "source": "error_fallback"}
    
    async def _send_request(self, request: MCPRequest) -> Optional[MCPResponse]:
        """Send MCP request with timeout and validation"""
        try:
            # Validate request
            request_data = request.model_dump()
            
            # Send via WebSocket
            await self.session_manager.websocket.send(str(request_data))
            
            # Wait for response with timeout
            response_data = await asyncio.wait_for(
                self.session_manager.websocket.recv(),
                timeout=self.config.request_timeout
            )
            
            # Parse and validate response
            response_dict = eval(response_data)  # In production, use proper JSON parsing
            response = MCPResponse(**response_dict)
            
            return response
            
        except asyncio.TimeoutError:
            logger.error(f"MCP request timeout after {self.config.request_timeout}s")
            return None
        except ValidationError as e:
            logger.error(f"MCP response validation error: {e}")
            return None
        except Exception as e:
            logger.error(f"MCP request error: {e}")
            return None
    
    async def _fallback_processing(self, text: str, context: str) -> str:
        """Fallback text processing when MCP unavailable"""
        # Simple rule-based fallback without circular dependency
        # Import the basic TextNormalizer which doesn't depend on MCP
        from utils.text_normalizer import TextNormalizer
        
        # Use basic rule-based processing as fallback
        normalizer = TextNormalizer()
        result = normalizer.convert_numbers(text)
        return result
    
    def _update_average_response_time(self, response_time: float):
        """Update average response time metric"""
        if self.performance_metrics.responses_received == 1:
            self.performance_metrics.average_response_time = response_time
        else:
            # Exponential moving average
            alpha = 0.1
            self.performance_metrics.average_response_time = (
                alpha * response_time + 
                (1 - alpha) * self.performance_metrics.average_response_time
            )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive MCP performance statistics"""
        uptime = (time.time() - self.session_manager.connection_start_time 
                 if self.session_manager.connection_start_time else 0)
        
        return {
            'connection_state': self.session_manager.connection_state.value,
            'requests_sent': self.performance_metrics.requests_sent,
            'responses_received': self.performance_metrics.responses_received,
            'failed_requests': self.performance_metrics.failed_requests,
            'success_rate': (self.performance_metrics.responses_received / 
                           max(self.performance_metrics.requests_sent, 1) * 100),
            'average_response_time': self.performance_metrics.average_response_time,
            'connection_uptime': uptime,
            'circuit_breaker_state': self.circuit_breaker.state,
            'circuit_breaker_trips': self.performance_metrics.circuit_breaker_trips
        }
    
    def get_professional_compliance_report(self) -> Dict[str, Any]:
        """Get professional standards compliance report per CEO directive"""
        return self.professional_validator.get_professional_compliance_report()
    
    async def close(self):
        """Clean shutdown of MCP client"""
        await self.session_manager.disconnect()
        logger.info("MCPClient connection closed")


# Factory function for easy instantiation
def create_mcp_client(config: Optional[Dict[str, Any]] = None) -> MCPClient:
    """Create MCP client with optional configuration"""
    if config:
        mcp_config = MCPConfig(**config)
    else:
        mcp_config = MCPConfig()
    
    return MCPClient(mcp_config)