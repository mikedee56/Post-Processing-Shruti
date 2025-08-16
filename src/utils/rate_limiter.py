"""
Epic 4.3 Rate Limiter for Production Load Management.

Implements production-grade rate limiting to protect system resources
and ensure consistent performance under high load conditions.
"""

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, Optional, Any


class RateLimitStrategy(Enum):
    """Rate limiting strategies."""
    TOKEN_BUCKET = "token_bucket"
    SLIDING_WINDOW = "sliding_window"
    FIXED_WINDOW = "fixed_window"
    LEAKY_BUCKET = "leaky_bucket"


@dataclass
class RateLimitMetrics:
    """Rate limiter performance metrics."""
    total_requests: int = 0
    allowed_requests: int = 0
    rejected_requests: int = 0
    
    # Time tracking
    window_start: datetime = field(default_factory=datetime.now)
    last_request: Optional[datetime] = None
    last_reset: datetime = field(default_factory=datetime.now)
    
    # Performance metrics
    rejection_rate: float = 0.0
    average_tokens_consumed: float = 0.0
    peak_request_rate: float = 0.0


class RateLimitError(Exception):
    """Raised when rate limit is exceeded."""
    pass


class RateLimiter:
    """
    Epic 4.3 Production-Grade Rate Limiter.
    
    Protects system resources by controlling request rates using:
    - Token bucket algorithm for burst handling
    - Sliding window for smooth rate limiting
    - Automatic token replenishment
    - Real-time metrics and monitoring
    """
    
    def __init__(self,
                 max_requests: int = 100,
                 time_window: int = 60,
                 strategy: RateLimitStrategy = RateLimitStrategy.TOKEN_BUCKET,
                 burst_size: Optional[int] = None,
                 replenish_rate: Optional[float] = None):
        """
        Initialize rate limiter with Epic 4.3 production settings.
        
        Args:
            max_requests: Maximum requests allowed in time window
            time_window: Time window in seconds
            strategy: Rate limiting strategy to use
            burst_size: Maximum burst size (defaults to max_requests)
            replenish_rate: Token replenishment rate per second
        """
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.max_requests = max_requests
        self.time_window = time_window
        self.strategy = strategy
        self.burst_size = burst_size or max_requests
        self.replenish_rate = replenish_rate or (max_requests / time_window)
        
        # State management
        self.lock = threading.RLock()
        self.metrics = RateLimitMetrics()
        
        # Token bucket state
        self.tokens = float(self.burst_size)
        self.last_replenish = time.time()
        
        # Sliding window state
        self.request_times = deque()
        
        # Fixed window state
        self.window_count = 0
        self.window_start = time.time()
        
        self.logger.info(f"RateLimiter initialized: {max_requests} req/{time_window}s, strategy={strategy.value}")
    
    def acquire(self, tokens: int = 1) -> bool:
        """
        Attempt to acquire tokens for request processing.
        
        Args:
            tokens: Number of tokens to acquire
            
        Returns:
            bool: True if tokens acquired, False if rate limited
        """
        with self.lock:
            current_time = time.time()
            self.metrics.total_requests += 1
            self.metrics.last_request = datetime.now()
            
            # Execute rate limiting strategy
            allowed = False
            
            if self.strategy == RateLimitStrategy.TOKEN_BUCKET:
                allowed = self._token_bucket_acquire(tokens, current_time)
            elif self.strategy == RateLimitStrategy.SLIDING_WINDOW:
                allowed = self._sliding_window_acquire(tokens, current_time)
            elif self.strategy == RateLimitStrategy.FIXED_WINDOW:
                allowed = self._fixed_window_acquire(tokens, current_time)
            elif self.strategy == RateLimitStrategy.LEAKY_BUCKET:
                allowed = self._leaky_bucket_acquire(tokens, current_time)
            
            # Update metrics
            if allowed:
                self.metrics.allowed_requests += 1
                self.metrics.average_tokens_consumed = (
                    (self.metrics.average_tokens_consumed * (self.metrics.allowed_requests - 1) + tokens) /
                    self.metrics.allowed_requests
                )
            else:
                self.metrics.rejected_requests += 1
            
            # Calculate rejection rate
            if self.metrics.total_requests > 0:
                self.metrics.rejection_rate = self.metrics.rejected_requests / self.metrics.total_requests
            
            return allowed
    
    def acquire_or_raise(self, tokens: int = 1) -> None:
        """
        Acquire tokens or raise RateLimitError.
        
        Args:
            tokens: Number of tokens to acquire
            
        Raises:
            RateLimitError: If rate limit exceeded
        """
        if not self.acquire(tokens):
            raise RateLimitError(f"Rate limit exceeded: {self.max_requests} req/{self.time_window}s")
    
    def get_available_tokens(self) -> int:
        """Get number of currently available tokens."""
        with self.lock:
            if self.strategy == RateLimitStrategy.TOKEN_BUCKET:
                self._replenish_tokens()
                return int(self.tokens)
            elif self.strategy == RateLimitStrategy.SLIDING_WINDOW:
                current_time = time.time()
                self._cleanup_sliding_window(current_time)
                return max(0, self.max_requests - len(self.request_times))
            elif self.strategy == RateLimitStrategy.FIXED_WINDOW:
                current_time = time.time()
                if current_time - self.window_start >= self.time_window:
                    self._reset_fixed_window(current_time)
                return max(0, self.max_requests - self.window_count)
            else:
                return int(self.tokens)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get rate limiter performance metrics."""
        with self.lock:
            return {
                'strategy': self.strategy.value,
                'max_requests': self.max_requests,
                'time_window': self.time_window,
                'available_tokens': self.get_available_tokens(),
                'total_requests': self.metrics.total_requests,
                'allowed_requests': self.metrics.allowed_requests,
                'rejected_requests': self.metrics.rejected_requests,
                'rejection_rate': self.metrics.rejection_rate,
                'average_tokens_consumed': self.metrics.average_tokens_consumed,
                'peak_request_rate': self.metrics.peak_request_rate,
                'last_request': self.metrics.last_request.isoformat() if self.metrics.last_request else None,
                'window_start': self.metrics.window_start.isoformat(),
                'last_reset': self.metrics.last_reset.isoformat()
            }
    
    def reset(self) -> None:
        """Reset rate limiter state."""
        with self.lock:
            self.tokens = float(self.burst_size)
            self.last_replenish = time.time()
            self.request_times.clear()
            self.window_count = 0
            self.window_start = time.time()
            self.metrics.last_reset = datetime.now()
            
            self.logger.info("Rate limiter state reset")
    
    def _token_bucket_acquire(self, tokens: int, current_time: float) -> bool:
        """Token bucket rate limiting algorithm."""
        # Replenish tokens based on time elapsed
        self._replenish_tokens()
        
        # Check if enough tokens available
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        
        return False
    
    def _sliding_window_acquire(self, tokens: int, current_time: float) -> bool:
        """Sliding window rate limiting algorithm."""
        # Remove old requests outside the window
        self._cleanup_sliding_window(current_time)
        
        # Check if adding this request would exceed limit
        if len(self.request_times) + tokens <= self.max_requests:
            # Add request timestamps
            for _ in range(tokens):
                self.request_times.append(current_time)
            return True
        
        return False
    
    def _fixed_window_acquire(self, tokens: int, current_time: float) -> bool:
        """Fixed window rate limiting algorithm."""
        # Check if we need to reset the window
        if current_time - self.window_start >= self.time_window:
            self._reset_fixed_window(current_time)
        
        # Check if adding this request would exceed limit
        if self.window_count + tokens <= self.max_requests:
            self.window_count += tokens
            return True
        
        return False
    
    def _leaky_bucket_acquire(self, tokens: int, current_time: float) -> bool:
        """Leaky bucket rate limiting algorithm."""
        # Similar to token bucket but with constant leak rate
        self._replenish_tokens()
        
        # Check capacity (leaky bucket has fixed capacity)
        if self.tokens + tokens <= self.burst_size:
            self.tokens += tokens
            return True
        
        return False
    
    def _replenish_tokens(self) -> None:
        """Replenish tokens based on elapsed time."""
        current_time = time.time()
        time_elapsed = current_time - self.last_replenish
        
        # Add tokens based on replenishment rate
        tokens_to_add = time_elapsed * self.replenish_rate
        self.tokens = min(self.burst_size, self.tokens + tokens_to_add)
        self.last_replenish = current_time
    
    def _cleanup_sliding_window(self, current_time: float) -> None:
        """Remove old requests from sliding window."""
        cutoff_time = current_time - self.time_window
        
        while self.request_times and self.request_times[0] <= cutoff_time:
            self.request_times.popleft()
    
    def _reset_fixed_window(self, current_time: float) -> None:
        """Reset fixed window counter."""
        self.window_count = 0
        self.window_start = current_time
        self.metrics.window_start = datetime.now()


class AdaptiveRateLimiter(RateLimiter):
    """
    Adaptive rate limiter that adjusts limits based on system performance.
    
    Monitors system health and automatically adjusts rate limits to maintain
    optimal performance under varying load conditions.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize adaptive rate limiter."""
        super().__init__(*args, **kwargs)
        
        # Adaptive configuration
        self.base_max_requests = self.max_requests
        self.adaptation_factor = 0.1  # How much to adjust by
        self.performance_threshold = 0.9  # Performance threshold for adjustments
        self.min_requests = max(1, self.max_requests // 4)
        self.max_requests_limit = self.max_requests * 2
        
        # Performance tracking
        self.performance_samples = deque(maxlen=10)
        self.last_adaptation = time.time()
        self.adaptation_cooldown = 30  # Seconds between adaptations
    
    def update_performance(self, performance_score: float) -> None:
        """
        Update system performance score for adaptive adjustment.
        
        Args:
            performance_score: Performance score (0.0 - 1.0)
        """
        with self.lock:
            self.performance_samples.append(performance_score)
            
            # Consider adaptation if cooldown period has passed
            current_time = time.time()
            if current_time - self.last_adaptation >= self.adaptation_cooldown:
                self._consider_adaptation()
    
    def _consider_adaptation(self) -> None:
        """Consider adapting rate limits based on performance."""
        if len(self.performance_samples) < 3:
            return
        
        avg_performance = sum(self.performance_samples) / len(self.performance_samples)
        
        if avg_performance < self.performance_threshold:
            # Performance degraded - reduce rate limit
            new_limit = max(self.min_requests, 
                          int(self.max_requests * (1 - self.adaptation_factor)))
            
            if new_limit != self.max_requests:
                self.max_requests = new_limit
                self.logger.warning(f"Rate limit reduced to {new_limit} due to performance: {avg_performance:.2f}")
        
        elif avg_performance > 0.95 and self.max_requests < self.base_max_requests:
            # Performance good - increase rate limit
            new_limit = min(self.max_requests_limit,
                          int(self.max_requests * (1 + self.adaptation_factor)))
            
            if new_limit != self.max_requests:
                self.max_requests = new_limit
                self.logger.info(f"Rate limit increased to {new_limit} due to good performance: {avg_performance:.2f}")
        
        self.last_adaptation = time.time()


def rate_limit(max_requests: int = 100,
               time_window: int = 60,
               strategy: RateLimitStrategy = RateLimitStrategy.TOKEN_BUCKET):
    """
    Decorator for applying rate limiting to functions.
    
    Args:
        max_requests: Maximum requests allowed in time window
        time_window: Time window in seconds
        strategy: Rate limiting strategy to use
    
    Example:
        @rate_limit(max_requests=10, time_window=60)
        def api_endpoint():
            # Function that should be rate limited
            pass
    """
    def decorator(func):
        limiter = RateLimiter(
            max_requests=max_requests,
            time_window=time_window,
            strategy=strategy
        )
        
        def wrapper(*args, **kwargs):
            limiter.acquire_or_raise()
            return func(*args, **kwargs)
        
        # Attach limiter to wrapper for external access
        wrapper.rate_limiter = limiter
        return wrapper
    
    return decorator