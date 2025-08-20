"""
Load Balancing and Auto-Scaling Infrastructure for Production Scalability
Provides dynamic load distribution and horizontal scaling capabilities
"""

import asyncio
import logging
import time
import threading
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import psutil
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class LoadBalancingStrategy(Enum):
    """Load balancing strategies"""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    RESPONSE_TIME = "response_time"
    CPU_USAGE = "cpu_usage"
    CUSTOM = "custom"


class InstanceStatus(Enum):
    """Instance status for load balancing"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    DRAINING = "draining"
    OFFLINE = "offline"


@dataclass
class ProcessingInstance:
    """Represents a processing instance in the load balancer"""
    instance_id: str
    endpoint: str
    weight: int = 1
    max_connections: int = 100
    current_connections: int = 0
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    response_time_avg: float = 0.0
    last_health_check: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    status: InstanceStatus = InstanceStatus.HEALTHY
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_available(self) -> bool:
        """Check if instance is available for new requests"""
        return (
            self.status in [InstanceStatus.HEALTHY, InstanceStatus.DEGRADED] and
            self.current_connections < self.max_connections
        )
    
    @property
    def load_factor(self) -> float:
        """Calculate load factor for this instance"""
        connection_load = self.current_connections / self.max_connections if self.max_connections > 0 else 0
        cpu_load = self.cpu_usage / 100.0
        return (connection_load + cpu_load) / 2.0


@dataclass
class LoadBalancingResult:
    """Result of load balancing decision"""
    instance: Optional[ProcessingInstance]
    reason: str
    alternatives_considered: int = 0
    decision_time_ms: float = 0.0


class LoadBalancingAlgorithm(ABC):
    """Abstract base class for load balancing algorithms"""
    
    @abstractmethod
    def select_instance(self, instances: List[ProcessingInstance], 
                       context: Dict[str, Any]) -> LoadBalancingResult:
        """Select the best instance for processing"""
        pass


class RoundRobinBalancer(LoadBalancingAlgorithm):
    """Round-robin load balancing algorithm"""
    
    def __init__(self):
        self.current_index = 0
        self.lock = threading.Lock()
    
    def select_instance(self, instances: List[ProcessingInstance], 
                       context: Dict[str, Any]) -> LoadBalancingResult:
        start_time = time.time()
        
        available_instances = [inst for inst in instances if inst.is_available]
        if not available_instances:
            return LoadBalancingResult(
                instance=None,
                reason="No available instances",
                alternatives_considered=len(instances),
                decision_time_ms=(time.time() - start_time) * 1000
            )
        
        with self.lock:
            selected = available_instances[self.current_index % len(available_instances)]
            self.current_index += 1
        
        return LoadBalancingResult(
            instance=selected,
            reason="Round-robin selection",
            alternatives_considered=len(available_instances),
            decision_time_ms=(time.time() - start_time) * 1000
        )


class LeastConnectionsBalancer(LoadBalancingAlgorithm):
    """Least connections load balancing algorithm"""
    
    def select_instance(self, instances: List[ProcessingInstance], 
                       context: Dict[str, Any]) -> LoadBalancingResult:
        start_time = time.time()
        
        available_instances = [inst for inst in instances if inst.is_available]
        if not available_instances:
            return LoadBalancingResult(
                instance=None,
                reason="No available instances",
                alternatives_considered=len(instances),
                decision_time_ms=(time.time() - start_time) * 1000
            )
        
        # Select instance with least connections
        selected = min(available_instances, key=lambda x: x.current_connections)
        
        return LoadBalancingResult(
            instance=selected,
            reason=f"Least connections ({selected.current_connections})",
            alternatives_considered=len(available_instances),
            decision_time_ms=(time.time() - start_time) * 1000
        )


class WeightedRoundRobinBalancer(LoadBalancingAlgorithm):
    """Weighted round-robin load balancing algorithm"""
    
    def __init__(self):
        self.current_weights = {}
        self.lock = threading.Lock()
    
    def select_instance(self, instances: List[ProcessingInstance], 
                       context: Dict[str, Any]) -> LoadBalancingResult:
        start_time = time.time()
        
        available_instances = [inst for inst in instances if inst.is_available]
        if not available_instances:
            return LoadBalancingResult(
                instance=None,
                reason="No available instances",
                alternatives_considered=len(instances),
                decision_time_ms=(time.time() - start_time) * 1000
            )
        
        with self.lock:
            # Initialize weights if needed
            for inst in available_instances:
                if inst.instance_id not in self.current_weights:
                    self.current_weights[inst.instance_id] = 0
            
            # Increment weights and find max
            max_current_weight = -1
            total_weight = 0
            selected = None
            
            for inst in available_instances:
                total_weight += inst.weight
                self.current_weights[inst.instance_id] += inst.weight
                
                if self.current_weights[inst.instance_id] > max_current_weight:
                    max_current_weight = self.current_weights[inst.instance_id]
                    selected = inst
            
            # Reduce selected instance's current weight
            if selected:
                self.current_weights[selected.instance_id] -= total_weight
        
        return LoadBalancingResult(
            instance=selected,
            reason=f"Weighted round-robin (weight={selected.weight if selected else 0})",
            alternatives_considered=len(available_instances),
            decision_time_ms=(time.time() - start_time) * 1000
        )


class ResponseTimeBalancer(LoadBalancingAlgorithm):
    """Response time based load balancing algorithm"""
    
    def select_instance(self, instances: List[ProcessingInstance], 
                       context: Dict[str, Any]) -> LoadBalancingResult:
        start_time = time.time()
        
        available_instances = [inst for inst in instances if inst.is_available]
        if not available_instances:
            return LoadBalancingResult(
                instance=None,
                reason="No available instances",
                alternatives_considered=len(instances),
                decision_time_ms=(time.time() - start_time) * 1000
            )
        
        # Select instance with lowest response time
        selected = min(available_instances, key=lambda x: x.response_time_avg)
        
        return LoadBalancingResult(
            instance=selected,
            reason=f"Best response time ({selected.response_time_avg:.2f}ms)",
            alternatives_considered=len(available_instances),
            decision_time_ms=(time.time() - start_time) * 1000
        )


class CpuUsageBalancer(LoadBalancingAlgorithm):
    """CPU usage based load balancing algorithm"""
    
    def select_instance(self, instances: List[ProcessingInstance], 
                       context: Dict[str, Any]) -> LoadBalancingResult:
        start_time = time.time()
        
        available_instances = [inst for inst in instances if inst.is_available]
        if not available_instances:
            return LoadBalancingResult(
                instance=None,
                reason="No available instances",
                alternatives_considered=len(instances),
                decision_time_ms=(time.time() - start_time) * 1000
            )
        
        # Select instance with lowest CPU usage
        selected = min(available_instances, key=lambda x: x.cpu_usage)
        
        return LoadBalancingResult(
            instance=selected,
            reason=f"Lowest CPU usage ({selected.cpu_usage:.1f}%)",
            alternatives_considered=len(available_instances),
            decision_time_ms=(time.time() - start_time) * 1000
        )


class HealthChecker:
    """Health checking system for processing instances"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.check_interval = config.get('health_check_interval_seconds', 30)
        self.timeout = config.get('health_check_timeout_seconds', 10)
        self.max_failures = config.get('max_consecutive_failures', 3)
        
        # Health check history
        self.failure_counts = {}
        self.last_checks = {}
        
        # Background health check thread
        self.health_check_thread = None
        self.stop_health_checks = threading.Event()
    
    def start_health_checks(self, instances: List[ProcessingInstance]):
        """Start background health checking"""
        if self.health_check_thread and self.health_check_thread.is_alive():
            return
        
        self.stop_health_checks.clear()
        self.health_check_thread = threading.Thread(
            target=self._health_check_loop, 
            args=(instances,), 
            daemon=True
        )
        self.health_check_thread.start()
        logger.info("Started background health checking")
    
    def stop_health_checking(self):
        """Stop background health checking"""
        if self.health_check_thread:
            self.stop_health_checks.set()
            self.health_check_thread.join(timeout=10)
        
    def check_instance_health(self, instance: ProcessingInstance) -> bool:
        """Check health of a single instance"""
        try:
            # This would typically make HTTP request to health endpoint
            # For now, simulate based on CPU/memory usage
            
            # Update metrics (simulate)
            instance.cpu_usage = psutil.cpu_percent(interval=0.1)
            instance.memory_usage = psutil.virtual_memory().percent
            instance.last_health_check = datetime.now(timezone.utc)
            
            # Determine health status
            if instance.cpu_usage > 95 or instance.memory_usage > 95:
                instance.status = InstanceStatus.UNHEALTHY
                return False
            elif instance.cpu_usage > 80 or instance.memory_usage > 85:
                instance.status = InstanceStatus.DEGRADED
                return True
            else:
                instance.status = InstanceStatus.HEALTHY
                return True
                
        except Exception as e:
            logger.error(f"Health check failed for instance {instance.instance_id}: {e}")
            return False
    
    def _health_check_loop(self, instances: List[ProcessingInstance]):
        """Background health check loop"""
        while not self.stop_health_checks.wait(self.check_interval):
            try:
                for instance in instances:
                    if instance.status == InstanceStatus.OFFLINE:
                        continue
                    
                    is_healthy = self.check_instance_health(instance)
                    instance_id = instance.instance_id
                    
                    if not is_healthy:
                        self.failure_counts[instance_id] = self.failure_counts.get(instance_id, 0) + 1
                        if self.failure_counts[instance_id] >= self.max_failures:
                            instance.status = InstanceStatus.UNHEALTHY
                            logger.warning(f"Instance {instance_id} marked as unhealthy after {self.max_failures} failures")
                    else:
                        self.failure_counts[instance_id] = 0
                        if instance.status == InstanceStatus.UNHEALTHY:
                            instance.status = InstanceStatus.HEALTHY
                            logger.info(f"Instance {instance_id} recovered to healthy status")
                    
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")


class AutoScaler:
    """Auto-scaling system for processing instances"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.min_instances = config.get('min_instances', 1)
        self.max_instances = config.get('max_instances', 10)
        self.target_cpu_percent = config.get('target_cpu_percent', 70)
        self.target_memory_percent = config.get('target_memory_percent', 80)
        self.scale_up_threshold = config.get('scale_up_threshold', 5)  # minutes above target
        self.scale_down_threshold = config.get('scale_down_threshold', 10)  # minutes below target
        
        # Auto-scaling state
        self.scale_up_timer = 0
        self.scale_down_timer = 0
        self.scaling_in_progress = False
        
        # Callbacks for scaling actions
        self.scale_up_callback: Optional[Callable] = None
        self.scale_down_callback: Optional[Callable] = None
        
    def set_scale_callbacks(self, scale_up_fn: Callable, scale_down_fn: Callable):
        """Set callbacks for scaling actions"""
        self.scale_up_callback = scale_up_fn
        self.scale_down_callback = scale_down_fn
    
    def evaluate_scaling(self, instances: List[ProcessingInstance]) -> Dict[str, Any]:
        """Evaluate if scaling action is needed"""
        if self.scaling_in_progress:
            return {'action': 'none', 'reason': 'Scaling already in progress'}
        
        healthy_instances = [inst for inst in instances if inst.status == InstanceStatus.HEALTHY]
        if not healthy_instances:
            return {'action': 'none', 'reason': 'No healthy instances for evaluation'}
        
        # Calculate average metrics
        avg_cpu = sum(inst.cpu_usage for inst in healthy_instances) / len(healthy_instances)
        avg_memory = sum(inst.memory_usage for inst in healthy_instances) / len(healthy_instances)
        avg_connections = sum(inst.current_connections for inst in healthy_instances) / len(healthy_instances)
        max_connections = max(inst.max_connections for inst in healthy_instances)
        connection_utilization = (avg_connections / max_connections) * 100 if max_connections > 0 else 0
        
        current_count = len(healthy_instances)
        
        # Check scale up conditions
        scale_up_needed = (
            (avg_cpu > self.target_cpu_percent or 
             avg_memory > self.target_memory_percent or
             connection_utilization > 80) and
            current_count < self.max_instances
        )
        
        # Check scale down conditions
        scale_down_needed = (
            avg_cpu < (self.target_cpu_percent * 0.5) and
            avg_memory < (self.target_memory_percent * 0.5) and
            connection_utilization < 20 and
            current_count > self.min_instances
        )
        
        if scale_up_needed:
            self.scale_up_timer += 1
            self.scale_down_timer = 0
            
            if self.scale_up_timer >= self.scale_up_threshold:
                return {
                    'action': 'scale_up',
                    'reason': f'High utilization: CPU={avg_cpu:.1f}%, Memory={avg_memory:.1f}%, Connections={connection_utilization:.1f}%',
                    'current_instances': current_count,
                    'target_instances': min(current_count + 1, self.max_instances),
                    'metrics': {
                        'cpu_percent': avg_cpu,
                        'memory_percent': avg_memory,
                        'connection_percent': connection_utilization
                    }
                }
        elif scale_down_needed:
            self.scale_down_timer += 1
            self.scale_up_timer = 0
            
            if self.scale_down_timer >= self.scale_down_threshold:
                return {
                    'action': 'scale_down',
                    'reason': f'Low utilization: CPU={avg_cpu:.1f}%, Memory={avg_memory:.1f}%, Connections={connection_utilization:.1f}%',
                    'current_instances': current_count,
                    'target_instances': max(current_count - 1, self.min_instances),
                    'metrics': {
                        'cpu_percent': avg_cpu,
                        'memory_percent': avg_memory,
                        'connection_percent': connection_utilization
                    }
                }
        else:
            # Reset timers if conditions are not met
            self.scale_up_timer = 0
            self.scale_down_timer = 0
        
        return {
            'action': 'none', 
            'reason': 'Utilization within target range',
            'metrics': {
                'cpu_percent': avg_cpu,
                'memory_percent': avg_memory,
                'connection_percent': connection_utilization
            }
        }
    
    def execute_scaling_decision(self, decision: Dict[str, Any]) -> bool:
        """Execute scaling decision"""
        action = decision.get('action')
        
        if action == 'scale_up' and self.scale_up_callback:
            self.scaling_in_progress = True
            try:
                result = self.scale_up_callback(decision)
                logger.info(f"Scale up executed: {decision['reason']}")
                return result
            finally:
                self.scaling_in_progress = False
                
        elif action == 'scale_down' and self.scale_down_callback:
            self.scaling_in_progress = True
            try:
                result = self.scale_down_callback(decision)
                logger.info(f"Scale down executed: {decision['reason']}")
                return result
            finally:
                self.scaling_in_progress = False
        
        return True


class LoadBalancer:
    """
    Main load balancer with health checking and auto-scaling
    Coordinates traffic distribution across processing instances
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Load balancing strategy
        strategy = config.get('strategy', LoadBalancingStrategy.ROUND_ROBIN.value)
        self.strategy = LoadBalancingStrategy(strategy)
        
        # Initialize balancing algorithm
        self.algorithm = self._create_balancing_algorithm()
        
        # Processing instances
        self.instances: List[ProcessingInstance] = []
        self.instances_lock = threading.Lock()
        
        # Health checker
        health_config = config.get('health_checking', {})
        self.health_checker = HealthChecker(health_config)
        
        # Auto-scaler
        scaling_config = config.get('auto_scaling', {})
        self.auto_scaler = AutoScaler(scaling_config)
        
        # Statistics
        self.request_count = 0
        self.successful_requests = 0
        self.failed_requests = 0
        
        logger.info(f"LoadBalancer initialized with {self.strategy.value} strategy")
    
    def _create_balancing_algorithm(self) -> LoadBalancingAlgorithm:
        """Create load balancing algorithm based on strategy"""
        if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
            return RoundRobinBalancer()
        elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            return LeastConnectionsBalancer()
        elif self.strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
            return WeightedRoundRobinBalancer()
        elif self.strategy == LoadBalancingStrategy.RESPONSE_TIME:
            return ResponseTimeBalancer()
        elif self.strategy == LoadBalancingStrategy.CPU_USAGE:
            return CpuUsageBalancer()
        else:
            return RoundRobinBalancer()  # Default fallback
    
    def add_instance(self, instance: ProcessingInstance):
        """Add processing instance to load balancer"""
        with self.instances_lock:
            self.instances.append(instance)
        logger.info(f"Added instance: {instance.instance_id}")
    
    def remove_instance(self, instance_id: str):
        """Remove processing instance from load balancer"""
        with self.instances_lock:
            self.instances = [inst for inst in self.instances if inst.instance_id != instance_id]
        logger.info(f"Removed instance: {instance_id}")
    
    def start(self):
        """Start load balancer services"""
        self.health_checker.start_health_checks(self.instances)
        logger.info("LoadBalancer started")
    
    def stop(self):
        """Stop load balancer services"""
        self.health_checker.stop_health_checking()
        logger.info("LoadBalancer stopped")
    
    def select_instance_for_request(self, context: Optional[Dict[str, Any]] = None) -> LoadBalancingResult:
        """Select the best instance for processing a request"""
        self.request_count += 1
        
        with self.instances_lock:
            instances_copy = self.instances.copy()
        
        result = self.algorithm.select_instance(instances_copy, context or {})
        
        if result.instance:
            result.instance.current_connections += 1
            self.successful_requests += 1
        else:
            self.failed_requests += 1
        
        return result
    
    def release_instance_connection(self, instance_id: str):
        """Release connection from instance after request completion"""
        with self.instances_lock:
            for instance in self.instances:
                if instance.instance_id == instance_id:
                    instance.current_connections = max(0, instance.current_connections - 1)
                    break
    
    def update_instance_metrics(self, instance_id: str, metrics: Dict[str, Any]):
        """Update instance performance metrics"""
        with self.instances_lock:
            for instance in self.instances:
                if instance.instance_id == instance_id:
                    instance.cpu_usage = metrics.get('cpu_usage', instance.cpu_usage)
                    instance.memory_usage = metrics.get('memory_usage', instance.memory_usage)
                    instance.response_time_avg = metrics.get('response_time_avg', instance.response_time_avg)
                    break
    
    def get_load_balancer_stats(self) -> Dict[str, Any]:
        """Get load balancer statistics"""
        with self.instances_lock:
            healthy_instances = len([inst for inst in self.instances if inst.status == InstanceStatus.HEALTHY])
            total_connections = sum(inst.current_connections for inst in self.instances)
            avg_cpu = sum(inst.cpu_usage for inst in self.instances) / len(self.instances) if self.instances else 0
            avg_memory = sum(inst.memory_usage for inst in self.instances) / len(self.instances) if self.instances else 0
        
        success_rate = (self.successful_requests / self.request_count * 100) if self.request_count > 0 else 0
        
        return {
            'strategy': self.strategy.value,
            'total_instances': len(self.instances),
            'healthy_instances': healthy_instances,
            'total_requests': self.request_count,
            'successful_requests': self.successful_requests,
            'failed_requests': self.failed_requests,
            'success_rate_percent': success_rate,
            'total_active_connections': total_connections,
            'average_cpu_usage': avg_cpu,
            'average_memory_usage': avg_memory
        }
    
    def evaluate_auto_scaling(self) -> Dict[str, Any]:
        """Evaluate and potentially execute auto-scaling"""
        with self.instances_lock:
            instances_copy = self.instances.copy()
        
        decision = self.auto_scaler.evaluate_scaling(instances_copy)
        
        if decision['action'] != 'none':
            self.auto_scaler.execute_scaling_decision(decision)
        
        return decision


# Context manager for request processing
class LoadBalancedRequest:
    """Context manager for load-balanced request processing"""
    
    def __init__(self, load_balancer: LoadBalancer, request_context: Optional[Dict[str, Any]] = None):
        self.load_balancer = load_balancer
        self.request_context = request_context
        self.selected_instance = None
        self.start_time = None
    
    def __enter__(self) -> Optional[ProcessingInstance]:
        self.start_time = time.time()
        result = self.load_balancer.select_instance_for_request(self.request_context)
        self.selected_instance = result.instance
        return self.selected_instance
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.selected_instance and self.start_time:
            # Update response time
            response_time = (time.time() - self.start_time) * 1000
            self.load_balancer.update_instance_metrics(
                self.selected_instance.instance_id,
                {'response_time_avg': response_time}
            )
            
            # Release connection
            self.load_balancer.release_instance_connection(self.selected_instance.instance_id)