"""
Distributed Processing System for Scalable SRT Processing
Provides distributed task processing, queue management, and result aggregation
"""

import asyncio
import logging
import json
import time
import threading
import uuid
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import pickle
import hashlib
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import queue
import redis
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Task processing status"""
    PENDING = "pending"
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


class TaskPriority(Enum):
    """Task priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4
    CRITICAL = 5


@dataclass
class ProcessingTask:
    """Represents a processing task in the distributed system"""
    task_id: str
    task_type: str
    input_data: Dict[str, Any]
    priority: TaskPriority = TaskPriority.NORMAL
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    worker_id: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    timeout_seconds: int = 300
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def processing_time(self) -> Optional[float]:
        """Calculate processing time in seconds"""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None
    
    @property
    def is_expired(self) -> bool:
        """Check if task has expired"""
        if self.started_at:
            return datetime.now(timezone.utc) > self.started_at + timedelta(seconds=self.timeout_seconds)
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary for serialization"""
        data = asdict(self)
        # Convert datetime objects to ISO strings
        for key, value in data.items():
            if isinstance(value, datetime):
                data[key] = value.isoformat() if value else None
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProcessingTask':
        """Create task from dictionary"""
        # Convert ISO strings back to datetime objects
        for key in ['created_at', 'started_at', 'completed_at']:
            if data.get(key):
                data[key] = datetime.fromisoformat(data[key])
        
        # Convert enums
        if 'priority' in data:
            data['priority'] = TaskPriority(data['priority']) if isinstance(data['priority'], int) else TaskPriority[data['priority']]
        if 'status' in data:
            data['status'] = TaskStatus(data['status']) if isinstance(data['status'], str) else data['status']
        
        return cls(**data)


@dataclass
class WorkerNode:
    """Represents a worker node in the distributed system"""
    worker_id: str
    node_name: str
    capabilities: List[str]
    max_concurrent_tasks: int = 4
    current_task_count: int = 0
    cpu_cores: int = 4
    memory_gb: int = 8
    last_heartbeat: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    status: str = "active"  # active, busy, idle, offline
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_available(self) -> bool:
        """Check if worker is available for new tasks"""
        return (
            self.status in ["active", "idle"] and
            self.current_task_count < self.max_concurrent_tasks and
            self.is_alive
        )
    
    @property
    def is_alive(self) -> bool:
        """Check if worker is alive (recent heartbeat)"""
        return datetime.now(timezone.utc) - self.last_heartbeat < timedelta(minutes=2)
    
    @property
    def load_percentage(self) -> float:
        """Calculate current load percentage"""
        return (self.current_task_count / self.max_concurrent_tasks) * 100


class TaskQueue(ABC):
    """Abstract base class for task queues"""
    
    @abstractmethod
    def enqueue(self, task: ProcessingTask) -> bool:
        """Add task to queue"""
        pass
    
    @abstractmethod
    def dequeue(self, worker_capabilities: List[str]) -> Optional[ProcessingTask]:
        """Get next task from queue"""
        pass
    
    @abstractmethod
    def update_task_status(self, task_id: str, status: TaskStatus, 
                          result: Optional[Dict[str, Any]] = None, 
                          error: Optional[str] = None):
        """Update task status"""
        pass
    
    @abstractmethod
    def get_task(self, task_id: str) -> Optional[ProcessingTask]:
        """Get task by ID"""
        pass
    
    @abstractmethod
    def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics"""
        pass


class InMemoryTaskQueue(TaskQueue):
    """In-memory task queue implementation"""
    
    def __init__(self):
        self.tasks: Dict[str, ProcessingTask] = {}
        self.pending_queue = queue.PriorityQueue()
        self.lock = threading.Lock()
    
    def enqueue(self, task: ProcessingTask) -> bool:
        """Add task to queue"""
        try:
            with self.lock:
                self.tasks[task.task_id] = task
                # Use negative priority for correct ordering (higher priority first)
                priority_value = -task.priority.value
                self.pending_queue.put((priority_value, task.created_at.timestamp(), task.task_id))
                task.status = TaskStatus.QUEUED
            return True
        except Exception as e:
            logger.error(f"Failed to enqueue task {task.task_id}: {e}")
            return False
    
    def dequeue(self, worker_capabilities: List[str]) -> Optional[ProcessingTask]:
        """Get next task from queue"""
        try:
            # Get next task that matches worker capabilities
            temp_items = []
            task = None
            
            with self.lock:
                while not self.pending_queue.empty():
                    priority, created_timestamp, task_id = self.pending_queue.get()
                    
                    if task_id in self.tasks:
                        candidate = self.tasks[task_id]
                        
                        # Check if task matches worker capabilities
                        if self._task_matches_capabilities(candidate, worker_capabilities):
                            candidate.status = TaskStatus.PROCESSING
                            candidate.started_at = datetime.now(timezone.utc)
                            task = candidate
                            break
                        else:
                            # Put back in queue
                            temp_items.append((priority, created_timestamp, task_id))
                
                # Put back items that didn't match
                for item in temp_items:
                    self.pending_queue.put(item)
            
            return task
            
        except Exception as e:
            logger.error(f"Failed to dequeue task: {e}")
            return None
    
    def update_task_status(self, task_id: str, status: TaskStatus, 
                          result: Optional[Dict[str, Any]] = None, 
                          error: Optional[str] = None):
        """Update task status"""
        with self.lock:
            if task_id in self.tasks:
                task = self.tasks[task_id]
                task.status = status
                
                if result:
                    task.result = result
                if error:
                    task.error = error
                
                if status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                    task.completed_at = datetime.now(timezone.utc)
    
    def get_task(self, task_id: str) -> Optional[ProcessingTask]:
        """Get task by ID"""
        with self.lock:
            return self.tasks.get(task_id)
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics"""
        with self.lock:
            total_tasks = len(self.tasks)
            status_counts = {}
            
            for task in self.tasks.values():
                status = task.status.value
                status_counts[status] = status_counts.get(status, 0) + 1
            
            return {
                'total_tasks': total_tasks,
                'pending_queue_size': self.pending_queue.qsize(),
                'status_counts': status_counts
            }
    
    def _task_matches_capabilities(self, task: ProcessingTask, capabilities: List[str]) -> bool:
        """Check if task can be handled by worker with given capabilities"""
        # For now, assume all workers can handle all task types
        # In a real system, this would check task requirements against worker capabilities
        return True


class RedisTaskQueue(TaskQueue):
    """Redis-based distributed task queue"""
    
    def __init__(self, redis_config: Dict[str, Any]):
        self.redis_client = redis.Redis(
            host=redis_config.get('host', 'localhost'),
            port=redis_config.get('port', 6379),
            password=redis_config.get('password'),
            db=redis_config.get('db', 0),
            decode_responses=True
        )
        self.queue_key = "asr:task_queue"
        self.task_key_prefix = "asr:task:"
        
    def enqueue(self, task: ProcessingTask) -> bool:
        """Add task to Redis queue"""
        try:
            # Store task data
            task_key = f"{self.task_key_prefix}{task.task_id}"
            task_data = json.dumps(task.to_dict())
            self.redis_client.set(task_key, task_data)
            
            # Add to priority queue
            priority_score = -task.priority.value  # Negative for correct ordering
            self.redis_client.zadd(self.queue_key, {task.task_id: priority_score})
            
            task.status = TaskStatus.QUEUED
            return True
            
        except Exception as e:
            logger.error(f"Failed to enqueue task {task.task_id} to Redis: {e}")
            return False
    
    def dequeue(self, worker_capabilities: List[str]) -> Optional[ProcessingTask]:
        """Get next task from Redis queue"""
        try:
            # Get highest priority task
            result = self.redis_client.zpopmax(self.queue_key)
            if not result:
                return None
            
            task_id, _ = result[0]
            
            # Get task data
            task_key = f"{self.task_key_prefix}{task_id}"
            task_data = self.redis_client.get(task_key)
            
            if task_data:
                task_dict = json.loads(task_data)
                task = ProcessingTask.from_dict(task_dict)
                
                # Update status
                task.status = TaskStatus.PROCESSING
                task.started_at = datetime.now(timezone.utc)
                
                # Save updated task
                self.redis_client.set(task_key, json.dumps(task.to_dict()))
                
                return task
                
        except Exception as e:
            logger.error(f"Failed to dequeue task from Redis: {e}")
            
        return None
    
    def update_task_status(self, task_id: str, status: TaskStatus, 
                          result: Optional[Dict[str, Any]] = None, 
                          error: Optional[str] = None):
        """Update task status in Redis"""
        try:
            task_key = f"{self.task_key_prefix}{task_id}"
            task_data = self.redis_client.get(task_key)
            
            if task_data:
                task_dict = json.loads(task_data)
                task = ProcessingTask.from_dict(task_dict)
                
                task.status = status
                if result:
                    task.result = result
                if error:
                    task.error = error
                
                if status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                    task.completed_at = datetime.now(timezone.utc)
                
                self.redis_client.set(task_key, json.dumps(task.to_dict()))
                
        except Exception as e:
            logger.error(f"Failed to update task status in Redis: {e}")
    
    def get_task(self, task_id: str) -> Optional[ProcessingTask]:
        """Get task by ID from Redis"""
        try:
            task_key = f"{self.task_key_prefix}{task_id}"
            task_data = self.redis_client.get(task_key)
            
            if task_data:
                task_dict = json.loads(task_data)
                return ProcessingTask.from_dict(task_dict)
                
        except Exception as e:
            logger.error(f"Failed to get task {task_id} from Redis: {e}")
        
        return None
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics from Redis"""
        try:
            # Get all task keys
            task_keys = self.redis_client.keys(f"{self.task_key_prefix}*")
            
            total_tasks = len(task_keys)
            pending_queue_size = self.redis_client.zcard(self.queue_key)
            
            # Count by status (this could be expensive for large numbers of tasks)
            status_counts = {}
            for task_key in task_keys[:1000]:  # Limit to avoid performance issues
                task_data = self.redis_client.get(task_key)
                if task_data:
                    task_dict = json.loads(task_data)
                    status = task_dict.get('status', 'unknown')
                    status_counts[status] = status_counts.get(status, 0) + 1
            
            return {
                'total_tasks': total_tasks,
                'pending_queue_size': pending_queue_size,
                'status_counts': status_counts
            }
            
        except Exception as e:
            logger.error(f"Failed to get queue stats from Redis: {e}")
            return {'error': str(e)}


class TaskProcessor(ABC):
    """Abstract base class for task processors"""
    
    @abstractmethod
    def process_task(self, task: ProcessingTask) -> Dict[str, Any]:
        """Process a task and return result"""
        pass
    
    @abstractmethod
    def get_supported_task_types(self) -> List[str]:
        """Get list of supported task types"""
        pass


class SRTProcessor(TaskProcessor):
    """SRT file processing task processor"""
    
    def __init__(self):
        # Import here to avoid circular dependencies
        from post_processors.sanskrit_post_processor import SanskritPostProcessor
        self.processor = SanskritPostProcessor()
    
    def process_task(self, task: ProcessingTask) -> Dict[str, Any]:
        """Process SRT file task"""
        try:
            input_data = task.input_data
            input_file = Path(input_data['input_file'])
            output_file = Path(input_data['output_file'])
            
            # Process the SRT file
            metrics = self.processor.process_srt_file(input_file, output_file)
            
            return {
                'success': True,
                'output_file': str(output_file),
                'metrics': {
                    'total_segments': metrics.total_segments,
                    'segments_modified': metrics.segments_modified,
                    'processing_time': metrics.processing_time,
                    'average_confidence': metrics.average_confidence
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_supported_task_types(self) -> List[str]:
        """Get supported task types"""
        return ['srt_processing', 'sanskrit_correction', 'text_normalization']


class WorkerManager:
    """Manages worker nodes and task distribution"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.workers: Dict[str, WorkerNode] = {}
        self.task_processors: Dict[str, TaskProcessor] = {}
        self.lock = threading.Lock()
        
        # Register default processors
        self._register_default_processors()
        
    def _register_default_processors(self):
        """Register default task processors"""
        srt_processor = SRTProcessor()
        for task_type in srt_processor.get_supported_task_types():
            self.task_processors[task_type] = srt_processor
    
    def register_worker(self, worker: WorkerNode):
        """Register a worker node"""
        with self.lock:
            self.workers[worker.worker_id] = worker
        logger.info(f"Registered worker: {worker.worker_id} ({worker.node_name})")
    
    def unregister_worker(self, worker_id: str):
        """Unregister a worker node"""
        with self.lock:
            if worker_id in self.workers:
                del self.workers[worker_id]
        logger.info(f"Unregistered worker: {worker_id}")
    
    def update_worker_heartbeat(self, worker_id: str):
        """Update worker heartbeat"""
        with self.lock:
            if worker_id in self.workers:
                self.workers[worker_id].last_heartbeat = datetime.now(timezone.utc)
    
    def get_available_workers(self, task_type: Optional[str] = None) -> List[WorkerNode]:
        """Get list of available workers"""
        with self.lock:
            available = []
            for worker in self.workers.values():
                if worker.is_available:
                    if task_type is None or task_type in worker.capabilities:
                        available.append(worker)
            return available
    
    def select_best_worker(self, task: ProcessingTask) -> Optional[WorkerNode]:
        """Select the best worker for a task"""
        available_workers = self.get_available_workers(task.task_type)
        
        if not available_workers:
            return None
        
        # Select worker with lowest load
        return min(available_workers, key=lambda w: w.load_percentage)
    
    def get_worker_stats(self) -> Dict[str, Any]:
        """Get worker statistics"""
        with self.lock:
            total_workers = len(self.workers)
            available_workers = len([w for w in self.workers.values() if w.is_available])
            busy_workers = len([w for w in self.workers.values() if w.status == "busy"])
            offline_workers = len([w for w in self.workers.values() if not w.is_alive])
            
            total_capacity = sum(w.max_concurrent_tasks for w in self.workers.values())
            current_load = sum(w.current_task_count for w in self.workers.values())
            
            return {
                'total_workers': total_workers,
                'available_workers': available_workers,
                'busy_workers': busy_workers,
                'offline_workers': offline_workers,
                'total_capacity': total_capacity,
                'current_load': current_load,
                'utilization_percent': (current_load / total_capacity * 100) if total_capacity > 0 else 0
            }


class DistributedProcessor:
    """
    Main distributed processing system
    Coordinates tasks, workers, and results
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Task queue
        if config.get('redis'):
            self.task_queue = RedisTaskQueue(config['redis'])
        else:
            self.task_queue = InMemoryTaskQueue()
        
        # Worker manager
        self.worker_manager = WorkerManager(config)
        
        # Task dispatcher
        self.dispatcher_thread = None
        self.stop_dispatcher = threading.Event()
        self.dispatch_interval = config.get('dispatch_interval_seconds', 1)
        
        # Statistics
        self.stats = {
            'tasks_processed': 0,
            'tasks_failed': 0,
            'average_processing_time': 0.0,
            'total_processing_time': 0.0
        }
        self.stats_lock = threading.Lock()
        
        logger.info("DistributedProcessor initialized")
    
    def start(self):
        """Start the distributed processor"""
        if self.dispatcher_thread and self.dispatcher_thread.is_alive():
            return
        
        self.stop_dispatcher.clear()
        self.dispatcher_thread = threading.Thread(target=self._dispatch_loop, daemon=True)
        self.dispatcher_thread.start()
        logger.info("DistributedProcessor started")
    
    def stop(self):
        """Stop the distributed processor"""
        if self.dispatcher_thread:
            self.stop_dispatcher.set()
            self.dispatcher_thread.join(timeout=10)
        logger.info("DistributedProcessor stopped")
    
    def submit_task(self, task_type: str, input_data: Dict[str, Any], 
                   priority: TaskPriority = TaskPriority.NORMAL,
                   timeout_seconds: int = 300) -> str:
        """Submit a new task for processing"""
        task = ProcessingTask(
            task_id=str(uuid.uuid4()),
            task_type=task_type,
            input_data=input_data,
            priority=priority,
            timeout_seconds=timeout_seconds
        )
        
        if self.task_queue.enqueue(task):
            logger.info(f"Submitted task: {task.task_id} ({task_type})")
            return task.task_id
        else:
            logger.error(f"Failed to submit task: {task.task_id}")
            return None
    
    def get_task_status(self, task_id: str) -> Optional[ProcessingTask]:
        """Get task status"""
        return self.task_queue.get_task(task_id)
    
    def wait_for_task_completion(self, task_id: str, timeout_seconds: int = 300) -> Optional[Dict[str, Any]]:
        """Wait for task completion and return result"""
        start_time = time.time()
        
        while time.time() - start_time < timeout_seconds:
            task = self.get_task_status(task_id)
            
            if task is None:
                return None
            
            if task.status == TaskStatus.COMPLETED:
                return task.result
            elif task.status == TaskStatus.FAILED:
                return {'error': task.error}
            
            time.sleep(1)  # Poll every second
        
        return {'error': 'Task timeout'}
    
    def process_srt_file_distributed(self, input_file: str, output_file: str,
                                   priority: TaskPriority = TaskPriority.NORMAL) -> str:
        """Submit SRT file for distributed processing"""
        return self.submit_task(
            task_type='srt_processing',
            input_data={
                'input_file': input_file,
                'output_file': output_file
            },
            priority=priority
        )
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        queue_stats = self.task_queue.get_queue_stats()
        worker_stats = self.worker_manager.get_worker_stats()
        
        with self.stats_lock:
            processing_stats = self.stats.copy()
        
        return {
            'queue': queue_stats,
            'workers': worker_stats,
            'processing': processing_stats,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    
    def _dispatch_loop(self):
        """Main task dispatching loop"""
        while not self.stop_dispatcher.wait(self.dispatch_interval):
            try:
                # Get next task from queue
                available_workers = self.worker_manager.get_available_workers()
                if not available_workers:
                    continue
                
                # Select a worker based on capabilities
                worker_capabilities = []
                for worker in available_workers:
                    worker_capabilities.extend(worker.capabilities)
                worker_capabilities = list(set(worker_capabilities))  # Remove duplicates
                
                task = self.task_queue.dequeue(worker_capabilities)
                if not task:
                    continue
                
                # Find best worker for this task
                worker = self.worker_manager.select_best_worker(task)
                if not worker:
                    # Put task back in queue
                    task.status = TaskStatus.PENDING
                    self.task_queue.enqueue(task)
                    continue
                
                # Assign task to worker
                task.worker_id = worker.worker_id
                worker.current_task_count += 1
                worker.status = "busy" if worker.current_task_count >= worker.max_concurrent_tasks else "active"
                
                # Process task in thread pool
                threading.Thread(
                    target=self._process_task_async,
                    args=(task, worker),
                    daemon=True
                ).start()
                
            except Exception as e:
                logger.error(f"Error in dispatch loop: {e}")
    
    def _process_task_async(self, task: ProcessingTask, worker: WorkerNode):
        """Process task asynchronously"""
        try:
            # Get appropriate processor
            processor = self.worker_manager.task_processors.get(task.task_type)
            if not processor:
                self.task_queue.update_task_status(
                    task.task_id, 
                    TaskStatus.FAILED, 
                    error=f"No processor available for task type: {task.task_type}"
                )
                return
            
            # Process the task
            result = processor.process_task(task)
            
            if result.get('success', False):
                self.task_queue.update_task_status(task.task_id, TaskStatus.COMPLETED, result=result)
                
                # Update statistics
                with self.stats_lock:
                    self.stats['tasks_processed'] += 1
                    if task.processing_time:
                        self.stats['total_processing_time'] += task.processing_time
                        self.stats['average_processing_time'] = (
                            self.stats['total_processing_time'] / self.stats['tasks_processed']
                        )
                
                logger.info(f"Task completed: {task.task_id}")
            else:
                self.task_queue.update_task_status(
                    task.task_id, 
                    TaskStatus.FAILED, 
                    error=result.get('error', 'Unknown error')
                )
                
                with self.stats_lock:
                    self.stats['tasks_failed'] += 1
                
                logger.error(f"Task failed: {task.task_id} - {result.get('error')}")
                
        except Exception as e:
            self.task_queue.update_task_status(task.task_id, TaskStatus.FAILED, error=str(e))
            
            with self.stats_lock:
                self.stats['tasks_failed'] += 1
            
            logger.error(f"Error processing task {task.task_id}: {e}")
        
        finally:
            # Release worker
            worker.current_task_count = max(0, worker.current_task_count - 1)
            worker.status = "idle" if worker.current_task_count == 0 else "active"


# Utility functions for distributed processing
def create_distributed_processor(config: Dict[str, Any]) -> DistributedProcessor:
    """Create and configure distributed processor"""
    return DistributedProcessor(config)


def process_files_in_batch(processor: DistributedProcessor, 
                          file_pairs: List[tuple],
                          priority: TaskPriority = TaskPriority.NORMAL) -> List[str]:
    """Process multiple SRT files in parallel"""
    task_ids = []
    
    for input_file, output_file in file_pairs:
        task_id = processor.process_srt_file_distributed(
            str(input_file), 
            str(output_file), 
            priority
        )
        if task_id:
            task_ids.append(task_id)
    
    return task_ids


def wait_for_batch_completion(processor: DistributedProcessor, 
                            task_ids: List[str],
                            timeout_seconds: int = 600) -> Dict[str, Dict[str, Any]]:
    """Wait for batch of tasks to complete"""
    results = {}
    
    for task_id in task_ids:
        result = processor.wait_for_task_completion(task_id, timeout_seconds)
        results[task_id] = result
    
    return results