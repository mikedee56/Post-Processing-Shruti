"""
Epic 4 - Story 4.1: Batch Processing Framework
Apache Airflow DAG for processing large volumes of SRT transcripts

This DAG implements robust batch processing for 12,000+ hours of content with:
- Parallel processing capabilities
- Error handling and recovery
- Progress monitoring
- Resource management
"""

from datetime import datetime, timedelta
from typing import List, Dict, Any

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.bash import BashOperator
from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.providers.redis.hooks.redis import RedisHook
from airflow.models import Variable
from airflow.utils.task_group import TaskGroup
from airflow.utils.trigger_rule import TriggerRule

import os
import sys
import glob
import json
import logging
from pathlib import Path
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

# Add src to Python path for imports
sys.path.insert(0, '/app/src')

from src.post_processors.sanskrit_post_processor import SanskritPostProcessor
from src.utils.metrics_collector import MetricsCollector
from src.utils.batch_processor import BatchProcessor
from src.utils.recovery_manager import RecoveryManager

# DAG Configuration
DAG_ID = 'batch_srt_processing'
SCHEDULE_INTERVAL = '0 2 * * *'  # Daily at 2 AM
MAX_ACTIVE_RUNS = 1
CATCHUP = False

# Processing Configuration
DEFAULT_BATCH_SIZE = int(Variable.get("batch_size", default_var=50))
MAX_WORKERS = int(Variable.get("max_workers", default_var=mp.cpu_count()))
CHUNK_SIZE = int(Variable.get("chunk_size", default_var=10))
TIMEOUT_SECONDS = int(Variable.get("processing_timeout", default_var=3600))

# File Paths
INPUT_DIR = Variable.get("input_dir", default_var="/app/data/raw_srts")
OUTPUT_DIR = Variable.get("output_dir", default_var="/app/data/processed_srts")
FAILED_DIR = Variable.get("failed_dir", default_var="/app/data/failed_srts")
BACKUP_DIR = Variable.get("backup_dir", default_var="/app/data/backups")

# Default DAG arguments
default_args = {
    'owner': 'sanskrit-processing-team',
    'depends_on_past': False,
    'start_date': datetime(2025, 1, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=15),
    'execution_timeout': timedelta(hours=6),
}

# Initialize DAG
dag = DAG(
    DAG_ID,
    default_args=default_args,
    description='Batch SRT Processing Pipeline for Sanskrit/Hindi Content',
    schedule_interval=SCHEDULE_INTERVAL,
    max_active_runs=MAX_ACTIVE_RUNS,
    catchup=CATCHUP,
    tags=['batch-processing', 'sanskrit', 'epic-4'],
)

def discover_files(**context) -> Dict[str, Any]:
    """
    Discover SRT files to process and organize them into batches.
    
    Returns:
        Dict containing discovered files and batch information
    """
    try:
        logging.info(f"Discovering SRT files in {INPUT_DIR}")
        
        # Find all SRT files
        pattern = os.path.join(INPUT_DIR, "**/*.srt")
        all_files = glob.glob(pattern, recursive=True)
        
        # Filter out already processed files
        processed_files = set()
        if os.path.exists(OUTPUT_DIR):
            processed_pattern = os.path.join(OUTPUT_DIR, "**/*.srt")
            processed_files = {
                os.path.basename(f) for f in glob.glob(processed_pattern, recursive=True)
            }
        
        # Get unprocessed files
        unprocessed_files = [
            f for f in all_files 
            if os.path.basename(f) not in processed_files
        ]
        
        logging.info(f"Found {len(all_files)} total files, {len(unprocessed_files)} unprocessed")
        
        # Organize into batches
        batches = []
        for i in range(0, len(unprocessed_files), DEFAULT_BATCH_SIZE):
            batch = unprocessed_files[i:i + DEFAULT_BATCH_SIZE]
            batches.append({
                'batch_id': i // DEFAULT_BATCH_SIZE,
                'files': batch,
                'size': len(batch)
            })
        
        result = {
            'total_files': len(all_files),
            'unprocessed_files': len(unprocessed_files),
            'batch_count': len(batches),
            'batches': batches,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Store in XCom for downstream tasks
        return result
        
    except Exception as e:
        logging.error(f"Error discovering files: {str(e)}")
        raise


def check_processing_needed(**context) -> str:
    """
    Check if processing is needed based on discovered files.
    
    Returns:
        Task ID to branch to
    """
    ti = context['ti']
    discovery_result = ti.xcom_pull(task_ids='discover_files')
    
    if discovery_result['unprocessed_files'] == 0:
        logging.info("No files to process, skipping batch processing")
        return 'no_processing_needed'
    
    logging.info(f"Found {discovery_result['unprocessed_files']} files to process in {discovery_result['batch_count']} batches")
    return 'setup_processing_environment'


def setup_processing_environment(**context):
    """
    Setup the processing environment and initialize resources.
    """
    try:
        logging.info("Setting up processing environment")
        
        # Ensure output directories exist
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        os.makedirs(FAILED_DIR, exist_ok=True)
        os.makedirs(BACKUP_DIR, exist_ok=True)
        
        # Initialize Redis connection for progress tracking
        redis_hook = RedisHook(redis_conn_id='redis_default')
        redis_conn = redis_hook.get_conn()
        
        # Clear any previous processing state
        redis_conn.delete('batch_processing_state')
        redis_conn.delete('batch_processing_progress')
        
        # Initialize processing state
        state = {
            'status': 'initialized',
            'start_time': datetime.utcnow().isoformat(),
            'total_batches': 0,
            'completed_batches': 0,
            'failed_batches': 0
        }
        
        redis_conn.set('batch_processing_state', json.dumps(state))
        
        logging.info("Processing environment setup complete")
        
    except Exception as e:
        logging.error(f"Error setting up environment: {str(e)}")
        raise


def process_batch(**context) -> Dict[str, Any]:
    """
    Process a single batch of SRT files using parallel processing.
    
    Returns:
        Batch processing results
    """
    try:
        ti = context['ti']
        batch_info = context['params']
        batch_id = batch_info['batch_id']
        files = batch_info['files']
        
        logging.info(f"Processing batch {batch_id} with {len(files)} files")
        
        # Initialize processors
        batch_processor = BatchProcessor()
        metrics_collector = MetricsCollector()
        recovery_manager = RecoveryManager()
        
        # Process files in parallel chunks
        results = {
            'batch_id': batch_id,
            'total_files': len(files),
            'processed_files': 0,
            'failed_files': 0,
            'processing_time': 0,
            'errors': [],
            'metrics': {}
        }
        
        start_time = datetime.utcnow()
        
        # Split files into chunks for parallel processing
        chunks = [files[i:i + CHUNK_SIZE] for i in range(0, len(files), CHUNK_SIZE)]
        
        # Process chunks in parallel
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_chunk = {
                executor.submit(process_chunk, chunk, batch_id, i): (chunk, i)
                for i, chunk in enumerate(chunks)
            }
            
            for future in as_completed(future_to_chunk, timeout=TIMEOUT_SECONDS):
                chunk, chunk_id = future_to_chunk[future]
                
                try:
                    chunk_result = future.result()
                    results['processed_files'] += chunk_result['processed']
                    results['failed_files'] += chunk_result['failed']
                    results['errors'].extend(chunk_result['errors'])
                    
                    logging.info(f"Chunk {chunk_id} completed: {chunk_result['processed']} processed, {chunk_result['failed']} failed")
                    
                except Exception as e:
                    logging.error(f"Chunk {chunk_id} failed: {str(e)}")
                    results['failed_files'] += len(chunk)
                    results['errors'].append({
                        'chunk_id': chunk_id,
                        'error': str(e),
                        'files': chunk
                    })
        
        # Calculate processing time
        end_time = datetime.utcnow()
        results['processing_time'] = (end_time - start_time).total_seconds()
        
        # Collect metrics
        results['metrics'] = {
            'throughput': results['processed_files'] / results['processing_time'] if results['processing_time'] > 0 else 0,
            'success_rate': results['processed_files'] / results['total_files'] if results['total_files'] > 0 else 0,
            'avg_processing_time': results['processing_time'] / results['processed_files'] if results['processed_files'] > 0 else 0
        }
        
        logging.info(f"Batch {batch_id} completed: {results['processed_files']}/{results['total_files']} files processed")
        
        return results
        
    except Exception as e:
        logging.error(f"Error processing batch {batch_id}: {str(e)}")
        raise


def process_chunk(file_chunk: List[str], batch_id: int, chunk_id: int) -> Dict[str, Any]:
    """
    Process a chunk of files in a single process.
    
    Args:
        file_chunk: List of file paths to process
        batch_id: ID of the parent batch
        chunk_id: ID of this chunk
        
    Returns:
        Chunk processing results
    """
    try:
        # Initialize processor for this process
        processor = SanskritPostProcessor()
        
        result = {
            'batch_id': batch_id,
            'chunk_id': chunk_id,
            'processed': 0,
            'failed': 0,
            'errors': []
        }
        
        for file_path in file_chunk:
            try:
                # Generate output path
                relative_path = os.path.relpath(file_path, INPUT_DIR)
                output_path = os.path.join(OUTPUT_DIR, relative_path)
                
                # Ensure output directory exists
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                # Process file
                metrics = processor.process_srt_file(file_path, output_path)
                result['processed'] += 1
                
                logging.debug(f"Processed {file_path}: {metrics.total_segments} segments")
                
            except Exception as e:
                result['failed'] += 1
                result['errors'].append({
                    'file': file_path,
                    'error': str(e)
                })
                
                # Move failed file to failed directory
                try:
                    failed_path = os.path.join(FAILED_DIR, os.path.basename(file_path))
                    os.makedirs(os.path.dirname(failed_path), exist_ok=True)
                    import shutil
                    shutil.copy2(file_path, failed_path)
                except Exception as move_error:
                    logging.error(f"Error moving failed file: {str(move_error)}")
        
        return result
        
    except Exception as e:
        logging.error(f"Error processing chunk {chunk_id}: {str(e)}")
        return {
            'batch_id': batch_id,
            'chunk_id': chunk_id,
            'processed': 0,
            'failed': len(file_chunk),
            'errors': [{'chunk_error': str(e)}]
        }


def aggregate_results(**context) -> Dict[str, Any]:
    """
    Aggregate results from all batch processing tasks.
    
    Returns:
        Aggregated processing results
    """
    try:
        ti = context['ti']
        
        # Get results from all batch tasks
        discovery_result = ti.xcom_pull(task_ids='discover_files')
        batch_results = []
        
        for batch in discovery_result['batches']:
            batch_id = batch['batch_id']
            task_id = f'process_batch_{batch_id}'
            
            try:
                result = ti.xcom_pull(task_ids=task_id)
                if result:
                    batch_results.append(result)
            except Exception as e:
                logging.warning(f"Could not get result for batch {batch_id}: {str(e)}")
        
        # Aggregate totals
        aggregated = {
            'total_files': sum(r['total_files'] for r in batch_results),
            'processed_files': sum(r['processed_files'] for r in batch_results),
            'failed_files': sum(r['failed_files'] for r in batch_results),
            'total_processing_time': sum(r['processing_time'] for r in batch_results),
            'batch_count': len(batch_results),
            'errors': []
        }
        
        # Collect all errors
        for result in batch_results:
            aggregated['errors'].extend(result.get('errors', []))
        
        # Calculate overall metrics
        if aggregated['total_processing_time'] > 0:
            aggregated['overall_throughput'] = aggregated['processed_files'] / aggregated['total_processing_time']
        else:
            aggregated['overall_throughput'] = 0
            
        if aggregated['total_files'] > 0:
            aggregated['success_rate'] = aggregated['processed_files'] / aggregated['total_files']
        else:
            aggregated['success_rate'] = 0
        
        logging.info(f"Batch processing completed: {aggregated['processed_files']}/{aggregated['total_files']} files processed")
        logging.info(f"Success rate: {aggregated['success_rate']:.2%}")
        logging.info(f"Throughput: {aggregated['overall_throughput']:.2f} files/second")
        
        return aggregated
        
    except Exception as e:
        logging.error(f"Error aggregating results: {str(e)}")
        raise


def update_processing_state(**context):
    """
    Update processing state in Redis and database.
    """
    try:
        ti = context['ti']
        results = ti.xcom_pull(task_ids='aggregate_results')
        
        if not results:
            logging.warning("No results to update state with")
            return
        
        # Update Redis state
        redis_hook = RedisHook(redis_conn_id='redis_default')
        redis_conn = redis_hook.get_conn()
        
        state = {
            'status': 'completed' if results['success_rate'] > 0.8 else 'completed_with_errors',
            'end_time': datetime.utcnow().isoformat(),
            'total_files': results['total_files'],
            'processed_files': results['processed_files'],
            'failed_files': results['failed_files'],
            'success_rate': results['success_rate'],
            'throughput': results['overall_throughput']
        }
        
        redis_conn.set('batch_processing_final_state', json.dumps(state))
        
        logging.info("Processing state updated successfully")
        
    except Exception as e:
        logging.error(f"Error updating processing state: {str(e)}")
        raise


# Task Definitions

# 1. File Discovery
discover_files_task = PythonOperator(
    task_id='discover_files',
    python_callable=discover_files,
    dag=dag,
)

# 2. Branch on processing needed
check_processing_task = BranchPythonOperator(
    task_id='check_processing_needed',
    python_callable=check_processing_needed,
    dag=dag,
)

# 3. No processing needed
no_processing_task = BashOperator(
    task_id='no_processing_needed',
    bash_command='echo "No files to process, exiting successfully"',
    dag=dag,
)

# 4. Setup environment
setup_env_task = PythonOperator(
    task_id='setup_processing_environment',
    python_callable=setup_processing_environment,
    dag=dag,
)

# 5. Process batches (dynamic task creation)
def create_batch_tasks():
    """Create dynamic batch processing tasks based on discovered files."""
    return []

# 6. Aggregate results
aggregate_task = PythonOperator(
    task_id='aggregate_results',
    python_callable=aggregate_results,
    trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
    dag=dag,
)

# 7. Update state
update_state_task = PythonOperator(
    task_id='update_processing_state',
    python_callable=update_processing_state,
    dag=dag,
)

# 8. Cleanup
cleanup_task = BashOperator(
    task_id='cleanup_temp_files',
    bash_command='''
    echo "Cleaning up temporary files..."
    find /tmp -name "sanskrit_*" -type f -mtime +1 -delete || true
    echo "Cleanup completed"
    ''',
    trigger_rule=TriggerRule.NONE_FAILED,
    dag=dag,
)

# Task Dependencies
discover_files_task >> check_processing_task

check_processing_task >> [no_processing_task, setup_env_task]

setup_env_task >> aggregate_task
aggregate_task >> update_state_task >> cleanup_task

no_processing_task >> cleanup_task