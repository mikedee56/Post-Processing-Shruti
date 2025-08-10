"""
Main entry point for the ASR Post-Processing Workflow.

This module provides the command-line interface and orchestrates the
post-processing pipeline for Yoga Vedanta lecture transcripts.
"""

import sys
import logging
from pathlib import Path
from typing import Optional

import click
import structlog
from tqdm import tqdm

# Import our modules
from post_processors.sanskrit_post_processor import SanskritPostProcessor
from storage.file_ingestion import FileIngestionSystem
from storage.output_management import OutputStorageManager
from config.config_loader import ConfigLoader


# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()


@click.group()
@click.option('--config', '-c', type=click.Path(exists=True), help='Path to configuration file')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.pass_context
def cli(ctx, config: Optional[str], verbose: bool):
    """ASR Post-Processing Workflow for Yoga Vedanta Lectures."""
    
    # Set up logging level
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=log_level)
    
    # Initialize context
    ctx.ensure_object(dict)
    ctx.obj['config_path'] = Path(config) if config else None
    ctx.obj['verbose'] = verbose
    
    logger.info("ASR Post-Processing Workflow initialized", 
                config_path=config, verbose=verbose)


@cli.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.argument('output_file', type=click.Path())
@click.option('--config', '-c', type=click.Path(exists=True), help='Processing configuration file')
@click.pass_context
def process_single(ctx, input_file: str, output_file: str, config: Optional[str]):
    """Process a single SRT file."""
    
    logger.info("Processing single file", input_file=input_file, output_file=output_file)
    
    try:
        # Load configuration
        config_path = Path(config) if config else ctx.obj.get('config_path')
        
        # Initialize processor
        processor = SanskritPostProcessor(config_path)
        
        # Process the file
        results = processor.process_srt_file(Path(input_file), Path(output_file))
        
        # Display results
        click.echo(f"SUCCESS: Processing completed successfully!")
        click.echo(f"Corrections made: {results.segments_modified}")
        click.echo(f"Segments flagged: {results.flagged_segments}")
        click.echo(f"Average confidence: {results.average_confidence:.2f}")
        
        logger.info("Single file processing completed", results=vars(results))
        
    except Exception as e:
        click.echo(f"ERROR: Error processing file: {e}", err=True)
        logger.error("Processing failed", error=str(e), input_file=input_file)
        sys.exit(1)


@cli.command()
@click.option('--input-dir', '-i', type=click.Path(exists=True), 
              default='data/raw_srts', help='Input directory for SRT files')
@click.option('--output-dir', '-o', type=click.Path(), 
              default='data/processed_srts', help='Output directory for processed files')
@click.option('--batch-size', '-b', type=int, default=10, help='Batch size for processing')
@click.option('--dry-run', is_flag=True, help='Show what would be processed without actually processing')
@click.pass_context
def process_batch(ctx, input_dir: str, output_dir: str, batch_size: int, dry_run: bool):
    """Process multiple SRT files in batch."""
    
    logger.info("Starting batch processing", 
                input_dir=input_dir, output_dir=output_dir, 
                batch_size=batch_size, dry_run=dry_run)
    
    try:
        # Initialize components
        config_path = ctx.obj.get('config_path')
        
        ingestion_system = FileIngestionSystem()
        processor = SanskritPostProcessor(config_path)
        output_manager = OutputStorageManager()
        
        # Discover files
        input_path = Path(input_dir)
        files_to_process = list(input_path.glob('**/*.srt'))
        
        if not files_to_process:
            click.echo(f"⚠️  No SRT files found in {input_dir}")
            return
        
        click.echo(f"📁 Found {len(files_to_process)} SRT files to process")
        
        if dry_run:
            click.echo("🔍 Dry run mode - files that would be processed:")
            for file_path in files_to_process:
                click.echo(f"  - {file_path}")
            return
        
        # Process files in batches
        output_path = Path(output_dir)
        total_corrections = 0
        total_flagged = 0
        
        with tqdm(total=len(files_to_process), desc="Processing files") as pbar:
            for i in range(0, len(files_to_process), batch_size):
                batch = files_to_process[i:i + batch_size]
                
                for file_path in batch:
                    try:
                        # Generate output path
                        relative_path = file_path.relative_to(input_path)
                        output_file = output_path / relative_path
                        output_file.parent.mkdir(parents=True, exist_ok=True)
                        
                        # Process file
                        results = processor.process_srt_file(file_path, output_file)
                        
                        total_corrections += results['corrections_made']
                        total_flagged += results['flagged_segments']
                        
                        pbar.set_postfix({
                            'corrections': total_corrections,
                            'flagged': total_flagged
                        })
                        
                    except Exception as e:
                        logger.error("Failed to process file", 
                                   file_path=str(file_path), error=str(e))
                        click.echo(f"ERROR: Failed to process {file_path}: {e}")
                    
                    pbar.update(1)
        
        # Summary
        click.echo(f"\nSUCCESS: Batch processing completed!")
        click.echo(f"Total corrections made: {total_corrections}")
        click.echo(f"Total segments flagged: {total_flagged}")
        
        logger.info("Batch processing completed", 
                   total_files=len(files_to_process),
                   total_corrections=total_corrections,
                   total_flagged=total_flagged)
        
    except Exception as e:
        click.echo(f"ERROR: Error in batch processing: {e}", err=True)
        logger.error("Batch processing failed", error=str(e))
        sys.exit(1)


@cli.command()
@click.pass_context
def stats(ctx):
    """Show statistics about lexicons and processor configuration."""
    
    try:
        config_path = ctx.obj.get('config_path')
        processor = SanskritPostProcessor(config_path)
        
        stats_data = processor.get_processing_stats()
        
        click.echo("📊 ASR Post-Processing Statistics")
        click.echo("=" * 40)
        
        # Lexicon statistics
        click.echo("\n📚 Lexicon Statistics:")
        for lexicon_name, count in stats_data['lexicons'].items():
            click.echo(f"  {lexicon_name.title()}: {count} entries")
        
        # Configuration
        click.echo(f"\nConfiguration:")
        click.echo(f"  Fuzzy matching threshold: {stats_data['fuzzy_threshold']}")
        click.echo(f"  Confidence threshold: {stats_data['config'].get('confidence_threshold', 'N/A')}")
        
        logger.info("Statistics displayed", stats=stats_data)
        
    except Exception as e:
        click.echo(f"ERROR: Error retrieving statistics: {e}", err=True)
        logger.error("Failed to get statistics", error=str(e))
        sys.exit(1)


@cli.command()
@click.argument('directory', type=click.Path(exists=True))
def validate(directory: str):
    """Validate SRT files in a directory for format and encoding issues."""
    
    logger.info("Starting validation", directory=directory)
    
    try:
        ingestion_system = FileIngestionSystem()
        
        dir_path = Path(directory)
        srt_files = list(dir_path.glob('**/*.srt'))
        
        if not srt_files:
            click.echo(f"⚠️  No SRT files found in {directory}")
            return
        
        click.echo(f"🔍 Validating {len(srt_files)} SRT files...")
        
        valid_files = 0
        invalid_files = 0
        
        with tqdm(total=len(srt_files), desc="Validating files") as pbar:
            for file_path in srt_files:
                try:
                    # This would use the validation logic from FileIngestionSystem
                    # For now, basic validation
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                    if '-->' in content and any(char.isdigit() for char in content):
                        valid_files += 1
                        pbar.set_postfix({'valid': valid_files, 'invalid': invalid_files})
                    else:
                        invalid_files += 1
                        click.echo(f"⚠️  Invalid format: {file_path}")
                        
                except Exception as e:
                    invalid_files += 1
                    click.echo(f"ERROR: Error reading {file_path}: {e}")
                
                pbar.update(1)
        
        click.echo(f"\nSUCCESS: Validation completed!")
        click.echo(f"📊 Valid files: {valid_files}")
        click.echo(f"ERROR: Invalid files: {invalid_files}")
        
        logger.info("Validation completed", 
                   valid_files=valid_files, invalid_files=invalid_files)
        
    except Exception as e:
        click.echo(f"ERROR: Error during validation: {e}", err=True)
        logger.error("Validation failed", error=str(e))
        sys.exit(1)


if __name__ == '__main__':
    cli()