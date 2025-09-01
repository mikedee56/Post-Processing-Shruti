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
    
    # Import error handling utilities
    from utils.exception_hierarchy import create_error_handler, handle_errors
    
    # Create error handler for CLI operations
    error_handler = create_error_handler(logger, "CLI_ProcessSingle")
    
    # Generate correlation ID for this processing session
    import uuid
    correlation_id = str(uuid.uuid4())
    error_handler.set_correlation_id(correlation_id)
    
    logger.info("Starting single file processing", 
                input_file=input_file, output_file=output_file,
                correlation_id=correlation_id)
    
    try:
        # Load configuration with error handling
        try:
            config_path = Path(config) if config else ctx.obj.get('config_path')
            logger.info("Configuration loaded", config_path=str(config_path) if config_path else "default")
        except Exception as e:
            validation_error = error_handler.handle_validation_error(
                "configuration_loading", e, 
                {"config_path": config, "context_config": ctx.obj.get('config_path')}
            )
            click.echo(f"ERROR: Configuration error: {str(validation_error)}", err=True)
            logger.error("Configuration loading failed", error=str(e), correlation_id=correlation_id)
            sys.exit(1)
        
        # Initialize processor with error handling
        try:
            processor = SanskritPostProcessor(config_path)
            logger.info("Processor initialized successfully", correlation_id=correlation_id)
        except Exception as e:
            processing_error = error_handler.handle_processing_error(
                "processor_initialization", e,
                {"config_path": str(config_path) if config_path else None}
            )
            click.echo(f"ERROR: Processor initialization failed: {str(processing_error)}", err=True)
            logger.error("Processor initialization failed", error=str(e), correlation_id=correlation_id)
            sys.exit(1)
        
        # Process the file with comprehensive error handling
        try:
            logger.info("Starting file processing", input_file=input_file, correlation_id=correlation_id)
            results = processor.process_srt_file(Path(input_file), Path(output_file))
            logger.info("File processing completed successfully", correlation_id=correlation_id)
        except Exception as e:
            processing_error = error_handler.handle_processing_error(
                "srt_file_processing", e,
                {"input_file": input_file, "output_file": output_file}
            )
            click.echo(f"ERROR: File processing failed: {str(processing_error)}", err=True)
            logger.error("SRT processing failed", error=str(e), 
                        input_file=input_file, output_file=output_file,
                        correlation_id=correlation_id)
            sys.exit(1)
        
        # Display results with validation
        try:
            if hasattr(results, 'segments_modified'):
                click.echo(f"SUCCESS: Processing completed successfully!")
                click.echo(f"Corrections made: {results.segments_modified}")
                click.echo(f"Segments flagged: {getattr(results, 'flagged_segments', 0)}")
                click.echo(f"Average confidence: {results.average_confidence:.2f}")
                
                logger.info("Single file processing completed", 
                           results=vars(results), correlation_id=correlation_id)
            else:
                logger.warning("Results object missing expected attributes", 
                             available_attrs=dir(results), correlation_id=correlation_id)
                click.echo("SUCCESS: Processing completed (limited result information)")
                
        except Exception as e:
            # Non-critical error in results display
            logger.warning("Error displaying results", error=str(e), correlation_id=correlation_id)
            click.echo("SUCCESS: Processing completed (error displaying detailed results)")
        
    except Exception as e:
        # Catch-all for any unexpected errors
        critical_error = error_handler.handle_processing_error(
            "unexpected_cli_error", e,
            {"operation": "process_single", "input_file": input_file}
        )
        click.echo(f"ERROR: Unexpected error during processing: {str(critical_error)}", err=True)
        logger.error("Unexpected CLI error", error=str(e), 
                    operation="process_single", correlation_id=correlation_id)
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
    
    # Import error handling utilities
    from utils.exception_hierarchy import create_error_handler
    
    # Create error handler for batch CLI operations
    error_handler = create_error_handler(logger, "CLI_ProcessBatch")
    
    # Generate correlation ID for this batch processing session
    import uuid
    correlation_id = str(uuid.uuid4())
    error_handler.set_correlation_id(correlation_id)
    
    logger.info("Starting batch processing", 
                input_dir=input_dir, output_dir=output_dir, 
                batch_size=batch_size, dry_run=dry_run,
                correlation_id=correlation_id)
    
    try:
        # Initialize components with error handling
        config_path = ctx.obj.get('config_path')
        
        # Initialize file ingestion system
        try:
            ingestion_system = FileIngestionSystem()
            logger.info("File ingestion system initialized", correlation_id=correlation_id)
        except Exception as e:
            processing_error = error_handler.handle_processing_error(
                "ingestion_system_initialization", e,
                {"config_path": str(config_path) if config_path else None}
            )
            click.echo(f"ERROR: Failed to initialize file ingestion system: {str(processing_error)}", err=True)
            logger.error("File ingestion system initialization failed", error=str(e), correlation_id=correlation_id)
            sys.exit(1)
        
        # Initialize processor
        try:
            processor = SanskritPostProcessor(config_path)
            logger.info("Processor initialized for batch processing", correlation_id=correlation_id)
        except Exception as e:
            processing_error = error_handler.handle_processing_error(
                "batch_processor_initialization", e,
                {"config_path": str(config_path) if config_path else None}
            )
            click.echo(f"ERROR: Failed to initialize processor: {str(processing_error)}", err=True)
            logger.error("Batch processor initialization failed", error=str(e), correlation_id=correlation_id)
            sys.exit(1)
        
        # Initialize output manager
        try:
            output_manager = OutputStorageManager()
            logger.info("Output storage manager initialized", correlation_id=correlation_id)
        except Exception as e:
            processing_error = error_handler.handle_processing_error(
                "output_manager_initialization", e,
                {"output_dir": output_dir}
            )
            click.echo(f"ERROR: Failed to initialize output manager: {str(processing_error)}", err=True)
            logger.error("Output manager initialization failed", error=str(e), correlation_id=correlation_id)
            sys.exit(1)
        
        # Discover files with error handling
        try:
            input_path = Path(input_dir)
            files_to_process = list(input_path.glob('**/*.srt'))
            logger.info("File discovery completed", 
                       total_files=len(files_to_process), correlation_id=correlation_id)
        except Exception as e:
            validation_error = error_handler.handle_validation_error(
                "file_discovery", e,
                {"input_dir": input_dir}
            )
            click.echo(f"ERROR: Failed to discover files: {str(validation_error)}", err=True)
            logger.error("File discovery failed", error=str(e), 
                        input_dir=input_dir, correlation_id=correlation_id)
            sys.exit(1)
        
        if not files_to_process:
            click.echo(f"⚠️  No SRT files found in {input_dir}")
            logger.warning("No files found for processing", 
                          input_dir=input_dir, correlation_id=correlation_id)
            return
        
        click.echo(f"📁 Found {len(files_to_process)} SRT files to process")
        
        if dry_run:
            click.echo("🔍 Dry run mode - files that would be processed:")
            for file_path in files_to_process:
                click.echo(f"  - {file_path}")
            logger.info("Dry run completed", 
                       files_count=len(files_to_process), correlation_id=correlation_id)
            return
        
        # Process files in batches with comprehensive error handling
        output_path = Path(output_dir)
        total_corrections = 0
        total_flagged = 0
        failed_files = []
        
        try:
            with tqdm(total=len(files_to_process), desc="Processing files") as pbar:
                for i in range(0, len(files_to_process), batch_size):
                    batch = files_to_process[i:i + batch_size]
                    batch_id = f"batch_{i//batch_size + 1}"
                    
                    logger.info("Starting batch processing", 
                               batch_id=batch_id, batch_size=len(batch), 
                               correlation_id=correlation_id)
                    
                    for file_path in batch:
                        try:
                            # Generate output path
                            relative_path = file_path.relative_to(input_path)
                            output_file = output_path / relative_path
                            output_file.parent.mkdir(parents=True, exist_ok=True)
                            
                            # Process file with individual error handling
                            logger.debug("Processing individual file", 
                                        file_path=str(file_path), correlation_id=correlation_id)
                            
                            results = processor.process_srt_file(file_path, output_file)
                            
                            # Update counters based on results structure
                            if hasattr(results, 'segments_modified'):
                                total_corrections += results.segments_modified
                            elif isinstance(results, dict):
                                total_corrections += results.get('corrections_made', 0)
                                total_flagged += results.get('flagged_segments', 0)
                            
                            pbar.set_postfix({
                                'corrections': total_corrections,
                                'flagged': total_flagged,
                                'failed': len(failed_files)
                            })
                            
                            logger.debug("File processed successfully", 
                                        file_path=str(file_path), correlation_id=correlation_id)
                            
                        except Exception as e:
                            processing_error = error_handler.handle_processing_error(
                                "individual_file_processing", e,
                                {"file_path": str(file_path), "batch_id": batch_id}
                            )
                            
                            failed_files.append(str(file_path))
                            logger.error("Failed to process individual file", 
                                       file_path=str(file_path), error=str(e),
                                       batch_id=batch_id, correlation_id=correlation_id)
                            click.echo(f"ERROR: Failed to process {file_path}: {str(processing_error)}")
                        
                        pbar.update(1)
                    
                    logger.info("Batch processing completed", 
                               batch_id=batch_id, correlation_id=correlation_id)
        
        except Exception as e:
            processing_error = error_handler.handle_processing_error(
                "batch_processing_loop", e,
                {"total_files": len(files_to_process), "batch_size": batch_size}
            )
            click.echo(f"ERROR: Critical error in batch processing: {str(processing_error)}", err=True)
            logger.error("Critical batch processing error", error=str(e), correlation_id=correlation_id)
            sys.exit(1)
        
        # Display comprehensive summary
        successful_files = len(files_to_process) - len(failed_files)
        
        click.echo(f"\n🎉 BATCH PROCESSING COMPLETED!")
        click.echo(f"📊 Files processed successfully: {successful_files}/{len(files_to_process)}")
        if failed_files:
            click.echo(f"❌ Files failed: {len(failed_files)}")
            click.echo(f"📝 Total corrections made: {total_corrections}")
            click.echo(f"🚩 Total segments flagged: {total_flagged}")
        else:
            click.echo(f"✅ All files processed successfully!")
            click.echo(f"📝 Total corrections made: {total_corrections}")
            click.echo(f"🚩 Total segments flagged: {total_flagged}")
        
        logger.info("Batch processing summary", 
                   total_files=len(files_to_process),
                   successful_files=successful_files,
                   failed_files=len(failed_files),
                   total_corrections=total_corrections,
                   total_flagged=total_flagged,
                   correlation_id=correlation_id)
        
        # Exit with error code if any files failed
        if failed_files:
            logger.warning("Batch processing completed with failures", 
                          failed_count=len(failed_files), correlation_id=correlation_id)
            sys.exit(1)
        
    except Exception as e:
        # Catch-all for any unexpected batch processing errors
        critical_error = error_handler.handle_processing_error(
            "unexpected_batch_error", e,
            {"operation": "process_batch", "input_dir": input_dir, "output_dir": output_dir}
        )
        click.echo(f"ERROR: Unexpected error during batch processing: {str(critical_error)}", err=True)
        logger.error("Unexpected batch processing error", error=str(e), 
                    operation="process_batch", correlation_id=correlation_id)
        sys.exit(1)


@cli.command()
@click.pass_context
def stats(ctx):
    """Show statistics about lexicons and processor configuration."""
    
    # Import error handling utilities
    from utils.exception_hierarchy import create_error_handler
    
    # Create error handler for stats CLI operations
    error_handler = create_error_handler(logger, "CLI_Stats")
    
    # Generate correlation ID for this stats operation
    import uuid
    correlation_id = str(uuid.uuid4())
    error_handler.set_correlation_id(correlation_id)
    
    logger.info("Starting statistics display", correlation_id=correlation_id)
    
    try:
        # Load configuration with error handling
        config_path = ctx.obj.get('config_path')
        
        # Initialize processor for statistics
        try:
            processor = SanskritPostProcessor(config_path)
            logger.info("Processor initialized for statistics", correlation_id=correlation_id)
        except Exception as e:
            processing_error = error_handler.handle_processing_error(
                "stats_processor_initialization", e,
                {"config_path": str(config_path) if config_path else None}
            )
            click.echo(f"ERROR: Failed to initialize processor for statistics: {str(processing_error)}", err=True)
            logger.error("Stats processor initialization failed", error=str(e), correlation_id=correlation_id)
            sys.exit(1)
        
        # Get processing statistics with error handling
        try:
            stats_data = processor.get_processing_stats()
            logger.info("Processing statistics retrieved", correlation_id=correlation_id)
        except Exception as e:
            processing_error = error_handler.handle_processing_error(
                "stats_data_retrieval", e,
                {"processor_type": type(processor).__name__}
            )
            click.echo(f"ERROR: Failed to retrieve processing statistics: {str(processing_error)}", err=True)
            logger.error("Stats data retrieval failed", error=str(e), correlation_id=correlation_id)
            sys.exit(1)
        
        # Display statistics with comprehensive error handling
        try:
            click.echo("📊 ASR Post-Processing Statistics")
            click.echo("=" * 40)
            
            # Lexicon statistics with validation
            if 'lexicons' in stats_data and isinstance(stats_data['lexicons'], dict):
                click.echo("\n📚 Lexicon Statistics:")
                for lexicon_name, count in stats_data['lexicons'].items():
                    display_name = lexicon_name.title() if isinstance(lexicon_name, str) else str(lexicon_name)
                    display_count = count if isinstance(count, (int, float)) else "N/A"
                    click.echo(f"  {display_name}: {display_count} entries")
                    
                logger.debug("Lexicon statistics displayed", 
                            lexicon_count=len(stats_data['lexicons']), correlation_id=correlation_id)
            else:
                click.echo("\n📚 Lexicon Statistics: Not available")
                logger.warning("Lexicon statistics not available in expected format", 
                              stats_keys=list(stats_data.keys()) if isinstance(stats_data, dict) else "non-dict",
                              correlation_id=correlation_id)
            
            # Configuration statistics with validation
            click.echo(f"\n⚙️  Configuration:")
            
            # Fuzzy threshold
            fuzzy_threshold = stats_data.get('fuzzy_threshold', 'N/A')
            click.echo(f"  Fuzzy matching threshold: {fuzzy_threshold}")
            
            # Confidence threshold from config
            config_data = stats_data.get('config', {})
            if isinstance(config_data, dict):
                confidence_threshold = config_data.get('confidence_threshold', 'N/A')
                click.echo(f"  Confidence threshold: {confidence_threshold}")
                
                # Additional configuration details if available
                if 'enable_ner' in config_data:
                    click.echo(f"  NER enabled: {config_data['enable_ner']}")
                if 'enable_mcp_processing' in config_data:
                    click.echo(f"  MCP processing enabled: {config_data.get('enable_mcp_processing', 'N/A')}")
            else:
                click.echo(f"  Confidence threshold: N/A")
                logger.warning("Configuration data not available in expected format", 
                              config_type=type(config_data), correlation_id=correlation_id)
            
            # Processing capabilities
            if hasattr(processor, 'enable_ner'):
                click.echo(f"\n🧠 Processing Capabilities:")
                click.echo(f"  NER Processing: {'Enabled' if processor.enable_ner else 'Disabled'}")
            
            if hasattr(processor, 'session_correlation_id'):
                click.echo(f"  Session Correlation ID: {processor.session_correlation_id}")
                
            logger.info("Statistics display completed successfully", correlation_id=correlation_id)
            
        except Exception as e:
            processing_error = error_handler.handle_processing_error(
                "stats_display_formatting", e,
                {"stats_data_keys": list(stats_data.keys()) if isinstance(stats_data, dict) else "non-dict"}
            )
            click.echo(f"ERROR: Error formatting statistics display: {str(processing_error)}", err=True)
            logger.error("Statistics display formatting failed", error=str(e), correlation_id=correlation_id)
            
            # Fallback: display raw statistics
            click.echo("\n📊 Raw Statistics Data:")
            try:
                click.echo(str(stats_data))
            except:
                click.echo("Unable to display statistics data")
            
            sys.exit(1)
        
        logger.info("Statistics operation completed successfully", 
                   stats_sections=list(stats_data.keys()) if isinstance(stats_data, dict) else 0,
                   correlation_id=correlation_id)
        
    except Exception as e:
        # Catch-all for any unexpected statistics errors
        critical_error = error_handler.handle_processing_error(
            "unexpected_stats_error", e,
            {"operation": "stats"}
        )
        click.echo(f"ERROR: Unexpected error retrieving statistics: {str(critical_error)}", err=True)
        logger.error("Unexpected statistics error", error=str(e), 
                    operation="stats", correlation_id=correlation_id)
        sys.exit(1)


@cli.command()
@click.argument('directory', type=click.Path(exists=True))
def validate(directory: str):
    """Validate SRT files in a directory for format and encoding issues."""
    
    # Import error handling utilities
    from utils.exception_hierarchy import create_error_handler
    
    # Create error handler for validation CLI operations
    error_handler = create_error_handler(logger, "CLI_Validate")
    
    # Generate correlation ID for this validation operation
    import uuid
    correlation_id = str(uuid.uuid4())
    error_handler.set_correlation_id(correlation_id)
    
    logger.info("Starting SRT file validation", directory=directory, correlation_id=correlation_id)
    
    try:
        # Initialize file ingestion system with error handling
        try:
            ingestion_system = FileIngestionSystem()
            logger.info("File ingestion system initialized for validation", correlation_id=correlation_id)
        except Exception as e:
            processing_error = error_handler.handle_processing_error(
                "validation_ingestion_system_init", e,
                {"directory": directory}
            )
            click.echo(f"ERROR: Failed to initialize file ingestion system: {str(processing_error)}", err=True)
            logger.error("Validation ingestion system initialization failed", error=str(e), correlation_id=correlation_id)
            sys.exit(1)
        
        # Discover SRT files with error handling
        try:
            dir_path = Path(directory)
            srt_files = list(dir_path.glob('**/*.srt'))
            logger.info("SRT file discovery completed", 
                       total_files=len(srt_files), directory=directory, correlation_id=correlation_id)
        except Exception as e:
            validation_error = error_handler.handle_validation_error(
                "srt_file_discovery", e,
                {"directory": directory}
            )
            click.echo(f"ERROR: Failed to discover SRT files: {str(validation_error)}", err=True)
            logger.error("SRT file discovery failed", error=str(e), 
                        directory=directory, correlation_id=correlation_id)
            sys.exit(1)
        
        if not srt_files:
            click.echo(f"⚠️  No SRT files found in {directory}")
            logger.warning("No SRT files found for validation", 
                          directory=directory, correlation_id=correlation_id)
            return
        
        click.echo(f"🔍 Validating {len(srt_files)} SRT files...")
        
        # Initialize validation counters
        valid_files = 0
        invalid_files = 0
        validation_errors = []
        
        # Process files with comprehensive error handling
        try:
            with tqdm(total=len(srt_files), desc="Validating files") as pbar:
                for file_path in srt_files:
                    try:
                        # Basic SRT file validation
                        logger.debug("Validating individual file", 
                                    file_path=str(file_path), correlation_id=correlation_id)
                        
                        # Read file with encoding error handling
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                        except UnicodeDecodeError:
                            # Try alternative encodings
                            try:
                                with open(file_path, 'r', encoding='latin-1') as f:
                                    content = f.read()
                                logger.warning("File read with latin-1 encoding", 
                                              file_path=str(file_path), correlation_id=correlation_id)
                            except Exception as encoding_error:
                                validation_error = error_handler.handle_validation_error(
                                    "file_encoding_validation", encoding_error,
                                    {"file_path": str(file_path), "encoding_tried": "utf-8,latin-1"}
                                )
                                validation_errors.append(f"{file_path}: Encoding error - {str(validation_error)}")
                                invalid_files += 1
                                pbar.set_postfix({'valid': valid_files, 'invalid': invalid_files})
                                logger.error("File encoding validation failed", 
                                           file_path=str(file_path), error=str(encoding_error), 
                                           correlation_id=correlation_id)
                                continue
                        
                        # Validate SRT format structure
                        is_valid_format = False
                        has_timestamps = False
                        has_content = False
                        
                        # Check for basic SRT structure
                        if '-->' in content:
                            has_timestamps = True
                        if any(char.isdigit() for char in content):
                            has_content = True
                        
                        # Enhanced validation: check for sequence numbers
                        lines = content.strip().split('\n')
                        if lines and lines[0].strip().isdigit():
                            is_valid_format = True
                        
                        # Validate overall format
                        if has_timestamps and has_content and is_valid_format:
                            valid_files += 1
                            logger.debug("File validation passed", 
                                        file_path=str(file_path), correlation_id=correlation_id)
                        else:
                            invalid_files += 1
                            validation_issues = []
                            if not has_timestamps:
                                validation_issues.append("missing timestamps")
                            if not has_content:
                                validation_issues.append("missing content")
                            if not is_valid_format:
                                validation_issues.append("invalid format structure")
                            
                            issue_description = ", ".join(validation_issues)
                            validation_errors.append(f"{file_path}: Format issues - {issue_description}")
                            
                            click.echo(f"⚠️  Invalid format: {file_path} ({issue_description})")
                            logger.warning("File validation failed", 
                                          file_path=str(file_path), issues=validation_issues,
                                          correlation_id=correlation_id)
                        
                        pbar.set_postfix({'valid': valid_files, 'invalid': invalid_files})
                        
                    except Exception as e:
                        # Individual file validation error
                        processing_error = error_handler.handle_processing_error(
                            "individual_file_validation", e,
                            {"file_path": str(file_path)}
                        )
                        invalid_files += 1
                        validation_errors.append(f"{file_path}: Processing error - {str(processing_error)}")
                        
                        click.echo(f"ERROR: Error validating {file_path}: {str(processing_error)}")
                        logger.error("Individual file validation error", 
                                   file_path=str(file_path), error=str(e), correlation_id=correlation_id)
                    
                    pbar.update(1)
                    
        except Exception as e:
            # Critical error in validation loop
            processing_error = error_handler.handle_processing_error(
                "validation_processing_loop", e,
                {"total_files": len(srt_files), "directory": directory}
            )
            click.echo(f"ERROR: Critical error during validation: {str(processing_error)}", err=True)
            logger.error("Critical validation loop error", error=str(e), correlation_id=correlation_id)
            sys.exit(1)
        
        # Display comprehensive validation summary
        total_files = len(srt_files)
        success_rate = (valid_files / total_files * 100) if total_files > 0 else 0
        
        click.echo(f"\n🎉 VALIDATION COMPLETED!")
        click.echo(f"📊 Validation Summary:")
        click.echo(f"  📈 Total files processed: {total_files}")
        click.echo(f"  ✅ Valid files: {valid_files}")
        click.echo(f"  ❌ Invalid files: {invalid_files}")
        click.echo(f"  📈 Success rate: {success_rate:.1f}%")
        
        # Display validation errors if any
        if validation_errors and len(validation_errors) <= 10:
            click.echo(f"\n📝 Validation Issues:")
            for error_msg in validation_errors:
                click.echo(f"  - {error_msg}")
        elif len(validation_errors) > 10:
            click.echo(f"\n📝 Validation Issues (showing first 10 of {len(validation_errors)}):")
            for error_msg in validation_errors[:10]:
                click.echo(f"  - {error_msg}")
            click.echo(f"  ... and {len(validation_errors) - 10} more issues")
        
        logger.info("Validation operation completed", 
                   total_files=total_files,
                   valid_files=valid_files, 
                   invalid_files=invalid_files,
                   success_rate=success_rate,
                   validation_errors_count=len(validation_errors),
                   correlation_id=correlation_id)
        
        # Exit with appropriate code
        if invalid_files > 0:
            logger.warning("Validation completed with invalid files detected", 
                          invalid_count=invalid_files, correlation_id=correlation_id)
            sys.exit(1)
        else:
            logger.info("All files validated successfully", correlation_id=correlation_id)
        
    except Exception as e:
        # Catch-all for any unexpected validation errors
        critical_error = error_handler.handle_processing_error(
            "unexpected_validation_error", e,
            {"operation": "validate", "directory": directory}
        )
        click.echo(f"ERROR: Unexpected error during validation: {str(critical_error)}", err=True)
        logger.error("Unexpected validation error", error=str(e), 
                    operation="validate", directory=directory, correlation_id=correlation_id)
        sys.exit(1)


if __name__ == '__main__':
    cli()