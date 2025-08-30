"""
Workflow Integration Manager for Story 3.6
Integrates semantic quality assurance with existing review workflows
"""

from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path

from post_processors.academic_polish_processor import AcademicPolishProcessor
from qa_module.academic_workflow_integrator import AcademicWorkflowIntegrator, AcademicStakeholderReport
from qa_module.academic_reporting_dashboard import AcademicReportingDashboard, AcademicReportingConfig


class WorkflowStage(Enum):
    """Workflow processing stages"""
    PRE_PROCESSING = "pre_processing"
    ACADEMIC_POLISH = "academic_polish" 
    QUALITY_VALIDATION = "quality_validation"
    EXPERT_REVIEW = "expert_review"
    POST_PROCESSING = "post_processing"
    FINAL_VALIDATION = "final_validation"


@dataclass
class WorkflowHook:
    """Configuration for workflow integration hooks"""
    stage: WorkflowStage
    callback: Callable
    priority: int = 100
    enabled: bool = True


@dataclass
class ReviewWorkflowConfig:
    """Configuration for review workflow integration"""
    enable_semantic_enhancement: bool = True
    enable_quality_gates: bool = True
    enable_expert_review_queue: bool = True
    enable_automated_reporting: bool = True
    quality_threshold_for_expert_review: float = 0.8
    critical_issues_threshold: int = 5
    output_directory: str = "output/academic"
    backup_original: bool = True
    generate_comparison_reports: bool = True


class WorkflowIntegrationManager:
    """
    Manages integration of semantic quality assurance with existing review workflows
    Provides hooks and callbacks for seamless integration
    """
    
    def __init__(self, config: Optional[ReviewWorkflowConfig] = None):
        """Initialize the workflow integration manager"""
        self.config = config or ReviewWorkflowConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.academic_integrator = AcademicWorkflowIntegrator()
        
        reporting_config = AcademicReportingConfig(
            output_directory=f"{self.config.output_directory}/reports"
        )
        self.reporting_dashboard = AcademicReportingDashboard(reporting_config)
        
        # Workflow hooks registry
        self.workflow_hooks: Dict[WorkflowStage, List[WorkflowHook]] = {
            stage: [] for stage in WorkflowStage
        }
        
        # Processing statistics
        self.processing_stats = {
            'total_processed': 0,
            'quality_enhanced': 0,
            'expert_review_required': 0,
            'automatic_approval': 0
        }
        
        # Ensure output directory exists
        Path(self.config.output_directory).mkdir(parents=True, exist_ok=True)
        
        self.logger.info("Workflow Integration Manager initialized")
    
    def register_workflow_hook(self, hook: WorkflowHook):
        """Register a workflow hook for specific processing stage"""
        self.workflow_hooks[hook.stage].append(hook)
        self.workflow_hooks[hook.stage].sort(key=lambda h: h.priority)
        self.logger.info(f"Registered workflow hook for {hook.stage.value} stage")
    
    def process_with_integrated_workflow(self, 
                                       content: str,
                                       filename: str,
                                       original_processor_func: Optional[Callable] = None,
                                       context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process content with integrated semantic quality workflow
        
        Args:
            content: Content to process
            filename: Original filename
            original_processor_func: Original processing function (for backward compatibility)
            context: Additional processing context
            
        Returns:
            Processing results with workflow metadata
        """
        
        context = context or {}
        workflow_results = {
            'filename': filename,
            'original_content_length': len(content),
            'stages_completed': [],
            'quality_report': None,
            'workflow_metadata': {},
            'processing_successful': False
        }
        
        try:
            self.logger.info(f"Starting integrated workflow processing for {filename}")
            
            # Stage 1: Pre-processing hooks
            content = self._execute_workflow_stage(
                WorkflowStage.PRE_PROCESSING, content, filename, context, workflow_results
            )
            
            # Stage 2: Academic Polish (with existing processor integration)
            if self.config.enable_semantic_enhancement:
                processed_content, academic_report = self._execute_academic_polish_stage(
                    content, filename, original_processor_func, context, workflow_results
                )
                content = processed_content
                workflow_results['quality_report'] = academic_report
            else:
                # Fallback to original processor if available
                if original_processor_func:
                    content = original_processor_func(content)
                    self.logger.info("Used original processor function (semantic enhancement disabled)")
            
            # Stage 3: Quality Validation
            if self.config.enable_quality_gates:
                content = self._execute_quality_validation_stage(
                    content, filename, context, workflow_results
                )
            
            # Stage 4: Expert Review Decision
            expert_review_needed = self._determine_expert_review_requirement(
                workflow_results.get('quality_report'), workflow_results
            )
            
            if expert_review_needed and self.config.enable_expert_review_queue:
                workflow_results = self._queue_for_expert_review(
                    content, filename, workflow_results
                )
            
            # Stage 5: Post-processing hooks
            content = self._execute_workflow_stage(
                WorkflowStage.POST_PROCESSING, content, filename, context, workflow_results
            )
            
            # Stage 6: Final validation and output
            final_results = self._execute_final_validation_stage(
                content, filename, context, workflow_results
            )
            
            workflow_results.update(final_results)
            workflow_results['processed_content'] = content
            workflow_results['processing_successful'] = True
            
            # Update statistics
            self._update_processing_statistics(workflow_results)
            
            self.logger.info(f"Integrated workflow processing completed successfully for {filename}")
            
        except Exception as e:
            self.logger.error(f"Workflow processing failed for {filename}: {str(e)}")
            workflow_results['error'] = str(e)
            workflow_results['processing_successful'] = False
        
        return workflow_results
    
    def _execute_workflow_stage(self, 
                               stage: WorkflowStage, 
                               content: str,
                               filename: str,
                               context: Dict[str, Any],
                               workflow_results: Dict[str, Any]) -> str:
        """Execute hooks for a specific workflow stage"""
        
        stage_hooks = [hook for hook in self.workflow_hooks[stage] if hook.enabled]
        
        if not stage_hooks:
            workflow_results['stages_completed'].append(stage.value)
            return content
        
        self.logger.debug(f"Executing {len(stage_hooks)} hooks for {stage.value} stage")
        
        for hook in stage_hooks:
            try:
                # Execute hook callback
                hook_result = hook.callback(content, filename, context)
                
                # Update content if hook returns modified content
                if isinstance(hook_result, str):
                    content = hook_result
                elif isinstance(hook_result, dict) and 'content' in hook_result:
                    content = hook_result['content']
                    # Store additional hook metadata
                    if 'metadata' in hook_result:
                        workflow_results['workflow_metadata'][f"{stage.value}_hook_{hook.priority}"] = hook_result['metadata']
                
            except Exception as e:
                self.logger.error(f"Hook execution failed in {stage.value} stage: {str(e)}")
                # Continue with other hooks - don't fail entire workflow for one hook
        
        workflow_results['stages_completed'].append(stage.value)
        return content
    
    def _execute_academic_polish_stage(self,
                                     content: str,
                                     filename: str,
                                     original_processor_func: Optional[Callable],
                                     context: Dict[str, Any],
                                     workflow_results: Dict[str, Any]) -> Tuple[str, AcademicStakeholderReport]:
        """Execute academic polish stage with semantic enhancement"""
        
        self.logger.debug("Executing academic polish stage with semantic enhancement")
        
        # Add filename to context
        processing_context = {**context, 'filename': filename}
        
        # Execute enhanced academic processing
        processed_content, stakeholder_report = self.academic_integrator.process_with_enhanced_quality(
            content, processing_context
        )
        
        # Generate reports if enabled
        if self.config.enable_automated_reporting:
            report_files = self.reporting_dashboard.generate_comprehensive_report(
                stakeholder_report, filename.replace('.srt', ''), processing_context
            )
            workflow_results['report_files'] = report_files
            self.logger.info(f"Generated {len(report_files)} report files")
        
        workflow_results['stages_completed'].append(WorkflowStage.ACADEMIC_POLISH.value)
        return processed_content, stakeholder_report
    
    def _execute_quality_validation_stage(self,
                                        content: str,
                                        filename: str,
                                        context: Dict[str, Any],
                                        workflow_results: Dict[str, Any]) -> str:
        """Execute quality validation stage"""
        
        self.logger.debug("Executing quality validation stage")
        
        # Execute quality validation hooks
        content = self._execute_workflow_stage(
            WorkflowStage.QUALITY_VALIDATION, content, filename, context, workflow_results
        )
        
        return content
    
    def _determine_expert_review_requirement(self,
                                           quality_report: Optional[AcademicStakeholderReport],
                                           workflow_results: Dict[str, Any]) -> bool:
        """Determine if expert review is required based on quality metrics"""
        
        if not quality_report:
            return False
        
        metrics = quality_report.quality_metrics
        
        # Check quality thresholds
        if metrics.overall_quality_score < self.config.quality_threshold_for_expert_review:
            self.logger.info("Expert review required: Overall quality below threshold")
            return True
        
        # Check critical issues
        if metrics.critical_issues_count >= self.config.critical_issues_threshold:
            self.logger.info("Expert review required: Too many critical issues")
            return True
        
        # Check specific compliance areas
        if metrics.iast_compliance_score < 0.8 or metrics.sanskrit_accuracy_score < 0.8:
            self.logger.info("Expert review required: Low compliance scores")
            return True
        
        return False
    
    def _queue_for_expert_review(self,
                               content: str,
                               filename: str,
                               workflow_results: Dict[str, Any]) -> Dict[str, Any]:
        """Queue content for expert review"""
        
        self.logger.info(f"Queueing {filename} for expert review")
        
        # Create expert review package
        review_package = {
            'filename': filename,
            'content': content,
            'quality_report': workflow_results.get('quality_report'),
            'workflow_metadata': workflow_results.get('workflow_metadata', {}),
            'timestamp': workflow_results.get('quality_report', {}).quality_metrics.processing_timestamp
        }
        
        # Save to expert review queue directory
        review_queue_dir = Path(self.config.output_directory) / "expert_review_queue"
        review_queue_dir.mkdir(parents=True, exist_ok=True)
        
        review_filename = f"{filename.replace('.srt', '')}_expert_review.json"
        review_file_path = review_queue_dir / review_filename
        
        import json
        with open(review_file_path, 'w', encoding='utf-8') as f:
            json.dump(review_package, f, indent=2, default=str)
        
        workflow_results['expert_review_queued'] = True
        workflow_results['expert_review_file'] = str(review_file_path)
        workflow_results['stages_completed'].append(WorkflowStage.EXPERT_REVIEW.value)
        
        return workflow_results
    
    def _execute_final_validation_stage(self,
                                      content: str,
                                      filename: str,
                                      context: Dict[str, Any],
                                      workflow_results: Dict[str, Any]) -> Dict[str, Any]:
        """Execute final validation stage"""
        
        self.logger.debug("Executing final validation stage")
        
        # Execute final validation hooks
        content = self._execute_workflow_stage(
            WorkflowStage.FINAL_VALIDATION, content, filename, context, workflow_results
        )
        
        # Save final output
        output_filename = f"{filename.replace('.srt', '')}_processed.srt"
        output_path = Path(self.config.output_directory) / output_filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        # Backup original if requested
        if self.config.backup_original:
            backup_filename = f"{filename.replace('.srt', '')}_original.srt"
            backup_path = Path(self.config.output_directory) / "backups" / backup_filename
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Note: We don't have the original content here, this would need to be handled earlier
            self.logger.info("Original backup would be created here (needs original content)")
        
        return {
            'output_file': str(output_path),
            'final_content_length': len(content),
            'processing_completed': True
        }
    
    def _update_processing_statistics(self, workflow_results: Dict[str, Any]):
        """Update processing statistics"""
        
        self.processing_stats['total_processed'] += 1
        
        if workflow_results.get('quality_report'):
            self.processing_stats['quality_enhanced'] += 1
        
        if workflow_results.get('expert_review_queued'):
            self.processing_stats['expert_review_required'] += 1
        else:
            self.processing_stats['automatic_approval'] += 1
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get current processing statistics"""
        return self.processing_stats.copy()
    
    def integrate_with_existing_processor(self, processor_instance):
        """
        Integrate workflow manager with existing processor instance
        Replaces or wraps existing processing methods
        """
        
        # Store reference to original methods
        original_methods = {}
        
        if hasattr(processor_instance, 'process_srt_file'):
            original_methods['process_srt_file'] = processor_instance.process_srt_file
            
            def enhanced_process_srt_file(input_file: str, output_file: str, **kwargs):
                """Enhanced processing with workflow integration"""
                
                # Read original content
                with open(input_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Extract filename
                filename = Path(input_file).name
                
                # Process with integrated workflow
                workflow_results = self.process_with_integrated_workflow(
                    content, filename, original_methods['process_srt_file'], kwargs
                )
                
                # Write results
                if workflow_results.get('processing_successful'):
                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write(workflow_results['processed_content'])
                
                return workflow_results
            
            processor_instance.process_srt_file = enhanced_process_srt_file
            self.logger.info("Integrated workflow manager with existing processor")
        
        else:
            self.logger.warning("Could not integrate - process_srt_file method not found")


def create_workflow_integration_manager(config: Optional[ReviewWorkflowConfig] = None) -> WorkflowIntegrationManager:
    """Factory function to create workflow integration manager"""
    return WorkflowIntegrationManager(config)