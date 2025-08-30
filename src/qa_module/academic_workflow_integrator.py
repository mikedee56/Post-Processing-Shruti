"""
Academic Workflow Integration for Story 3.6
Integrates semantic quality assurance with existing academic workflows
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import logging
from pathlib import Path

from post_processors.academic_polish_processor import AcademicPolishProcessor, PolishIssue
from qa_module.academic_validator import AcademicValidator, AcademicValidationResult
from qa_module.quality_gate import QualityGate, QualityReport, ComplianceScore
from utils.metrics_collector import MetricsCollector


@dataclass
class SemanticQualityMetrics:
    """Enhanced quality metrics with semantic awareness"""
    iast_compliance_score: float = 0.0
    sanskrit_accuracy_score: float = 0.0
    terminology_consistency_score: float = 0.0
    academic_formatting_score: float = 0.0
    semantic_coherence_score: float = 0.0
    polish_enhancement_count: int = 0
    critical_issues_count: int = 0
    overall_quality_score: float = 0.0
    processing_timestamp: str = ""


@dataclass
class AcademicStakeholderReport:
    """Report tailored for academic stakeholders"""
    executive_summary: str
    quality_metrics: SemanticQualityMetrics
    enhancement_details: List[Dict[str, Any]]
    compliance_findings: List[Dict[str, Any]]
    recommendations: List[str]
    iast_validation_results: Dict[str, Any]
    content_quality_trends: Dict[str, float]


class AcademicWorkflowIntegrator:
    """
    Integrates semantic quality assurance with existing academic workflows
    Provides enhanced reporting and validation for academic stakeholders
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the academic workflow integrator"""
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize existing components
        self.academic_polish_processor = AcademicPolishProcessor()
        
        # Initialize QA framework if enabled
        qa_config = self.config.get('qa_framework', {})
        self.academic_validator = AcademicValidator(config=qa_config)
        
        # Initialize quality gate
        quality_gate_config = self.config.get('quality_gate', {})
        self.quality_gate = QualityGate(config=quality_gate_config)
        
        # Initialize metrics collection
        metrics_config = self.config.get('metrics', {})
        self.metrics_collector = MetricsCollector(metrics_config)
        
        self.logger.info("Academic Workflow Integrator initialized")
    
    def process_with_enhanced_quality(self, content: str, context: Optional[Dict[str, Any]] = None) -> Tuple[str, AcademicStakeholderReport]:
        """
        Process content with enhanced academic quality assurance
        
        Args:
            content: SRT content to process
            context: Additional processing context
            
        Returns:
            Tuple of (processed_content, academic_report)
        """
        context = context or {}
        
        self.logger.info("Starting enhanced academic quality processing")
        
        # Step 1: Apply academic polish processing
        polished_content, polish_issues = self.academic_polish_processor.polish_srt_content(content)
        
        # Step 2: Apply quality gate validation
        quality_report = self.quality_gate.validate_content(polished_content, context)
        
        # Step 3: Academic validation
        academic_validation = self.academic_validator.validate_content(polished_content, context)
        
        # Step 4: Calculate semantic quality metrics
        quality_metrics = self._calculate_semantic_quality_metrics(
            polish_issues, quality_report, academic_validation
        )
        
        # Step 5: Generate academic stakeholder report
        stakeholder_report = self._generate_stakeholder_report(
            quality_metrics, polish_issues, quality_report, academic_validation
        )
        
        # Step 6: Collect metrics for trend analysis
        self._collect_processing_metrics(quality_metrics, context)
        
        self.logger.info(f"Enhanced processing completed. Overall quality score: {quality_metrics.overall_quality_score:.2f}")
        
        return polished_content, stakeholder_report
    
    def _calculate_semantic_quality_metrics(self, 
                                          polish_issues: List[PolishIssue],
                                          quality_report: QualityReport,
                                          academic_validation: AcademicValidationResult) -> SemanticQualityMetrics:
        """Calculate comprehensive semantic quality metrics"""
        
        # Extract base scores from quality report
        iast_score = quality_report.compliance_scores.get('iast_compliance', 0.0)
        sanskrit_score = quality_report.compliance_scores.get('sanskrit_accuracy', 0.0)
        terminology_score = quality_report.compliance_scores.get('terminology_consistency', 0.0)
        formatting_score = quality_report.compliance_scores.get('academic_formatting', 0.0)
        
        # Calculate semantic coherence from academic validation
        semantic_coherence = academic_validation.overall_score if academic_validation else 0.0
        
        # Count polish enhancements
        polish_enhancement_count = len(polish_issues)
        critical_issues_count = len([issue for issue in polish_issues if issue.priority == 'critical'])
        
        # Calculate overall quality score (weighted average)
        overall_score = (
            iast_score * 0.25 +
            sanskrit_score * 0.25 +
            terminology_score * 0.20 +
            formatting_score * 0.15 +
            semantic_coherence * 0.15
        )
        
        # Adjust for critical issues (penalty)
        if critical_issues_count > 0:
            penalty = min(0.1 * critical_issues_count, 0.3)  # Max 30% penalty
            overall_score = max(0.0, overall_score - penalty)
        
        import datetime
        
        return SemanticQualityMetrics(
            iast_compliance_score=iast_score,
            sanskrit_accuracy_score=sanskrit_score,
            terminology_consistency_score=terminology_score,
            academic_formatting_score=formatting_score,
            semantic_coherence_score=semantic_coherence,
            polish_enhancement_count=polish_enhancement_count,
            critical_issues_count=critical_issues_count,
            overall_quality_score=overall_score,
            processing_timestamp=datetime.datetime.now().isoformat()
        )
    
    def _generate_stakeholder_report(self,
                                   quality_metrics: SemanticQualityMetrics,
                                   polish_issues: List[PolishIssue],
                                   quality_report: QualityReport,
                                   academic_validation: AcademicValidationResult) -> AcademicStakeholderReport:
        """Generate comprehensive report for academic stakeholders"""
        
        # Executive summary
        executive_summary = self._generate_executive_summary(quality_metrics, polish_issues)
        
        # Enhancement details
        enhancement_details = self._format_enhancement_details(polish_issues)
        
        # Compliance findings
        compliance_findings = self._format_compliance_findings(quality_report)
        
        # Recommendations
        recommendations = self._generate_recommendations(quality_metrics, polish_issues)
        
        # IAST validation results
        iast_results = self._extract_iast_validation_results(quality_report)
        
        # Quality trends (placeholder - would integrate with historical data)
        quality_trends = {
            'overall_quality': quality_metrics.overall_quality_score,
            'iast_compliance': quality_metrics.iast_compliance_score,
            'sanskrit_accuracy': quality_metrics.sanskrit_accuracy_score
        }
        
        return AcademicStakeholderReport(
            executive_summary=executive_summary,
            quality_metrics=quality_metrics,
            enhancement_details=enhancement_details,
            compliance_findings=compliance_findings,
            recommendations=recommendations,
            iast_validation_results=iast_results,
            content_quality_trends=quality_trends
        )
    
    def _generate_executive_summary(self, quality_metrics: SemanticQualityMetrics, polish_issues: List[PolishIssue]) -> str:
        """Generate executive summary for stakeholders"""
        
        quality_level = "Excellent" if quality_metrics.overall_quality_score >= 0.9 else \
                       "Good" if quality_metrics.overall_quality_score >= 0.8 else \
                       "Adequate" if quality_metrics.overall_quality_score >= 0.7 else \
                       "Needs Improvement"
        
        critical_issues = quality_metrics.critical_issues_count
        total_enhancements = quality_metrics.polish_enhancement_count
        
        summary = f"""
ACADEMIC QUALITY ASSESSMENT SUMMARY

Overall Quality Level: {quality_level} ({quality_metrics.overall_quality_score:.1%})

Key Metrics:
• IAST Compliance: {quality_metrics.iast_compliance_score:.1%}
• Sanskrit Accuracy: {quality_metrics.sanskrit_accuracy_score:.1%}
• Terminology Consistency: {quality_metrics.terminology_consistency_score:.1%}
• Academic Formatting: {quality_metrics.academic_formatting_score:.1%}
• Semantic Coherence: {quality_metrics.semantic_coherence_score:.1%}

Processing Results:
• Total Academic Enhancements Applied: {total_enhancements}
• Critical Issues Requiring Review: {critical_issues}

The content has been processed through our enhanced academic quality pipeline, 
integrating semantic analysis with traditional text processing to ensure 
compliance with IAST transliteration standards and academic formatting requirements.
"""
        
        if critical_issues > 0:
            summary += f"\n⚠️  ATTENTION: {critical_issues} critical issue(s) require immediate expert review."
        else:
            summary += "\n✅ No critical issues detected. Content meets academic publication standards."
            
        return summary.strip()
    
    def _format_enhancement_details(self, polish_issues: List[PolishIssue]) -> List[Dict[str, Any]]:
        """Format polish enhancement details for reporting"""
        
        details = []
        
        # Group issues by type
        issue_groups = {}
        for issue in polish_issues:
            if issue.issue_type not in issue_groups:
                issue_groups[issue.issue_type] = []
            issue_groups[issue.issue_type].append(issue)
        
        # Format each group
        for issue_type, issues in issue_groups.items():
            details.append({
                'enhancement_type': issue_type.replace('_', ' ').title(),
                'count': len(issues),
                'priority_breakdown': {
                    'critical': len([i for i in issues if i.priority == 'critical']),
                    'major': len([i for i in issues if i.priority == 'major']),
                    'minor': len([i for i in issues if i.priority == 'minor'])
                },
                'examples': [
                    {
                        'line': issue.line_number,
                        'original': issue.original_text,
                        'corrected': issue.suggested_fix,
                        'description': issue.description
                    } for issue in issues[:3]  # Show first 3 examples
                ]
            })
        
        return details
    
    def _format_compliance_findings(self, quality_report: QualityReport) -> List[Dict[str, Any]]:
        """Format compliance findings from quality report"""
        
        findings = []
        
        for metric_name, score in quality_report.compliance_scores.items():
            compliance_level = "Excellent" if score >= 0.95 else \
                             "Good" if score >= 0.85 else \
                             "Adequate" if score >= 0.75 else \
                             "Needs Improvement"
            
            findings.append({
                'compliance_area': metric_name.replace('_', ' ').title(),
                'score': score,
                'level': compliance_level,
                'meets_academic_standard': score >= 0.8,
                'recommendations': self._get_compliance_recommendations(metric_name, score)
            })
        
        return findings
    
    def _get_compliance_recommendations(self, metric_name: str, score: float) -> List[str]:
        """Get specific recommendations based on compliance scores"""
        
        recommendations = []
        
        if metric_name == 'iast_compliance' and score < 0.9:
            recommendations.extend([
                "Review IAST transliteration accuracy for Sanskrit terms",
                "Verify diacritical mark usage in academic contexts",
                "Consider expert linguistic review for complex terms"
            ])
        
        if metric_name == 'sanskrit_accuracy' and score < 0.85:
            recommendations.extend([
                "Validate Sanskrit term recognition and correction",
                "Review lexicon entries for domain-specific terminology",
                "Consider contextual validation for ambiguous terms"
            ])
        
        if metric_name == 'terminology_consistency' and score < 0.8:
            recommendations.extend([
                "Standardize terminology usage across document",
                "Review proper noun capitalization patterns",
                "Validate spiritual and philosophical term usage"
            ])
        
        if metric_name == 'academic_formatting' and score < 0.85:
            recommendations.extend([
                "Review academic citation and reference formatting",
                "Validate sentence structure and capitalization",
                "Check punctuation and spacing consistency"
            ])
        
        return recommendations
    
    def _generate_recommendations(self, quality_metrics: SemanticQualityMetrics, polish_issues: List[PolishIssue]) -> List[str]:
        """Generate overall recommendations for quality improvement"""
        
        recommendations = []
        
        # Overall quality recommendations
        if quality_metrics.overall_quality_score < 0.8:
            recommendations.append("Consider comprehensive expert review before academic publication")
        
        # Critical issues
        if quality_metrics.critical_issues_count > 0:
            recommendations.append(f"Address {quality_metrics.critical_issues_count} critical issues requiring immediate attention")
        
        # Specific metric recommendations
        if quality_metrics.iast_compliance_score < 0.85:
            recommendations.append("Enhance IAST transliteration accuracy through expert linguistic review")
        
        if quality_metrics.sanskrit_accuracy_score < 0.8:
            recommendations.append("Improve Sanskrit term recognition and correction processes")
        
        if quality_metrics.terminology_consistency_score < 0.85:
            recommendations.append("Standardize terminology usage for better academic consistency")
        
        # Polish enhancement recommendations
        if quality_metrics.polish_enhancement_count > 50:
            recommendations.append("High number of corrections applied - consider review of source material quality")
        
        # Positive reinforcement
        if quality_metrics.overall_quality_score >= 0.9:
            recommendations.append("Content meets excellent academic standards - ready for publication workflow")
        
        return recommendations
    
    def _extract_iast_validation_results(self, quality_report: QualityReport) -> Dict[str, Any]:
        """Extract IAST-specific validation results"""
        
        return {
            'compliance_score': quality_report.compliance_scores.get('iast_compliance', 0.0),
            'total_sanskrit_terms_checked': quality_report.metrics.get('sanskrit_terms_validated', 0),
            'iast_corrections_applied': quality_report.metrics.get('iast_corrections', 0),
            'diacritical_accuracy': quality_report.metrics.get('diacritical_accuracy', 0.0),
            'transliteration_consistency': quality_report.metrics.get('transliteration_consistency', 0.0)
        }
    
    def _collect_processing_metrics(self, quality_metrics: SemanticQualityMetrics, context: Dict[str, Any]):
        """Collect metrics for trend analysis and monitoring"""
        
        metrics_data = {
            'overall_quality_score': quality_metrics.overall_quality_score,
            'iast_compliance_score': quality_metrics.iast_compliance_score,
            'sanskrit_accuracy_score': quality_metrics.sanskrit_accuracy_score,
            'terminology_consistency_score': quality_metrics.terminology_consistency_score,
            'academic_formatting_score': quality_metrics.academic_formatting_score,
            'semantic_coherence_score': quality_metrics.semantic_coherence_score,
            'polish_enhancement_count': quality_metrics.polish_enhancement_count,
            'critical_issues_count': quality_metrics.critical_issues_count,
            'processing_timestamp': quality_metrics.processing_timestamp
        }
        
        # Add context information
        metrics_data.update({
            'content_length': context.get('content_length', 0),
            'content_type': context.get('content_type', 'srt'),
            'processing_mode': 'semantic_enhanced'
        })
        
        # Collect through existing metrics system
        file_metrics = self.metrics_collector.create_file_metrics(
            context.get('filename', 'semantic_processing')
        )
        
        for key, value in metrics_data.items():
            file_metrics.add_metric(key, value)
        
        self.logger.info("Processing metrics collected for trend analysis")
    
    def integrate_with_existing_workflow(self, processor_instance) -> None:
        """
        Integrate with existing SanskritPostProcessor workflow
        Add semantic quality hooks to existing processing pipeline
        """
        
        # Store reference to original process method
        original_process = getattr(processor_instance, 'process_srt_file', None)
        
        if original_process:
            def enhanced_process_srt_file(input_file: str, output_file: str, **kwargs):
                """Enhanced processing with semantic quality integration"""
                
                # Read content
                with open(input_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Apply enhanced quality processing
                processed_content, stakeholder_report = self.process_with_enhanced_quality(
                    content, {'filename': input_file, **kwargs}
                )
                
                # Write processed content
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(processed_content)
                
                # Generate quality report file
                report_file = output_file.replace('.srt', '_quality_report.txt')
                with open(report_file, 'w', encoding='utf-8') as f:
                    f.write(stakeholder_report.executive_summary)
                
                return processed_content, stakeholder_report
            
            # Replace the method
            setattr(processor_instance, 'process_srt_file', enhanced_process_srt_file)
            self.logger.info("Enhanced semantic quality processing integrated with existing workflow")
        else:
            self.logger.warning("Could not integrate - original process method not found")


def create_academic_workflow_integration(config: Optional[Dict[str, Any]] = None) -> AcademicWorkflowIntegrator:
    """Factory function to create academic workflow integrator"""
    return AcademicWorkflowIntegrator(config)