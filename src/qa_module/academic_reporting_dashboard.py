"""
Academic Stakeholder Reporting Dashboard for Story 3.6
Provides enhanced reporting interfaces for academic stakeholders
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import csv
from datetime import datetime, timedelta
import logging

from qa_module.academic_workflow_integrator import SemanticQualityMetrics, AcademicStakeholderReport


@dataclass 
class AcademicReportingConfig:
    """Configuration for academic reporting dashboard"""
    output_directory: str = "reports/academic"
    enable_html_reports: bool = True
    enable_pdf_export: bool = False  # Requires additional dependencies
    enable_csv_export: bool = True
    enable_json_export: bool = True
    report_retention_days: int = 90
    quality_threshold_excellent: float = 0.9
    quality_threshold_good: float = 0.8
    quality_threshold_adequate: float = 0.7


class AcademicReportingDashboard:
    """
    Academic stakeholder reporting dashboard
    Generates comprehensive reports for linguistic experts and academic reviewers
    """
    
    def __init__(self, config: Optional[AcademicReportingConfig] = None):
        """Initialize the academic reporting dashboard"""
        self.config = config or AcademicReportingConfig()
        self.logger = logging.getLogger(__name__)
        
        # Ensure output directory exists
        self.output_dir = Path(self.config.output_directory)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Academic Reporting Dashboard initialized. Output: {self.output_dir}")
    
    def generate_comprehensive_report(self, 
                                    stakeholder_report: AcademicStakeholderReport,
                                    filename: str,
                                    additional_context: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        """
        Generate comprehensive academic stakeholder report in multiple formats
        
        Args:
            stakeholder_report: The academic stakeholder report
            filename: Base filename for reports
            additional_context: Additional context for reporting
            
        Returns:
            Dict mapping format types to generated file paths
        """
        
        additional_context = additional_context or {}
        report_timestamp = datetime.now().isoformat()
        
        generated_files = {}
        
        # Generate HTML report
        if self.config.enable_html_reports:
            html_path = self._generate_html_report(stakeholder_report, filename, report_timestamp, additional_context)
            generated_files['html'] = str(html_path)
        
        # Generate CSV export
        if self.config.enable_csv_export:
            csv_path = self._generate_csv_export(stakeholder_report, filename, report_timestamp)
            generated_files['csv'] = str(csv_path)
        
        # Generate JSON export
        if self.config.enable_json_export:
            json_path = self._generate_json_export(stakeholder_report, filename, report_timestamp, additional_context)
            generated_files['json'] = str(json_path)
        
        # Generate summary report
        summary_path = self._generate_summary_report(stakeholder_report, filename, report_timestamp)
        generated_files['summary'] = str(summary_path)
        
        self.logger.info(f"Generated {len(generated_files)} report files for {filename}")
        
        return generated_files
    
    def _generate_html_report(self, 
                            report: AcademicStakeholderReport,
                            filename: str,
                            timestamp: str,
                            context: Dict[str, Any]) -> Path:
        """Generate comprehensive HTML report for web viewing"""
        
        html_filename = f"{filename}_academic_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        html_path = self.output_dir / html_filename
        
        # Generate quality indicators
        quality_indicators = self._generate_quality_indicators_html(report.quality_metrics)
        
        # Generate enhancement details
        enhancement_html = self._generate_enhancement_details_html(report.enhancement_details)
        
        # Generate compliance findings
        compliance_html = self._generate_compliance_findings_html(report.compliance_findings)
        
        # Generate recommendations
        recommendations_html = self._generate_recommendations_html(report.recommendations)
        
        # Generate IAST validation results
        iast_html = self._generate_iast_validation_html(report.iast_validation_results)
        
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Academic Quality Assessment Report - {filename}</title>
    <style>
        body {{ 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            margin: 0; 
            padding: 20px; 
            background-color: #f8f9fa; 
        }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .header {{ 
            border-bottom: 3px solid #007bff; 
            padding-bottom: 20px; 
            margin-bottom: 30px; 
            text-align: center;
        }}
        .header h1 {{ color: #007bff; margin: 0; }}
        .header .metadata {{ color: #6c757d; margin-top: 10px; }}
        .section {{ margin-bottom: 30px; }}
        .section h2 {{ 
            color: #495057; 
            border-left: 4px solid #007bff; 
            padding-left: 15px; 
            margin-bottom: 20px;
        }}
        .quality-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; }}
        .quality-card {{ 
            background: #f8f9fa; 
            border: 1px solid #dee2e6; 
            border-radius: 6px; 
            padding: 20px; 
            text-align: center;
        }}
        .quality-score {{ font-size: 2em; font-weight: bold; margin: 10px 0; }}
        .score-excellent {{ color: #28a745; }}
        .score-good {{ color: #17a2b8; }}
        .score-adequate {{ color: #ffc107; }}
        .score-poor {{ color: #dc3545; }}
        .enhancement-item {{ 
            background: #f8f9fa; 
            border-left: 4px solid #007bff; 
            padding: 15px; 
            margin-bottom: 15px;
        }}
        .compliance-item {{ 
            background: #f8f9fa; 
            border-left: 4px solid #28a745; 
            padding: 15px; 
            margin-bottom: 15px;
        }}
        .recommendation {{ 
            background: #fff3cd; 
            border: 1px solid #ffeaa7; 
            padding: 10px; 
            margin-bottom: 10px; 
            border-radius: 4px;
        }}
        .critical {{ border-left-color: #dc3545 !important; }}
        .major {{ border-left-color: #ffc107 !important; }}
        .minor {{ border-left-color: #28a745 !important; }}
        .executive-summary {{ 
            background: #e9ecef; 
            padding: 20px; 
            border-radius: 6px; 
            margin-bottom: 30px;
            white-space: pre-line;
        }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 15px; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #dee2e6; }}
        th {{ background-color: #f8f9fa; font-weight: 600; }}
        .footer {{ 
            margin-top: 40px; 
            padding-top: 20px; 
            border-top: 1px solid #dee2e6; 
            text-align: center; 
            color: #6c757d; 
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Academic Quality Assessment Report</h1>
            <div class="metadata">
                <strong>File:</strong> {filename}<br>
                <strong>Generated:</strong> {timestamp}<br>
                <strong>Processing Mode:</strong> Semantic Enhanced Academic Workflow
            </div>
        </div>
        
        <div class="section">
            <h2>Executive Summary</h2>
            <div class="executive-summary">{report.executive_summary}</div>
        </div>
        
        <div class="section">
            <h2>Quality Metrics Overview</h2>
            {quality_indicators}
        </div>
        
        <div class="section">
            <h2>Academic Enhancement Details</h2>
            {enhancement_html}
        </div>
        
        <div class="section">
            <h2>Compliance Assessment</h2>
            {compliance_html}
        </div>
        
        <div class="section">
            <h2>IAST Transliteration Validation</h2>
            {iast_html}
        </div>
        
        <div class="section">
            <h2>Recommendations for Academic Excellence</h2>
            {recommendations_html}
        </div>
        
        <div class="footer">
            <p>Generated by Academic Workflow Integration System (Story 3.6)<br>
            Post-Processing Pipeline - Semantic Enhanced Quality Assurance</p>
        </div>
    </div>
</body>
</html>
"""
        
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        self.logger.info(f"HTML report generated: {html_path}")
        return html_path
    
    def _generate_quality_indicators_html(self, metrics: SemanticQualityMetrics) -> str:
        """Generate HTML for quality indicators grid"""
        
        def get_score_class(score: float) -> str:
            if score >= self.config.quality_threshold_excellent:
                return "score-excellent"
            elif score >= self.config.quality_threshold_good:
                return "score-good"
            elif score >= self.config.quality_threshold_adequate:
                return "score-adequate"
            else:
                return "score-poor"
        
        return f"""
        <div class="quality-grid">
            <div class="quality-card">
                <h3>Overall Quality</h3>
                <div class="quality-score {get_score_class(metrics.overall_quality_score)}">
                    {metrics.overall_quality_score:.1%}
                </div>
                <p>Academic Excellence Standard</p>
            </div>
            <div class="quality-card">
                <h3>IAST Compliance</h3>
                <div class="quality-score {get_score_class(metrics.iast_compliance_score)}">
                    {metrics.iast_compliance_score:.1%}
                </div>
                <p>Transliteration Accuracy</p>
            </div>
            <div class="quality-card">
                <h3>Sanskrit Accuracy</h3>
                <div class="quality-score {get_score_class(metrics.sanskrit_accuracy_score)}">
                    {metrics.sanskrit_accuracy_score:.1%}
                </div>
                <p>Term Recognition & Correction</p>
            </div>
            <div class="quality-card">
                <h3>Terminology Consistency</h3>
                <div class="quality-score {get_score_class(metrics.terminology_consistency_score)}">
                    {metrics.terminology_consistency_score:.1%}
                </div>
                <p>Standardized Usage</p>
            </div>
            <div class="quality-card">
                <h3>Academic Formatting</h3>
                <div class="quality-score {get_score_class(metrics.academic_formatting_score)}">
                    {metrics.academic_formatting_score:.1%}
                </div>
                <p>Publication Standards</p>
            </div>
            <div class="quality-card">
                <h3>Semantic Coherence</h3>
                <div class="quality-score {get_score_class(metrics.semantic_coherence_score)}">
                    {metrics.semantic_coherence_score:.1%}
                </div>
                <p>Contextual Understanding</p>
            </div>
        </div>
        <div style="margin-top: 20px; padding: 15px; background: #e9ecef; border-radius: 6px;">
            <strong>Processing Statistics:</strong><br>
            • Total Academic Enhancements: {metrics.polish_enhancement_count}<br>
            • Critical Issues Identified: {metrics.critical_issues_count}<br>
            • Processing Completed: {metrics.processing_timestamp}
        </div>
        """
    
    def _generate_enhancement_details_html(self, enhancement_details: List[Dict[str, Any]]) -> str:
        """Generate HTML for enhancement details"""
        
        if not enhancement_details:
            return "<p>No enhancements were required. Content already meets academic standards.</p>"
        
        html = ""
        
        for detail in enhancement_details:
            priority_breakdown = detail['priority_breakdown']
            examples_html = ""
            
            for example in detail['examples']:
                examples_html += f"""
                <div style="margin: 10px 0; padding: 10px; background: white; border-radius: 4px;">
                    <strong>Line {example['line']}:</strong> {example['description']}<br>
                    <span style="color: #dc3545;">Original:</span> {example['original']}<br>
                    <span style="color: #28a745;">Corrected:</span> {example['corrected']}
                </div>
                """
            
            html += f"""
            <div class="enhancement-item">
                <h3>{detail['enhancement_type']} ({detail['count']} instances)</h3>
                <p><strong>Priority Breakdown:</strong> 
                   Critical: {priority_breakdown['critical']}, 
                   Major: {priority_breakdown['major']}, 
                   Minor: {priority_breakdown['minor']}</p>
                <div style="margin-top: 15px;">
                    <strong>Examples:</strong>
                    {examples_html}
                </div>
            </div>
            """
        
        return html
    
    def _generate_compliance_findings_html(self, compliance_findings: List[Dict[str, Any]]) -> str:
        """Generate HTML for compliance findings"""
        
        html = ""
        
        for finding in compliance_findings:
            compliance_class = "score-excellent" if finding['score'] >= 0.9 else \
                             "score-good" if finding['score'] >= 0.8 else \
                             "score-adequate" if finding['score'] >= 0.7 else "score-poor"
            
            recommendations_html = ""
            for rec in finding['recommendations']:
                recommendations_html += f"<li>{rec}</li>"
            
            status_icon = "✅" if finding['meets_academic_standard'] else "⚠️"
            
            html += f"""
            <div class="compliance-item">
                <h3>{status_icon} {finding['compliance_area']}</h3>
                <div style="display: flex; align-items: center; margin: 10px 0;">
                    <span class="quality-score {compliance_class}" style="font-size: 1.2em; margin-right: 15px;">
                        {finding['score']:.1%}
                    </span>
                    <span>Level: <strong>{finding['level']}</strong></span>
                </div>
                {f'<div style="margin-top: 15px;"><strong>Recommendations:</strong><ul>{recommendations_html}</ul></div>' if finding['recommendations'] else ''}
            </div>
            """
        
        return html
    
    def _generate_recommendations_html(self, recommendations: List[str]) -> str:
        """Generate HTML for recommendations"""
        
        if not recommendations:
            return "<p>No specific recommendations. Content quality is excellent.</p>"
        
        html = ""
        for i, rec in enumerate(recommendations, 1):
            html += f'<div class="recommendation">{i}. {rec}</div>'
        
        return html
    
    def _generate_iast_validation_html(self, iast_results: Dict[str, Any]) -> str:
        """Generate HTML for IAST validation results"""
        
        return f"""
        <table>
            <tr><th>Validation Metric</th><th>Value</th><th>Status</th></tr>
            <tr>
                <td>IAST Compliance Score</td>
                <td>{iast_results.get('compliance_score', 0.0):.1%}</td>
                <td>{'✅ Excellent' if iast_results.get('compliance_score', 0.0) >= 0.9 else '⚠️ Needs Review'}</td>
            </tr>
            <tr>
                <td>Sanskrit Terms Validated</td>
                <td>{iast_results.get('total_sanskrit_terms_checked', 0)}</td>
                <td>{'✅ Processed' if iast_results.get('total_sanskrit_terms_checked', 0) > 0 else 'ℹ️ None Found'}</td>
            </tr>
            <tr>
                <td>IAST Corrections Applied</td>
                <td>{iast_results.get('iast_corrections_applied', 0)}</td>
                <td>{'ℹ️ Corrections Made' if iast_results.get('iast_corrections_applied', 0) > 0 else '✅ No Corrections Needed'}</td>
            </tr>
            <tr>
                <td>Diacritical Mark Accuracy</td>
                <td>{iast_results.get('diacritical_accuracy', 0.0):.1%}</td>
                <td>{'✅ Accurate' if iast_results.get('diacritical_accuracy', 0.0) >= 0.9 else '⚠️ Review Needed'}</td>
            </tr>
            <tr>
                <td>Transliteration Consistency</td>
                <td>{iast_results.get('transliteration_consistency', 0.0):.1%}</td>
                <td>{'✅ Consistent' if iast_results.get('transliteration_consistency', 0.0) >= 0.85 else '⚠️ Inconsistencies Found'}</td>
            </tr>
        </table>
        """
    
    def _generate_csv_export(self, report: AcademicStakeholderReport, filename: str, timestamp: str) -> Path:
        """Generate CSV export for data analysis"""
        
        csv_filename = f"{filename}_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        csv_path = self.output_dir / csv_filename
        
        # Prepare data rows
        csv_data = []
        
        # Quality metrics row
        metrics = report.quality_metrics
        csv_data.append({
            'filename': filename,
            'timestamp': timestamp,
            'overall_quality_score': metrics.overall_quality_score,
            'iast_compliance_score': metrics.iast_compliance_score,
            'sanskrit_accuracy_score': metrics.sanskrit_accuracy_score,
            'terminology_consistency_score': metrics.terminology_consistency_score,
            'academic_formatting_score': metrics.academic_formatting_score,
            'semantic_coherence_score': metrics.semantic_coherence_score,
            'polish_enhancement_count': metrics.polish_enhancement_count,
            'critical_issues_count': metrics.critical_issues_count,
            'processing_timestamp': metrics.processing_timestamp
        })
        
        # Write CSV
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            if csv_data:
                fieldnames = csv_data[0].keys()
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(csv_data)
        
        self.logger.info(f"CSV export generated: {csv_path}")
        return csv_path
    
    def _generate_json_export(self, 
                            report: AcademicStakeholderReport, 
                            filename: str, 
                            timestamp: str,
                            context: Dict[str, Any]) -> Path:
        """Generate JSON export for programmatic access"""
        
        json_filename = f"{filename}_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        json_path = self.output_dir / json_filename
        
        # Convert report to dictionary
        json_data = {
            'metadata': {
                'filename': filename,
                'generation_timestamp': timestamp,
                'report_version': '3.6.0',
                'processing_context': context
            },
            'executive_summary': report.executive_summary,
            'quality_metrics': asdict(report.quality_metrics),
            'enhancement_details': report.enhancement_details,
            'compliance_findings': report.compliance_findings,
            'recommendations': report.recommendations,
            'iast_validation_results': report.iast_validation_results,
            'content_quality_trends': report.content_quality_trends
        }
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"JSON export generated: {json_path}")
        return json_path
    
    def _generate_summary_report(self, report: AcademicStakeholderReport, filename: str, timestamp: str) -> Path:
        """Generate concise summary report for quick review"""
        
        summary_filename = f"{filename}_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        summary_path = self.output_dir / summary_filename
        
        metrics = report.quality_metrics
        
        summary_content = f"""
ACADEMIC QUALITY ASSESSMENT SUMMARY
{'=' * 50}

File: {filename}
Generated: {timestamp}
Processing Mode: Semantic Enhanced Academic Workflow

QUALITY METRICS:
Overall Quality Score: {metrics.overall_quality_score:.1%}
IAST Compliance: {metrics.iast_compliance_score:.1%}
Sanskrit Accuracy: {metrics.sanskrit_accuracy_score:.1%}
Terminology Consistency: {metrics.terminology_consistency_score:.1%}
Academic Formatting: {metrics.academic_formatting_score:.1%}
Semantic Coherence: {metrics.semantic_coherence_score:.1%}

PROCESSING RESULTS:
Total Enhancements: {metrics.polish_enhancement_count}
Critical Issues: {metrics.critical_issues_count}

EXECUTIVE SUMMARY:
{report.executive_summary}

TOP RECOMMENDATIONS:
"""
        
        for i, rec in enumerate(report.recommendations[:5], 1):  # Top 5 recommendations
            summary_content += f"{i}. {rec}\n"
        
        summary_content += f"\n{'=' * 50}\nReport generated by Academic Workflow Integration System (Story 3.6)\n"
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(summary_content)
        
        self.logger.info(f"Summary report generated: {summary_path}")
        return summary_path
    
    def cleanup_old_reports(self) -> int:
        """Clean up old reports based on retention policy"""
        
        if self.config.report_retention_days <= 0:
            return 0
        
        cutoff_date = datetime.now() - timedelta(days=self.config.report_retention_days)
        deleted_count = 0
        
        for file_path in self.output_dir.iterdir():
            if file_path.is_file():
                try:
                    file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                    if file_time < cutoff_date:
                        file_path.unlink()
                        deleted_count += 1
                except Exception as e:
                    self.logger.warning(f"Error deleting old report {file_path}: {e}")
        
        if deleted_count > 0:
            self.logger.info(f"Cleaned up {deleted_count} old report files")
        
        return deleted_count


def create_academic_reporting_dashboard(config: Optional[AcademicReportingConfig] = None) -> AcademicReportingDashboard:
    """Factory function to create academic reporting dashboard"""
    return AcademicReportingDashboard(config)