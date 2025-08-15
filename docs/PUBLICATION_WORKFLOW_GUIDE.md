# Publication Workflow Guide
## Story 4.5: Research-Grade Scripture Processing

### Overview

This guide provides step-by-step instructions for preparing processed scripture content for academic publication. It covers all stages from content processing to final publication-ready output.

## Publication Pipeline

### Stage 1: Content Preparation

#### 1.1 Initial Processing
```bash
# Process SRT files with academic enhancement
python src/main.py process-academic \
  --input "data/raw_srts/lecture_series_001.srt" \
  --output "data/processed_srts/lecture_series_001_academic.srt" \
  --academic-level "publication_grade" \
  --citation-style "indological_standard"
```

#### 1.2 Academic Enhancement
```python
from scripture_processing.scripture_processor import ScriptureProcessor
from scripture_processing.academic_citation_manager import AcademicCitationManager

# Initialize with academic settings
processor = ScriptureProcessor(academic_enhancement=True)
citation_manager = AcademicCitationManager(
    citation_style="indological_standard",
    validation_level="publication_grade"
)

# Process with academic standards
result = processor.process_srt_file(
    input_file="data/raw_srts/source.srt",
    output_file="data/processed_srts/academic_output.srt",
    academic_config={
        "enable_citation_enhancement": True,
        "require_iast_validation": True,
        "publication_target": "peer_reviewed_journal"
    }
)
```

### Stage 2: Academic Validation

#### 2.1 Quality Assessment
```python
from utils.academic_validator import AcademicValidator

validator = AcademicValidator()

# Comprehensive quality check
quality_report = validator.validate_content(
    content_path="data/processed_srts/academic_output.srt",
    target_publication="peer_reviewed_journal"
)

print(f"Academic Compliance Score: {quality_report.academic_compliance_score:.2f}")
print(f"Citation Accuracy: {quality_report.citation_accuracy:.2f}")
print(f"IAST Compliance: {quality_report.iast_compliance:.2f}")
print(f"Publication Readiness: {quality_report.publication_readiness:.2f}")
```

#### 2.2 IAST Transliteration Verification
```python
# Verify IAST compliance
iast_report = validator.validate_iast_compliance(
    content="data/processed_srts/academic_output.srt"
)

if iast_report.compliance_score >= 0.98:
    print("✅ IAST compliance meets publication standards")
else:
    print("❌ IAST compliance needs improvement")
    for issue in iast_report.validation_issues:
        print(f"  - {issue.description} (Line {issue.line_number})")
```

### Stage 3: Publication Formatting

#### 3.1 Format Selection
```python
from scripture_processing.publication_formatter import PublicationFormatter

formatter = PublicationFormatter()

# Available publication formats
formats = {
    "academic_journal": {
        "citation_style": "indological_standard",
        "bibliography": True,
        "footnotes": True,
        "quality_threshold": 0.95
    },
    "book_chapter": {
        "citation_style": "chicago",
        "bibliography": True,
        "index_generation": True,
        "quality_threshold": 0.98
    },
    "conference_paper": {
        "citation_style": "apa",
        "bibliography": True,
        "abstract": True,
        "quality_threshold": 0.92
    }
}
```

#### 3.2 Document Generation
```python
# Generate publication-ready document
publication_result = formatter.format_for_publication(
    content_path="data/processed_srts/academic_output.srt",
    target_publication="academic_journal",
    output_format="latex",  # or "docx", "pdf", "html"
    citation_style="indological_standard"
)

# Generate supporting documents
bibliography = formatter.generate_bibliography(publication_result)
quality_report = formatter.generate_quality_report(publication_result)
```

### Stage 4: Citation Management

#### 4.1 Citation Generation
```python
from scripture_processing.academic_citation_manager import AcademicCitationManager

citation_manager = AcademicCitationManager()

# Generate citations for scriptural references
citations = citation_manager.generate_citations_for_content(
    content_path="data/processed_srts/academic_output.srt",
    citation_style="indological_standard"
)

# Format bibliography
bibliography = citation_manager.format_bibliography(citations)
```

#### 4.2 Citation Styles Configuration
```yaml
# Citation style examples in config/academic_standards_config.yaml

citation_styles:
  indological_standard:
    verse_citation: "{source_abbrev} {chapter}.{verse}"
    full_citation: "{title}, {chapter}.{verse} ({translator}, {year})"
    with_text: "{citation}: \"{canonical_text}\" ({translation})"
    bibliography_format: "{title}. Translated by {translator}. {publisher}, {year}."
  
  mla:
    verse_citation: "({source_abbrev} {chapter}.{verse})"
    full_citation: "{title} {chapter}.{verse}"
    bibliography_format: "{title}. Trans. {translator}. {publisher}, {year}. Print."
```

### Stage 5: Consultant Review

#### 5.1 Review Submission
```python
# Submit for academic consultant review
review_submission = formatter.submit_for_consultant_review(
    content=publication_result,
    consultant_email="sanskrit.expert@university.edu",
    review_type="publication_readiness",
    priority="standard",
    notes="Academic journal submission - please verify citation accuracy"
)

print(f"Review submitted: {review_submission.review_id}")
print(f"Expected completion: {review_submission.expected_completion_date}")
```

#### 5.2 Review Tracking
```python
# Check review status
status = formatter.get_review_status(review_submission.review_id)

print(f"Current status: {status.current_status}")
print(f"Consultant feedback: {status.consultant_feedback}")

if status.approval_status == "approved":
    print("✅ Content approved for publication")
elif status.approval_status == "requires_revision":
    print("📝 Revisions required:")
    for revision in status.required_revisions:
        print(f"  - {revision.description}")
```

### Stage 6: Final Export

#### 6.1 Export Formats
```python
# Export in multiple formats for different publication venues
export_options = {
    "latex": {
        "template": "academic_journal",
        "include_bibliography": True,
        "include_footnotes": True
    },
    "docx": {
        "template": "manuscript",
        "citation_style": "indological_standard"
    },
    "html": {
        "template": "web_publication",
        "include_interactive_citations": True
    },
    "pdf": {
        "template": "camera_ready",
        "include_metadata": True
    }
}

# Generate final exports
for format_type, options in export_options.items():
    export_result = formatter.export_document(
        content=publication_result,
        format=format_type,
        options=options,
        output_path=f"output/publication.{format_type}"
    )
    print(f"✅ {format_type.upper()} export completed: {export_result.output_path}")
```

#### 6.2 Quality Validation
```python
# Final quality check before publication
final_validation = validator.final_publication_check(
    document_path="output/publication.pdf",
    target_venue="peer_reviewed_journal"
)

if final_validation.ready_for_publication:
    print("🎉 Document ready for publication!")
    print(f"Final quality score: {final_validation.overall_quality_score:.2f}")
else:
    print("❌ Document requires additional work:")
    for issue in final_validation.remaining_issues:
        print(f"  - {issue.description} (Priority: {issue.priority})")
```

## Quality Standards by Publication Type

### Academic Journal (Standard: 95%)
- **Citation Accuracy**: ≥95%
- **IAST Compliance**: ≥98%
- **Academic Formatting**: ≥92%
- **Consultant Review**: Required
- **Bibliography**: Complete and accurate

### Book Publication (Standard: 98%)
- **Citation Accuracy**: ≥98%
- **IAST Compliance**: ≥99%
- **Academic Formatting**: ≥95%
- **Consultant Review**: Multiple reviews required
- **Index Generation**: Required

### Conference Paper (Standard: 92%)
- **Citation Accuracy**: ≥92%
- **IAST Compliance**: ≥95%
- **Academic Formatting**: ≥90%
- **Abstract**: Required
- **Bibliography**: Required

## Example Workflows

### Workflow 1: Journal Article Preparation
```bash
# 1. Process content
python src/main.py process-academic \
  --input "lecture_series.srt" \
  --academic-level "publication_grade"

# 2. Validate quality
python -c "
from utils.academic_validator import AcademicValidator
validator = AcademicValidator()
report = validator.validate_content('processed_content.srt', 'academic_journal')
print(f'Ready for publication: {report.publication_readiness >= 0.95}')
"

# 3. Generate formatted output
python -c "
from scripture_processing.publication_formatter import PublicationFormatter
formatter = PublicationFormatter()
result = formatter.format_for_publication(
    'processed_content.srt', 
    'academic_journal',
    'latex'
)
print(f'Publication document: {result.output_path}')
"
```

### Workflow 2: Book Chapter Preparation
```bash
# Enhanced processing for book-level quality
python src/main.py process-academic \
  --input "chapter_content.srt" \
  --academic-level "book_publication" \
  --citation-style "chicago" \
  --require-multiple-reviews

# Generate book-ready format
python -c "
from scripture_processing.publication_formatter import PublicationFormatter
formatter = PublicationFormatter()
result = formatter.format_for_publication(
    'chapter_content_processed.srt',
    'book_chapter',
    'docx',
    {'include_index': True, 'include_bibliography': True}
)
"
```

## Troubleshooting

### Common Issues

1. **Low IAST Compliance Score**
   ```python
   # Check specific IAST issues
   iast_report = validator.validate_iast_compliance(content)
   for issue in iast_report.validation_issues:
       print(f"Line {issue.line_number}: {issue.description}")
       print(f"Suggestion: {issue.correction_suggestion}")
   ```

2. **Citation Format Issues**
   ```python
   # Validate citation formatting
   citation_report = citation_manager.validate_citations(content)
   for error in citation_report.formatting_errors:
       print(f"Citation error: {error.description}")
       print(f"Expected format: {error.expected_format}")
   ```

3. **Publication Readiness Gaps**
   ```python
   # Identify specific readiness issues
   readiness_report = formatter.analyze_publication_readiness(content)
   for requirement in readiness_report.unmet_requirements:
       print(f"Missing: {requirement.description}")
       print(f"Required for: {requirement.publication_type}")
   ```

### Support Resources

- **Academic Standards**: `config/academic_standards_config.yaml`
- **Citation Templates**: Source code in `academic_citation_manager.py`
- **Quality Metrics**: Documentation in `utils/academic_validator.py`
- **Consultant Workflow**: `docs/ACADEMIC_CONSULTANT_WORKFLOW.md`

---

*Publication Workflow Guide v1.0*  
*Last updated: 2025-08-15*  
*For Story 4.5: Scripture Intelligence Enhancement*