# Academic Consultant Workflow Guide
## Story 4.5: Scripture Intelligence Enhancement

### Overview

This guide provides detailed instructions for academic consultants working with the Scripture Intelligence Enhancement system. The workflow ensures research publication quality standards are met throughout the development and validation process.

## Academic Consultant Integration Process

### Phase 1: Onboarding and System Assessment (Week 3)

#### Initial Setup
1. **System Access Setup**
   ```bash
   # Clone the repository
   git clone <repository-url>
   cd post-processing-shruti
   
   # Activate virtual environment
   source .venv/bin/activate  # Linux/Mac
   # OR
   .venv\Scripts\activate.bat  # Windows
   
   # Verify installation
   python test_story_4_5_simple.py
   ```

2. **Academic Standards Review**
   - Review current IAST transliteration implementation
   - Assess existing citation standards compliance
   - Evaluate sanskrit/Hindi lexicon quality
   - Identify gaps in academic formatting

3. **Configuration Assessment**
   ```yaml
   # Review: config/academic_standards_config.yaml
   academic_standards:
     citation_standards:
       default_style: "indological_standard"
       require_peer_review: true
       publication_grade_validation: true
   ```

#### Documentation Tasks
- [ ] Complete Academic Standards Assessment Report
- [ ] Identify improvement areas for publication readiness
- [ ] Document required citation style configurations
- [ ] Establish quality benchmarks for academic validation

### Phase 2: Standards Implementation and Validation (Week 4-5)

#### Academic Citation Standards Implementation

1. **Citation Style Configuration**
   ```python
   from scripture_processing.academic_citation_manager import (
       AcademicCitationManager, 
       CitationStyle,
       CitationValidationLevel
   )
   
   # Initialize with consultant-approved settings
   citation_manager = AcademicCitationManager(
       default_style=CitationStyle.INDOLOGICAL_STANDARD,
       validation_level=CitationValidationLevel.PUBLICATION_GRADE,
       require_peer_review=True
   )
   ```

2. **IAST Transliteration Validation**
   - Review all Sanskrit term transliterations
   - Validate adherence to academic IAST standards
   - Approve lexicon entries for publication use
   - Configure transliteration validation rules

3. **Quality Metrics Establishment**
   ```python
   from utils.academic_validator import AcademicValidator
   
   validator = AcademicValidator()
   
   # Set consultant-approved quality thresholds
   quality_standards = {
       "citation_accuracy": 0.95,
       "iast_compliance": 0.98,
       "academic_formatting": 0.92,
       "publication_readiness": 0.90
   }
   ```

#### Validation Workflow

1. **Content Review Process**
   ```bash
   # Run academic validation
   python -c "
   from utils.academic_validator import AcademicValidator
   validator = AcademicValidator()
   
   # Generate comprehensive quality report
   report = validator.generate_comprehensive_report('sample_text.srt')
   print(f'Academic compliance: {report.academic_compliance_score:.2f}')
   "
   ```

2. **Citation Standards Verification**
   - Verify all scriptural references follow academic standards
   - Validate bibliography generation
   - Approve citation formatting templates
   - Review cross-reference accuracy

3. **Quality Assurance Checklist**
   - [ ] IAST transliteration accuracy ≥98%
   - [ ] Citation formatting compliance ≥95%
   - [ ] Academic bibliography generation functional
   - [ ] Peer review workflow operational
   - [ ] Publication-ready output formatting

### Phase 3: Research Publication Readiness (Week 6-7)

#### Publication Quality Assessment

1. **Research-Grade Validation**
   ```python
   from scripture_processing.publication_formatter import PublicationFormatter
   
   formatter = PublicationFormatter()
   
   # Generate publication-ready content
   pub_result = formatter.format_for_publication(
       content="processed_content.srt",
       target_publication="academic_journal",
       citation_style="indological_standard"
   )
   
   # Validate publication readiness
   readiness = formatter.validate_publication_readiness(pub_result)
   ```

2. **Academic Formatting Validation**
   - Review formatted output for academic standards
   - Validate citation consistency throughout document
   - Approve bibliography formatting
   - Verify footnote and reference accuracy

3. **Consultant Review Workflow**
   ```python
   # Submit for consultant review
   review_submission = formatter.submit_for_consultant_review(
       content=pub_result,
       consultant_id="academic_consultant_001",
       review_type="publication_readiness"
   )
   
   # Track review status
   status = formatter.get_review_status(review_submission.review_id)
   ```

#### Publication Standards Configuration

1. **Academic Publication Targets**
   ```yaml
   publication_targets:
     basic_academic:
       citation_accuracy: 0.90
       iast_compliance: 0.95
     peer_reviewed_journal:
       citation_accuracy: 0.95
       iast_compliance: 0.98
       require_consultant_approval: true
     book_publication:
       citation_accuracy: 0.98
       iast_compliance: 0.99
       require_multiple_reviews: true
   ```

2. **Quality Validation Pipeline**
   - Automated academic compliance checking
   - Consultant approval workflow
   - Publication readiness scoring
   - Export format validation

### Phase 4: Final Academic Compliance Certification (Week 8)

#### Certification Process

1. **Comprehensive System Validation**
   ```bash
   # Run complete validation suite
   python test_story_4_5_final_validation.py
   
   # Generate academic compliance report
   python -c "
   from utils.academic_validator import AcademicValidator
   from scripture_processing.publication_formatter import PublicationFormatter
   
   validator = AcademicValidator()
   formatter = PublicationFormatter()
   
   # Generate final certification report
   cert_report = validator.generate_certification_report()
   print(f'System certified for academic use: {cert_report.certification_status}')
   "
   ```

2. **Final Quality Metrics**
   - [ ] Academic compliance score ≥95%
   - [ ] IAST transliteration accuracy ≥98%
   - [ ] Citation formatting accuracy ≥95%
   - [ ] Publication readiness score ≥90%
   - [ ] Consultant approval obtained

3. **Academic Certification Documentation**
   - Complete Academic Standards Compliance Report
   - Publication Readiness Certification
   - Quality Metrics Documentation
   - Consultant Approval Documentation

## Consultant Review Interface

### Review Submission System

```python
from scripture_processing.publication_formatter import PublicationFormatter

formatter = PublicationFormatter()

# Submit content for academic review
review_request = formatter.submit_for_consultant_review(
    content_path="data/processed_srts/lecture_001.srt",
    consultant_email="consultant@academic-institution.edu",
    review_type="publication_readiness",
    deadline="2025-08-30",
    notes="Initial review for academic journal submission"
)

# Track review progress
status = formatter.get_review_status(review_request.review_id)
print(f"Review status: {status.current_status}")
print(f"Consultant feedback: {status.consultant_feedback}")
```

### Quality Metrics Dashboard

```python
from utils.academic_validator import AcademicValidator

validator = AcademicValidator()

# Generate consultant dashboard
dashboard = validator.generate_consultant_dashboard()

print("Academic Quality Dashboard:")
print(f"- Citation Accuracy: {dashboard.citation_accuracy:.2f}%")
print(f"- IAST Compliance: {dashboard.iast_compliance:.2f}%")
print(f"- Publication Readiness: {dashboard.publication_readiness:.2f}%")
print(f"- Content Requiring Review: {dashboard.items_requiring_review}")
```

## Academic Standards Configuration

### Citation Styles Supported

- **Indological Standard**: Traditional Sanskrit studies citation format
- **MLA**: Modern Language Association format
- **APA**: American Psychological Association format
- **Chicago**: Chicago Manual of Style format
- **Custom Academic**: Institution-specific formatting

### IAST Transliteration Standards

- Unicode-compliant IAST character set
- Academic diacritical mark requirements
- Consistent transliteration patterns
- Quality validation and error detection

### Publication Quality Levels

1. **Basic Academic** (90% threshold)
   - Standard citation formatting
   - Basic IAST compliance
   - General academic formatting

2. **Peer-Reviewed Journal** (95% threshold)
   - Rigorous citation standards
   - High IAST accuracy
   - Consultant review required

3. **Book Publication** (98% threshold)
   - Exceptional quality standards
   - Multiple consultant reviews
   - Publication-ready formatting

## Support and Contact

### Technical Support
- System issues: `tech-support@project.org`
- Configuration help: See `config/academic_standards_config.yaml`
- API documentation: See `docs/API_REFERENCE.md`

### Academic Support
- Standards questions: `academic-coordinator@project.org`
- Citation format issues: Review citation templates in source code
- Quality concerns: Use the consultant review submission system

### Emergency Contact
- Critical academic compliance issues: `emergency-academic@project.org`
- Immediate consultant review needed: Use priority review submission

---

*Last updated: 2025-08-15*  
*Document version: 1.0*  
*For Story 4.5: Scripture Intelligence Enhancement*