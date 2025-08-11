#!/usr/bin/env python3
"""
Validation script to verify QA corrections were properly applied
"""

from qa_quality_validation_rules import SRTQualityValidator
from pathlib import Path

def validate_corrected_files(input_dir: str = "data/processed_srts"):
    """Validate that QA_CORRECTED files have fewer issues than original emergency_safe files"""
    
    input_path = Path(input_dir)
    validator = SRTQualityValidator()
    
    results = {
        'before': {},
        'after': {},
        'improvement': {}
    }
    
    print("CORRECTION VALIDATION REPORT")
    print("=" * 50)
    
    for qa_file in input_path.glob("*_QA_CORRECTED.srt"):
        base_name = qa_file.stem.replace("_QA_CORRECTED", "_emergency_safe")
        original_file = input_path / f"{base_name}.srt"
        
        if original_file.exists():
            # Validate original file
            with open(original_file, 'r', encoding='utf-8') as f:
                original_content = f.read()
            original_issues = validator.validate_srt_content(original_content)
            
            # Validate corrected file  
            with open(qa_file, 'r', encoding='utf-8') as f:
                corrected_content = f.read()
            corrected_issues = validator.validate_srt_content(corrected_content)
            
            # Calculate improvement
            original_count = len(original_issues)
            corrected_count = len(corrected_issues)
            improvement = original_count - corrected_count
            improvement_pct = (improvement / original_count * 100) if original_count > 0 else 100
            
            results['before'][qa_file.name] = original_count
            results['after'][qa_file.name] = corrected_count
            results['improvement'][qa_file.name] = improvement
            
            print(f"\nFILE: {qa_file.name}")
            print(f"Before: {original_count} issues")
            print(f"After: {corrected_count} issues") 
            print(f"Improvement: {improvement} issues ({improvement_pct:.1f}% reduction)")
            
            if corrected_count > 0:
                print(f"WARNING: {corrected_count} issues remain")
                # Show remaining issues
                for issue in corrected_issues[:5]:  # Show first 5 remaining issues
                    print(f"  - Line {issue.line_number}: {issue.description}")
                if len(corrected_issues) > 5:
                    print(f"  - ... and {len(corrected_issues) - 5} more")
    
    # Summary
    total_before = sum(results['before'].values())
    total_after = sum(results['after'].values())
    total_improvement = total_before - total_after
    overall_pct = (total_improvement / total_before * 100) if total_before > 0 else 100
    
    print(f"\nOVERALL SUMMARY:")
    print(f"Total issues before: {total_before}")
    print(f"Total issues after: {total_after}")
    print(f"Total improvement: {total_improvement} ({overall_pct:.1f}% reduction)")
    
    return results

if __name__ == "__main__":
    validate_corrected_files()