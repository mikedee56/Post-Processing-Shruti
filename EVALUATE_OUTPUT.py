#!/usr/bin/env python3
"""
Epic 2.4 Output Quality Evaluation & Tuning Tool
"""
import sys
from pathlib import Path
import json
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def analyze_batch_results():
    """Analyze the quality of your batch processing results"""
    
    print("=== Epic 2.4 Output Quality Analysis ===")
    print()
    
    raw_dir = Path("data/raw_srts")
    processed_dir = Path("data/processed_srts")
    
    # Find input and output files
    raw_files = list(raw_dir.glob("*.srt"))
    processed_files = list(processed_dir.glob("*_enhanced.srt"))
    
    print(f"Input files: {len(raw_files)}")
    print(f"Enhanced files: {len(processed_files)}")
    print(f"Success rate: {len(processed_files)}/{len(raw_files)} ({len(processed_files)/len(raw_files)*100:.1f}%)")
    print()
    
    # Quality metrics from processing
    metrics_files = list(processed_dir.glob("*_metrics.json"))
    if metrics_files:
        print("=== Processing Metrics Found ===")
        for metrics_file in metrics_files:
            with open(metrics_file, 'r') as f:
                data = json.load(f)
            print(f"Batch: {data.get('batch_id', 'Unknown')}")
            print(f"  Average confidence: {data.get('average_confidence', 'N/A'):.3f}")
            print(f"  Enhancement rate: {data.get('enhanced_segments', 0)}/{data.get('total_segments', 0)}")
            print()
    
    # Sample quality check
    print("=== Sample Quality Check ===")
    if processed_files:
        sample_file = processed_files[0]
        print(f"Checking: {sample_file.name}")
        
        with open(sample_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Basic quality indicators
        indicators = {
            "IAST transliteration": any(char in content for char in "āīūṛḷēōṃḥ"),
            "Sanskrit terms": any(term in content.lower() for term in ["yoga", "dharma", "karma", "atman", "gita"]),
            "Proper formatting": content.count("-->") > 0,
            "Clean text": "um," not in content.lower() and "uh," not in content.lower()
        }
        
        for indicator, present in indicators.items():
            status = "✓" if present else "✗"
            print(f"  {status} {indicator}: {'Found' if present else 'Not found'}")
        
        print()
    
    # Generate evaluation report
    evaluation = {
        "timestamp": datetime.now().isoformat(),
        "input_files": len(raw_files),
        "processed_files": len(processed_files),
        "success_rate": len(processed_files)/len(raw_files) if raw_files else 0,
        "quality_indicators": indicators if processed_files else {},
        "recommendations": generate_recommendations(raw_files, processed_files, indicators if processed_files else {})
    }
    
    # Save evaluation
    eval_file = processed_dir / f"quality_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(eval_file, 'w', encoding='utf-8') as f:
        json.dump(evaluation, f, indent=2, ensure_ascii=False)
    
    print("=== Recommendations ===")
    for rec in evaluation["recommendations"]:
        print(f"• {rec}")
    
    print()
    print(f"Full evaluation saved to: {eval_file}")
    
    return evaluation

def generate_recommendations(raw_files, processed_files, quality_indicators):
    """Generate specific recommendations based on results"""
    recommendations = []
    
    # Success rate recommendations
    success_rate = len(processed_files)/len(raw_files) if raw_files else 0
    if success_rate < 0.95:
        recommendations.append(f"SUCCESS RATE ({success_rate:.1%}) - Check failed files, may need encoding fixes")
    
    # Quality-based recommendations
    if not quality_indicators.get("IAST transliteration", False):
        recommendations.append("IAST TRANSLITERATION - No Sanskrit diacritics found. Check if Sanskrit terms are being processed")
    
    if not quality_indicators.get("Clean text", False):
        recommendations.append("FILLER REMOVAL - Still finding 'um/uh' words. May need stronger conversational cleanup")
    
    if not quality_indicators.get("Sanskrit terms", False):
        recommendations.append("SANSKRIT TERMS - No common Sanskrit terms found. Check if your content contains Sanskrit terminology")
    
    # General recommendations
    if len(processed_files) > 0:
        recommendations.append("SPOT CHECK - Manually review 2-3 files to verify quality meets your standards")
        recommendations.append("COMPARE - Use COMPARE_FILES.py to see before/after differences")
    
    if not recommendations:
        recommendations.append("QUALITY LOOKS GOOD - System appears to be working well!")
    
    return recommendations

def compare_files():
    """Compare original vs enhanced files"""
    print("=== Before vs After Comparison ===")
    
    raw_dir = Path("data/raw_srts")
    processed_dir = Path("data/processed_srts")
    
    # Find matching pairs
    raw_files = {f.stem: f for f in raw_dir.glob("*.srt")}
    
    pairs_found = 0
    for processed_file in processed_dir.glob("*_enhanced.srt"):
        # Find matching original
        original_name = processed_file.name.replace("_enhanced.srt", ".srt")
        original_file = raw_dir / original_name
        
        if original_file.exists():
            pairs_found += 1
            print(f"\nComparing: {original_name}")
            
            # Read both files
            with open(original_file, 'r', encoding='utf-8') as f:
                original = f.read()
            with open(processed_file, 'r', encoding='utf-8') as f:
                enhanced = f.read()
            
            # Basic comparison stats
            orig_lines = len(original.split('\n'))
            enhanced_lines = len(enhanced.split('\n'))
            
            print(f"  Original: {len(original)} chars, {orig_lines} lines")
            print(f"  Enhanced: {len(enhanced)} chars, {enhanced_lines} lines")
            
            # Look for specific improvements
            improvements = []
            if "ā" in enhanced or "ī" in enhanced or "ū" in enhanced:
                improvements.append("IAST transliteration added")
            if original.count(" um,") > enhanced.count(" um,"):
                improvements.append("Filler words removed")
            if original.count(" uh,") > enhanced.count(" uh,"):
                improvements.append("More filler words removed")
            
            if improvements:
                print(f"  Improvements: {', '.join(improvements)}")
            else:
                print(f"  Note: No obvious improvements detected (may be subtle)")
            
            if pairs_found >= 3:  # Limit output
                break
    
    if pairs_found == 0:
        print("No matching file pairs found for comparison")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "compare":
        compare_files()
    else:
        analyze_batch_results()