#!/usr/bin/env python3
"""
Simplified Batch Processor for Epic 2.4 - Windows 11 Compatible
"""
import sys
import time
from pathlib import Path
from datetime import datetime

# Add src to path  
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from post_processors.sanskrit_post_processor import SanskritPostProcessor

def process_all_srts():
    """Process all SRT files in data/raw_srts/"""
    
    input_dir = Path("data/raw_srts")
    output_dir = Path("data/processed_srts")
    
    # Find all SRT files
    srt_files = list(input_dir.glob("*.srt"))
    
    if not srt_files:
        print(f"No SRT files found in {input_dir}")
        print("Please place your .srt files in data/raw_srts/ and try again")
        return
    
    print(f"📥 Found {len(srt_files)} SRT files")
    print(f"📤 Output will be saved to {output_dir}")
    print("🚀 Starting Epic 2.4 processing...")
    print()
    
    # Initialize processor
    processor = SanskritPostProcessor()
    
    # Process files
    successful = 0
    total_segments = 0
    enhanced_segments = 0
    start_time = time.time()
    
    for i, srt_file in enumerate(srt_files, 1):
        try:
            print(f"Processing {i}/{len(srt_files)}: {srt_file.name}")
            
            # Create output filename
            output_file = output_dir / f"{srt_file.stem}_enhanced.srt"
            
            # Process file
            metrics = processor.process_srt_file(srt_file, output_file)
            
            # Accumulate stats
            successful += 1
            total_segments += metrics.total_segments
            enhanced_segments += metrics.segments_modified
            
            print(f"  ✓ {metrics.total_segments} segments, {metrics.segments_modified} enhanced")
            
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            continue
    
    # Final results
    end_time = time.time()
    processing_time = end_time - start_time
    
    print()
    print("🎉 Batch Processing Complete!")
    print(f"📊 Success Rate: {successful}/{len(srt_files)} ({successful/len(srt_files)*100:.1f}%)")
    print(f"📈 Enhanced Segments: {enhanced_segments:,}/{total_segments:,} ({enhanced_segments/total_segments*100:.1f}%)")
    print(f"⚡ Processing Time: {processing_time:.2f}s")
    print(f"📋 Enhanced files saved to: {output_dir}")
    
    print()
    print("✨ Epic 2.4 Enhancements Applied:")
    print("   🧠 Sanskrit/Hindi term correction")
    print("   📚 IAST transliteration")
    print("   🗣️  Conversational cleanup")
    print("   🔢 Number normalization")
    print("   📜 Scripture verse identification")

if __name__ == "__main__":
    process_all_srts()