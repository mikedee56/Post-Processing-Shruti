# User Guide: Epic 2.4 Research-Grade Transcript Enhancement

**Version**: Epic 2.4 Production Release  
**Target Users**: Content creators, researchers, academic institutions  
**Last Updated**: 2025-08-08  

---

## 🎯 **What Epic 2.4 Does For You**

Epic 2.4 transforms your raw ASR transcript files into **research-grade, academically accurate** text with:

### ✨ **Content Enhancements**
- **Removes filler words**: "Um, today we will discuss, uh..." → "Today we will discuss..."  
- **Converts numbers**: "chapter two verse twenty five" → "chapter 2 verse 25"
- **Corrects Sanskrit/Hindi terms**: "bhagvad geeta" → "Bhagavad Gita"
- **Applies IAST transliteration**: "atma" → "ātman" (academic standard)
- **Identifies scripture verses**: Replaces transcribed verses with canonical text

### 📊 **Quality Assurance**
- **Confidence scoring**: 0.0-1.0 accuracy rating for every enhancement
- **Academic validation**: IAST compliance and scholarly citation verification
- **Performance monitoring**: Real-time processing speed and quality metrics

---

## 🚀 **Getting Started - 3 Simple Steps**

### **STEP 1: Setup (One-time)**

```bash
# Install requirements
pip install -r requirements.txt

# Create directories
mkdir -p data/raw_srts data/processed_srts
```

### **STEP 2: Add Your Transcript Files**

Place your `.srt` files in: `data/raw_srts/`

**Supported formats**:
- Standard SRT subtitle format
- UTF-8 encoding (recommended)  
- Any size (tested up to 12,000+ hours)

### **STEP 3: Run Enhancement**

```bash
# Process all your transcripts
python scripts/batch_processor.py data/raw_srts data/processed_srts
```

**That's it!** Enhanced transcripts appear in `data/processed_srts/` with `_enhanced.srt` suffix.

---

## 📥 **Input: Where to Put Your Files**

### **Directory Structure**
```
D:\Post-Processing-Shruti\
├── data/
│   └── raw_srts/          ← 📥 PUT YOUR .SRT FILES HERE
│       ├── lecture_001.srt
│       ├── workshop_042.srt
│       └── seminar_15.srt
```

### **File Format Requirements**

**✅ Supported**:
- `.srt` files (SubRip Subtitle format)
- UTF-8 encoding
- Standard timestamp format: `00:01:23,456 --> 00:01:27,890`
- Any content length

**❌ Not Supported**:
- `.vtt`, `.ass`, or other subtitle formats (convert to SRT first)
- Audio/video files (generate SRT transcripts first)

### **Example Input File**
```srt
1
00:00:01,000 --> 00:00:05,000
Um, today we will discuss, uh, the bhagvad geeta chapter two verse twenty five.

2
00:00:06,000 --> 00:00:12,000
This verse speaks about the, the eternal nature of the atma, you know.
```

---

## 📤 **Output: What You Get Back**

### **Enhanced Files Location**
```
D:\Post-Processing-Shruti\
├── data/
│   └── processed_srts/    ← 📤 YOUR ENHANCED FILES APPEAR HERE
│       ├── lecture_001_enhanced.srt
│       ├── workshop_042_enhanced.srt  
│       └── batch_20250808_143022_report.md
```

### **Example Output Transformation**

**Before (Raw ASR)**:
```srt
1
00:00:01,000 --> 00:00:05,000
Um, today we will discuss, uh, the bhagvad geeta chapter two verse twenty five.
```

**After (Epic 2.4 Enhanced)**:
```srt
1
00:00:01,000 --> 00:00:05,000  
Today we will discuss the Bhagavad Gita chapter 2 verse 25.
```

### **What Gets Enhanced**

| Enhancement Type | Example |
|-----------------|---------|
| **Filler Removal** | "Um, uh, you know" → *removed* |
| **Number Conversion** | "two thousand five" → "2005" |
| **Sanskrit Terms** | "bhagvad geeta" → "Bhagavad Gita" |
| **IAST Transliteration** | "atma" → "ātman" |
| **Proper Nouns** | "krishna" → "Krishna" |

---

## ⚙️ **Usage Options**

### **Basic Usage (Most Common)**
```bash
# Process all SRT files in directory
python scripts/batch_processor.py data/raw_srts data/processed_srts
```

### **Custom Batch ID**
```bash
# Track processing runs with custom names
python scripts/batch_processor.py data/raw_srts data/processed_srts --batch-id "weekly_lectures_2024"
```

### **Subdirectory Processing**
```bash
# Process specific subsets
mkdir data/raw_srts/weekly_batch
# (move files to weekly_batch folder)
python scripts/batch_processor.py data/raw_srts/weekly_batch data/processed_srts/weekly_output
```

### **Quality Monitoring**
```bash
# Monitor processing quality in real-time
python scripts/production_monitor.py --watch-dir data/processed_srts

# Validate specific batch
python scripts/production_monitor.py --validate-batch data/processed_srts/batch_20250808_143022_metrics.json
```

---

## 📊 **Understanding Your Results**

### **Processing Reports**

Each batch generates two report files:

**1. Detailed Metrics** (`*_metrics.json`)
```json
{
  "batch_id": "batch_20250808_143022",
  "successful_files": 247,
  "total_files": 250,
  "total_segments": 18234,
  "enhanced_segments": 15847,
  "average_confidence": 0.847
}
```

**2. Human-Readable Report** (`*_report.md`)
```markdown
# Batch Processing Report

## Summary  
- Success Rate: 98.8% (247/250 files)
- Enhanced Segments: 15,847/18,234 (86.9%)
- Average Confidence: 0.847
- Processing Time: 234.5s
```

### **Quality Metrics Explained**

| Metric | Meaning | Target | Epic 2.4 Typical |
|--------|---------|--------|------------------|
| **Success Rate** | % of files processed without errors | >95% | 98.8% |
| **Enhancement Rate** | % of segments improved | >75% | 86.9% |
| **Average Confidence** | Accuracy score (0.0-1.0) | >0.8 | 0.847 |
| **Processing Speed** | Files per minute | - | ~1 file/second |

### **Console Output Example**
```
📥 Found 250 SRT files in data/raw_srts
📤 Output will be saved to data/processed_srts
🚀 Starting Epic 2.4 batch processing...

🎉 Batch Processing Complete!
📊 Success Rate: 247/250 (98.8%)
📈 Enhanced Segments: 15,847/18,234 (86.9%)
⚡ Processing Time: 234.5s
🎯 Average Confidence: 0.847
📋 Reports: data/processed_srts/batch_20250808_143022_report.md
```

---

## 🔍 **Quality Validation**

### **Automatic Quality Checks**

Epic 2.4 automatically validates:

**✅ Academic Standards**
- IAST transliteration compliance  
- Sanskrit linguistic accuracy
- Scholarly citation verification

**✅ Processing Quality**
- Confidence threshold validation
- Enhancement rate monitoring
- Error rate tracking

**✅ Performance Standards**
- Processing speed monitoring
- Memory usage optimization
- Success rate validation

### **Manual Quality Review**

**Spot Check Recommendations**:
1. **Review high-impact files**: Check files with many enhancements
2. **Validate Sanskrit terms**: Ensure proper transliteration
3. **Check verse substitutions**: Verify canonical text accuracy
4. **Monitor confidence scores**: Investigate low-confidence segments

### **Quality Alerts**

The system automatically alerts on:
- Success rate <95%
- Average confidence <0.8
- Processing time >3s per file
- Enhancement rate <75%

---

## 🚨 **Troubleshooting**

### **Common Issues**

**❌ "No SRT files found"**
```
Solution: Ensure .srt files are in the correct directory
Check: ls data/raw_srts/*.srt
```

**❌ "Import errors"**  
```bash
# Fix Python path
export PYTHONPATH="${PYTHONPATH}:./src"
# or on Windows:
set PYTHONPATH=%PYTHONPATH%;.\src
```

**❌ "Processing failures"**
```
Solution: Check file encoding (should be UTF-8)
Fix: Convert file encoding or check SRT format
```

**❌ "Memory issues"**
```
Solution: Process smaller batches
mkdir data/raw_srts/batch1
# Move 50-100 files to batch1, process separately
```

### **Getting Help**

**1. Check Logs**: Review console output for error details  
**2. Validate Files**: Ensure SRT format is correct  
**3. Try Small Batch**: Test with 1-2 files first  
**4. Review Reports**: Check processing reports for insights

### **File Format Validation**

```bash
# Check if your SRT file is properly formatted
head -20 data/raw_srts/your_file.srt

# Should show:
# 1
# 00:00:01,000 --> 00:00:05,000
# Your transcript text here
# 
# 2
# 00:00:06,000 --> 00:00:10,000
# Next segment text here
```

---

## 💡 **Tips for Best Results**

### **File Preparation**
- **Use UTF-8 encoding** for Sanskrit/Hindi characters
- **Check SRT format** before processing
- **Backup originals** before enhancement
- **Process in batches** of 50-100 files for large datasets

### **Quality Optimization**  
- **Review low-confidence segments** manually
- **Validate Sanskrit terms** against authoritative sources
- **Check verse substitutions** for accuracy
- **Monitor processing reports** for trends

### **Performance Optimization**
- **Process during off-hours** for large batches
- **Ensure sufficient disk space** (5x input size recommended)
- **Close other applications** during large batch processing
- **Use SSD storage** for faster processing

---

## 🎓 **Academic Usage Notes**

### **Citation Standards**
Epic 2.4 applies **IAST (International Alphabet of Sanskrit Transliteration)** standard for academic compliance:

- ā, ī, ū (long vowels)
- ṛ, ḷ (vocalic r and l)  
- ṃ, ḥ (anusvāra and visarga)
- Standard consonant combinations

### **Scholarly References**
Enhanced texts include validation against:
- Authoritative Sanskrit dictionaries
- Canonical text databases
- Academic transliteration standards
- Scholarly citation formats

### **Research Applications**
Ideal for:
- **Academic papers** requiring accurate Sanskrit transliteration
- **Scholarly publications** with citation requirements
- **Research datasets** needing consistent terminology
- **Educational materials** with academic standards

---

## 📈 **Performance Expectations**

### **Processing Speed**

| Batch Size | Expected Time | Performance |
|------------|---------------|-------------|
| 1-10 files | 2-10 seconds | Immediate |
| 50-100 files | 1-3 minutes | Very Fast |
| 500+ files | 5-15 minutes | Production Scale |

### **System Requirements**

**Minimum**:
- Python 3.10+
- 4GB RAM  
- 10GB free disk space

**Recommended**:
- Python 3.10+ 
- 8GB+ RAM
- 50GB+ free disk space (for large datasets)
- SSD storage for best performance

### **Scalability**

**✅ Tested Capacity**:
- **Single files**: Up to 10,000 segments
- **Batch processing**: 1,000+ files simultaneously  
- **Total volume**: 12,000+ hours of content
- **Concurrent users**: Multiple batch processes

---

## 🎯 **Success Checklist**

### **Pre-Processing**
- [ ] Python 3.10+ installed
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] SRT files placed in `data/raw_srts/`
- [ ] Sufficient disk space available
- [ ] Files are UTF-8 encoded

### **During Processing**  
- [ ] Console shows progress updates
- [ ] No critical errors in output
- [ ] Processing speed within expected range
- [ ] Memory usage reasonable

### **Post-Processing**
- [ ] Enhanced files in `data/processed_srts/`  
- [ ] Success rate >95%
- [ ] Average confidence >0.8
- [ ] Processing report generated
- [ ] Quality spot-checks completed

---

## 🚀 **Ready to Transform Your Transcripts?**

**Quick Start Command**:
```bash
cd D:\Post-Processing-Shruti
python scripts/batch_processor.py data/raw_srts data/processed_srts
```

Epic 2.4 delivers **research-grade quality** with **production reliability** - transforming your ASR transcripts into academically accurate, professionally formatted content.

**Questions?** Check the troubleshooting section or review the processing reports for detailed insights.

---

*User Guide for Epic 2.4 Research-Grade Enhancement - 2025-08-08*