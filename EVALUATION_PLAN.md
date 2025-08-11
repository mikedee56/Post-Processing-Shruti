# Epic 2.4 Output Evaluation & Tuning Plan

## 🎯 **Your 3-Step Evaluation Process**

### **STEP 1: Automated Quality Check**
```bash
py -3.10 EVALUATE_OUTPUT.py
```
**This gives you:**
- Success rate (target: >95%)
- Quality indicators (IAST, Sanskrit terms, cleanup)
- Specific recommendations
- JSON report with metrics

### **STEP 2: Manual Comparison**
```bash
py -3.10 COMPARE_FILES.py
```
**This shows you:**
- Side-by-side before/after for 3 sample files
- Specific improvements made
- Quality overview of all files

### **STEP 3: System Tuning** (if needed)
```bash
py -3.10 TUNE_SYSTEM.py
```
**This helps with:**
- Configuration review
- Custom term additions
- Performance adjustments
- Improvement suggestions

---

## 📊 **Quality Benchmarks**

### **Epic 2.4 Target Metrics**
| Metric | Target | Meaning |
|--------|---------|---------|
| Success Rate | >95% | Files processed without errors |
| Enhancement Rate | >75% | Segments improved |
| Average Confidence | >0.8 | Quality score (0.0-1.0) |
| IAST Compliance | Yes | Sanskrit diacritics present |
| Filler Cleanup | Yes | "um/uh" removed |

### **Red Flags to Watch For**
- ❌ Success rate <90% = File format/encoding issues
- ❌ No IAST characters = Sanskrit processing not working  
- ❌ Still seeing "um, uh" = Conversational cleanup failed
- ❌ Low confidence <0.6 = System unsure about changes

---

## 🔧 **Common Tuning Scenarios**

### **Scenario 1: "System too conservative"**
**Problem:** Few enhancements, high confidence
**Solution:** Lower confidence thresholds in lexicons

### **Scenario 2: "Missing specific Sanskrit terms"**
**Problem:** Your teacher uses terms not in default lexicon
**Solution:** Add custom terms via `TUNE_SYSTEM.py add`

### **Scenario 3: "Wrong capitalizations"**  
**Problem:** Names/places not capitalized correctly
**Solution:** Update `data/lexicons/proper_nouns.yaml`

### **Scenario 4: "Still has filler words"**
**Problem:** "um, uh" not being removed
**Solution:** Check `data/lexicons/phrases.yaml` patterns

---

## 📋 **Evaluation Workflow**

### **After Each Batch:**
1. **Run evaluation**: `py -3.10 EVALUATE_OUTPUT.py`
2. **Check 2-3 files manually** for quality
3. **Compare before/after**: `py -3.10 COMPARE_FILES.py`
4. **Tune if needed**: `py -3.10 TUNE_SYSTEM.py improve`

### **Weekly/Monthly Review:**
1. **Aggregate metrics** across batches
2. **Update lexicons** with new terms found
3. **Performance optimization** if processing large volumes
4. **Backup configurations** that work well

---

## 🎓 **Understanding Your Content**

### **Yoga/Vedanta Content Quality Indicators:**
- ✅ **Sanskrit Terms**: dharma, karma, yoga, ātman, brahman
- ✅ **Proper Nouns**: Krishna, Arjuna, Patanjali, Sankara  
- ✅ **Scriptures**: Bhagavad Gita, Upanishads, Yoga Sutras
- ✅ **IAST**: ā, ī, ū, ṛ, ḷ, ṃ, ḥ (academic standard)
- ✅ **Numbers**: "chapter 2 verse 47" not "chapter two verse forty seven"

### **Content-Specific Tuning:**
- **Different teacher/tradition?** → Add their specific terminology
- **Regional pronunciations?** → Add variation patterns  
- **Technical terms?** → Update transliteration preferences
- **Lecture style?** → Adjust conversational cleanup strength

---

## 🚀 **Next Steps After Your Batch Completes**

1. **Wait for batch to finish**
2. **Run**: `py -3.10 EVALUATE_OUTPUT.py`
3. **Check results** in `data/processed_srts/`
4. **Review recommendations** from evaluation
5. **Make adjustments** if needed
6. **Re-run small test** to verify improvements

**The goal:** Get Epic 2.4 perfectly tuned to your specific content and quality standards!