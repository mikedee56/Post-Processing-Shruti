# Project Brief: Advanced ASR Post-Processing Workflow

### Executive Summary

This project aims to develop an advanced, integrated post-processing workflow for ASR-generated transcripts of Yoga Vedanta lectures, transforming raw SRT outputs into highly accurate, semantically refined, and consistently formatted textual data. The primary problem addressed is the current lack of precision in ASR outputs for specialized domain content, particularly concerning Sanskrit and Hindi terms, which hinders readability and academic rigor for its target audience of students and scholars. By leveraging a multi-phase approach encompassing pre-computation, hybrid language modeling, semantic refinement, and tiered human validation, this solution offers a robust, scalable, and academically rigorous method for enhancing transcript quality, ensuring authenticity and improved accessibility of invaluable lecture content.

### Problem Statement

The current state of ASR-generated transcripts for Yoga Vedanta lectures, while achieving high initial accuracy for English content, consistently falls short in critical areas, leading to significant pain points for students, scholars, and practitioners. Specifically, the existing `sanskrit_post_processor.py` script requires enhancement to address fundamental inconsistencies in punctuation, numeric representations, and the presence of filler words, which detract from readability and professional presentation. More critically, a lack of robust, domain-specific identification and correction mechanisms for Sanskrit and Hindi terms results in frequent misrecognitions, incorrect transliterations, and the absence of canonical scripture identification. These inaccuracies directly impact the academic rigor and authenticity of the content, making it challenging for users to reliably study and reference the profound philosophical teachings. Existing generic ASR solutions and preliminary post-processing scripts are inadequate as they lack the specialized linguistic and contextual understanding necessary to consistently transform these transcripts into a high-quality, academically sound resource. The urgency of solving this now stems from the immense value of the 12,000 hours of lecture content, which, without precise post-processing, remains less accessible and less credible for its intended audience over a 50-year period of diverse lectures.

### Proposed Solution

This project will establish an **Advanced ASR Post-Processing Workflow** designed to systematically enhance the accuracy, consistency, and academic integrity of automatically generated transcripts for Yoga Vedanta lectures. The core concept involves a multi-phased, integrated approach that moves beyond generic error correction to specialized linguistic and semantic refinement. Key differentiators include the development of externalized, domain-specific Sanskrit/Hindi lexicons, the implementation of hybrid language modeling for nuanced identification and correction of non-English terms, advanced Named Entity Recognition tailored for Yoga Vedanta concepts, and a sophisticated tiered human review and feedback loop system. This solution is poised to succeed where others have fallen short due to its unparalleled focus on the unique linguistic and contextual challenges of Yoga Vedanta content, ensuring precise transliteration standards, contextual accuracy, and continuous improvement through an iterative human-in-the-loop validation process. The post-processing workflow must also maintain the integrity of SRT timestamps and the intention, tone, and nuanced style of Guru Sri Swami Jyotirmayananda's speech.

### Target Users

The intended users of this advanced post-processing workflow are primarily students, scholars, and dedicated practitioners of Yoga Vedanta who require accurate and reliable textual resources.

#### Primary User Segment: Students & Scholars

* **Needs & Pain Points:** Their biggest concern is the integrity and accuracy of Sanskrit and Hindi terms within the transcripts. They need to be able to trust that a transcribed term matches the actual scriptural verse, phrase, or term. This includes correctly identifying Hindi proverbs and sayings.
* **Goals:** They seek a high-quality, searchable, and authoritative database of lecture content for academic research and deep study. A key goal is to quickly and reliably look up specific references or study topics within the vast corpus of lectures.
* **Behaviors & Workflows:** They will primarily use the transcripts as a searchable database, but also for video captions and as the main source of material for book publication.

#### Secondary User Segment: General Practitioners & Seekers

* **Needs & Pain Points:** This segment requires highly readable and consistent transcripts for general study and understanding. Punctuation errors, inconsistent spelling, and misrecognized English words detract from their reading experience.
* **Goals:** They want a clean, easy-to-read textual companion to the audio lectures that accurately reflects the teachings without linguistic distractions.
* **Behaviors & Workflows:** They are most likely to use the transcripts for video captions and personal study, valuing readability and flow.

### Goals & Success Metrics

The following objectives and key performance indicators (KPIs) will define and measure the success of the Advanced ASR Post-Processing Workflow.

#### Business Objectives
* **Create a publishable-quality resource:** To produce transcripts that are consistent, accurate, and suitable for book publication and video captions.
* **Enhance user engagement:** To provide students, scholars, and practitioners with a reliable and easy-to-use searchable database for research and study.
* **Unlock content value:** To transform 12,000 hours of lecture audio into an accessible and credible textual corpus.

#### User Success Metrics
* **Transcription Accuracy Improvement:** Quantitatively measure the post-processing script's effectiveness by improving standard transcription accuracy criteria, including a reduction in Word Error Rate (WER) and Character Error Rate (CER) on a "golden dataset" of manually perfected transcripts.
* **Increased Searchability:** The final product should provide a fast and efficient way for users to look up specific references or study topics, potentially by presenting a list of choices for possible references of interest from the corpus.
* **IAST Transliteration Standard:** All Sanskrit and Hindi terms will adhere to the IAST transliteration standard for consistency and academic rigor.

#### Key Performance Indicators (KPIs)
* **WER/CER Reduction:** Achieve a measurable reduction in Word Error Rate and Character Error Rate on the golden dataset, with a specific focus on Sanskrit and Hindi terms.
* **Consistency Score:** Develop a custom metric to quantify consistency across punctuation, numeric normalization, transliteration, and proper noun capitalization.
* **Manual Correction Rate:** Track a reduction in the rate of manual corrections required per hour of audio post-processing.
* **Corpus Detection Rate:** Monitor the percentage of detected Sanskrit/Hindi terms, scriptural verses, and proper nouns to ensure the system is working as expected.

### MVP Scope

This MVP will focus on building the core post-processing workflow to handle a batch of ASR-generated SRT files, ensuring consistency and accuracy in a domain-specific context.

#### Core Features (Must Have)
* **Automated Error Correction Pipeline:** A script or framework that processes raw SRT files to enforce standardized punctuation, numeric normalization, and filler word removal.
* **Externalized Lexicon Management:** A system to manage external JSON or YAML files for Sanskrit and Hindi terms, phrases, and proper nouns.
* **Hybrid Language Identification & Correction:** A mechanism to identify and correctly replace misrecognized Sanskrit/Hindi terms using dictionary lookups, phonetic matching, and contextual analysis. The system will enforce IAST transliteration as the standard for all Sanskrit and Hindi terms.
* **Verse and Scripture Identification:** The system will be able to recognize particular verses from scriptures (e.g., "Gita chapter 2, verse 25") and substitute the correct scriptural reference and canonical text from the lexicon.
* **Automated Quality Assurance (QA) Metrics:** Development of automated metrics to flag problematic sections based on low ASR confidence scores, high out-of-vocabulary (OOV) rates, and other anomalies for prioritized human review.
* **Tiered Human Review System:** A defined workflow for human review, including a Tier 1 for SMEs focused on critical error correction and a Tier 2 for general proofreading.
* **Feedback Loop for Script Improvement:** A process to systematically incorporate human corrections back into the post-processing script and external lexicons.

#### Out of Scope for MVP
* Real-time processing or streaming integration. The MVP will focus on a local batch processing framework.
* A fully developed user interface for human review. The MVP will rely on a defined workflow using standard files and annotation platforms.
* A completely new ASR engine. The project assumes an existing ASR output is being provided as input.
* Advanced discourse coherence checks that require extensive grammatical parsing.
* Automated speaker diarization refinement (if not already handled by the ASR engine).