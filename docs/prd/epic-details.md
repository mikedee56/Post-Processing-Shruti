# Epic Details

### Epic 1: Foundation & Pre-processing Pipeline

**Epic Goal**: Establish the core processing framework, externalize lexicons, and implement foundational pre-processing corrections for punctuation, numbers, and filler words. This will ensure a clean, consistent input for later stages of the workflow.

#### Story 1.1: File Naming Conventions & Storage Management
**As a** developer,
**I want** to set up a clear and scalable file structure,
**so that** all raw and processed data can be easily organized and managed.

**Acceptance Criteria**
1.  A directory structure is created for raw SRTs, processed SRTs, lexicons, and a golden dataset.
2.  A consistent file naming convention is defined and documented for all SRT files.
3.  The system can ingest and process files from this directory structure.
4.  All outputs are stored in the appropriate `processed_srts/` directory, mirroring the original file structure.

#### Story 1.2: Project Scaffolding & Core Setup
**As a** developer,
**I want** to set up the project structure and core dependencies,
**so that** I can begin building the post-processing workflow.

**Acceptance Criteria**
1.  The project repository is set up with a `docs/` folder for documentation and a `src/` folder for the application code.
2.  A Python-based application environment is configured with a `requirements.txt` file to manage dependencies.
3.  The `sanskrit_post_processor.py` script is integrated into the new project structure.
4.  A placeholder `config/` directory is created for future configuration files.

#### Story 1.3: Lexicon Externalization
**As a** linguistic expert,
**I want** to manage Sanskrit and Hindi terms in external files,
**so that** I can easily update and version the lexicons without changing the core script.

**Acceptance Criteria**
1.  A system is in place to read correction data from external JSON or YAML files located in `data/lexicons/`.
2.  The existing `self.corrections`, `self.proper_nouns`, and `self.phrases` dictionaries in `sanskrit_post_processor.py` are refactored to be loaded from these external files.
3.  The system can handle a variety of lexicon types (e.g., proper nouns, phrases, verses) from separate files.
4.  The core script continues to function correctly, using the data from the new external files.

#### Story 1.4: Foundational Post-processing Corrections
**As a** human reviewer,
**I want** the transcripts to have basic errors corrected automatically,
**so that** I can focus my efforts on more complex issues.

**Acceptance Criteria**
1.  The script successfully normalizes punctuation, including consistent spacing after periods.
2.  The script accurately converts spoken numbers into their digital format (e.g., "two thousand five" to "2005").
3.  The script identifies and removes common English filler words (e.g., "um," "uh") from the transcript.
4.  All corrections made by this process maintain the integrity of the original SRT timestamps.
5.  The script handles and corrects conversational nuances such as partial or rescinded phrases, ensuring grammatical correctness where possible.

### Epic 2: Sanskrit & Hindi Identification & Correction
2

#### Story 2.1: Lexicon-Based Correction System
**As a** post-processing script,
**I want** to identify and replace misrecognized Sanskrit and Hindi words,
**so that** the transcript is more accurate and readable.

**Acceptance Criteria**
1.  The system can identify words not found in a standard English dictionary but present in the externalized Sanskrit/Hindi lexicon.
2.  The system can use fuzzy matching (e.g., Levenshtein distance) to suggest corrections from the lexicon for near misses.
3.  The system can apply high-confidence fuzzy matches to replace the ASR output with the correct transliterated spelling.
4.  The system can enforce the IAST transliteration standard for all Sanskrit and Hindi terms in the transcript.

#### Story 2.2: Hybrid Language & Contextual Modeling
**As a** post-processing script,
**I want** to use contextual clues to improve correction accuracy,
**so that** I can handle complex linguistic nuances.

**Acceptance Criteria**
1.  The system can use n-gram models or rule-based systems to predict the likelihood of a Sanskrit/Hindi word appearing in a given context.
2.  The system can correct misrecognized terms by generating phonetic representations of the ASR output and comparing them to phonetic representations of the Sanskrit/Hindi lexicon.
3.  The system can apply contextual rules to ensure consistent and correct terminology (e.g., if "karma" is followed by "yoga," ensure it's "karma yoga").
4.  The system can handle context-dependent spelling and shortened words, falling back to a more formal, standardized spelling for consistency.

#### Story 2.3: Verse and Scripture Substitution
**As a** post-processing script,
**I want** to identify and replace scriptural verses with the canonical text,
**so that** the transcript is academically sound and authoritative.

**Acceptance Criteria**
1.  The system can identify longer Sanskrit/Hindi passages that correspond to known scriptural verses in the lexicon.
2.  The system can replace the transcribed passage with the canonical text for that verse.
3.  The verse and scripture identification is based on a standardized presentation (e.g., IAST) from the lexicon.
4.  The system can provide a list of potential canonical verses for selection from standardized sources in the lexicon.

### Epic 3: Semantic Refinement & QA Framework

**Epic Goal**: Implement advanced semantic refinements, including Named Entity Recognition, and establish the robust, multi-tiered automated and human review system.

#### Story 3.1: Yoga Vedanta Named Entity Recognition (NER)
**As a** post-processing script,
**I want** to identify and correctly capitalize proper nouns,
**so that** the transcript is more readable and accurate.

**Acceptance Criteria**
1.  The system can train or adapt an NER model to identify proper nouns specific to Yoga Vedanta (e.g., "Patanjali," "Himalayas").
2.  The system uses the externalized lexicon to correctly capitalize these proper nouns in the transcript.
3.  The NER model can be expanded to include new proper nouns as needed.

#### Story 3.2: Automated Quality Assurance (QA) Flagging
**As a** headless system,
**I want** to automatically flag sections that are likely to have errors,
**so that** human reviewers can focus their efforts on the most critical content.

**Acceptance Criteria**
1.  The system can flag sections with unusually low ASR confidence scores.
2.  The system can flag sections with a high number of Out-Of-Vocabulary (OOV) words not found in the combined lexicon.
3.  The system can identify and flag sudden shifts in language or acoustic properties that might indicate an ASR error.
4.  The system can generate a report that highlights potential errors identified by fuzzy matches to prioritize sections for human review.

#### Story 3.3: Tiered Human Review Workflow
**As a** human reviewer,
**I want** to be guided through a clear and efficient review process,
**so that** I can provide accurate corrections based on my expertise.

**Acceptance Criteria**
1.  The system provides a mechanism for a general proofreader (GP) to easily flag issues for a Subject Matter Expert (SME) review, with a line and a comment/question, in a Google Docs-style workflow.
2.  The system implements a rating system that matches the editor's level of expertise with the difficulty or complexity of the scripture involved in a particular transcript.
3.  The system provides a feedback loop to incorporate human corrections from the review process back into tche post-processing script and external lexicons.

### Epic 4: Deployment & Scalability

**Epic Goal**: Design the system for robust batch processing, implement version control, and set up benchmarking for continuous evaluation.

#### Story 4.1: Batch Processing Framework
**As a** system administrator,
**I want** to process large volumes of transcripts efficiently,
**so that** I can handle the full corpus of 12,000 hours of lecture content.

**Acceptance Criteria**
1.  The post-processing script is designed to operate within a robust batch processing framework (e.g., Apache Airflow, custom Python scripts with multiprocessing).
2.  The framework includes robust error handling, logging, progress monitoring, and restart capabilities.
3.  The system can scale to handle the 12,000 hours of audio efficiently.

#### Story 4.2: Version Control & Documentation
**As a** project lead,
**I want** to maintain a clear and organized project,
**so that** I can ensure long-term maintainability and collaboration.

**Acceptance Criteria**
1.  Strict version control is implemented for the script, externalized lexicons, and models.
2.  The methodology, parameters, and design choices of the project are thoroughly documented.
3.  The `docs/` folder contains all the necessary project artifacts, including the PRD, architecture, and epic details.

#### Story 4.3: Benchmarking & Continuous Improvement
**As a** project owner,
**I want** to measure the success of the project and ensure continuous improvement,
**so that** I can provide concrete evidence of the enhancements' effectiveness.

**Acceptance Criteria**
1.  A "golden dataset" of manually perfected transcripts is used to quantitatively measure the script's accuracy improvements (e.g., Word Error Rate reduction on Sanskrit terms).
2.  A systematic process is established to incorporate human corrections back into the post-processing script.
3.  The system can utilize these corrections to fine-tune linguistic models and expand externalized lexicons.
4.  The system provides a mechanism to track and report on automated QA metrics and editor-specific corrections.

---

