# Advanced ASR Post-Processing Workflow Product Requirements Document (PRD)

## Goals and Background Context

#### Goals
* To produce transcripts that are consistent, accurate, and suitable for book publication and video captions.
* To provide a reliable and easy-to-use searchable database for research and study.
* To transform 12,000 hours of lecture audio into an accessible and credible textual corpus.
* To improve standard transcription accuracy criteria, including WER and CER.
* All Sanskrit and Hindi terms will adhere to the IAST transliteration standard for consistency and academic rigor.

#### Background Context
The existing ASR-generated transcripts for Yoga Vedanta lectures, while achieving high initial accuracy in English, require a specialized post-processing workflow to address significant inconsistencies and misrecognitions. A key challenge is the precise identification and correction of Sanskrit and Hindi terms and scriptural references, which impacts the academic rigor and authenticity of the content. This project aims to solve this problem by developing a multi-phased post-processing solution that will transform raw SRT outputs into a high-quality, publishable resource.

#### Change Log
| Date | Version | Description | Author |
|---|---|---|---|
| August 5, 2025 | 1.0 | Initial draft based on Project Brief | John, PM |

## Requirements

#### Functional
1.  (FR1): The system must process ASR-generated SRT files to produce a post-processed transcript that adheres to standardized punctuation and capitalization.
2.  (FR2): The system must convert spoken numbers to digits (e.g., "two thousand five" to "2005").
3.  (FR3): The system must identify and correct common ASR errors, including filler words like "um" and "uh".
4.  (FR4): The system must leverage a domain-specific lexicon to identify and correctly transliterate Sanskrit and Hindi words, phrases, and verses.
5.  (FR5): The system must apply a strict transliteration standard (e.g., IAST) to all Sanskrit and Hindi terms for consistency and academic rigor.
6.  (FR6): The system must identify and replace scriptural verses (e.g., "Gita chapter 2, verse 25") with the canonical text from the lexicon.
7.  (FR7): The system must identify and correctly capitalize proper nouns specific to Yoga Vedanta (e.g., "Patanjali," "Himalayas").

#### Non Functional
1.  (NFR1): The system must maintain the integrity of the original SRT timestamps.
2.  (NFR2): The system must be able to handle a large volume of data efficiently (e.g., 12,000 hours of audio).
3.  (NFR3): The post-processing must preserve the original intention, tone, and stylistic nuances of the Guru's speech.
4.  (NFR4): The system must be scalable to handle future increases in lecture volume.
5.  (NFR5): The system must support the use of externalized lexicons (JSON or YAML files) for easy updates and versioning by linguistic experts.
6.  (NFR6): The post-processing script must be able to handle conversational nuances such as partial or rescinded phrases, ensuring grammatical correctness while preserving timestamp integrity.

## User Interface Design Goals

**This section is to guide future design efforts for the post-processing workflow, even if the MVP is not a graphical user interface.**

#### Overall UX Vision
The UX vision for this project is to provide a user experience that is transparent, authoritative, and efficient. The system will be headless for automated flagging, with an innovative UI for human review. This UI will be designed to enhance functionality for all users, with a structured but non-repetitive workflow that preserves the editor's sanity.

#### Key Interaction Paradigms
The primary interaction paradigm will be focused on feedback and validation. The system will present information in a clear, structured manner and provide a straightforward way for human reviewers to provide corrections and approval. The human review UI will have an editable, timestamped segment that, when clicked, seeks the audio to that specific spot and begins playback. A core feature will be a domain-specific correction tool that, for example, offers one-click replacement for fuzzy-matched Sanskrit terms, designed to be more efficient than a general-purpose tool like Grammarly. A key part of this workflow will be the ability for a GP editor to easily flag an issue for SME review, with a line and a comment/question, mimicking a Google Docs-style workflow.

#### Core Screens and Views
* **Correction Dashboard:** A conceptual view that shows a list of transcripts and their status (e.g., "Pending Review," "Flagged for SME," "Approved").
* **Transcript Review View:** A conceptual view that guides the editor through a structured workflow, highlighting flagged sections or words and allowing for easy navigation, correction, and contextual reference. A rating system will match the editor's expertise with the difficulty or complexity of the scripture involved in a particular transcript. The UI will provide a list of potential canonical verses for selection, but only from sources that provide a standardized presentation (IAST, etc.). We will avoid a free-form LLM lookup to maintain control and avoid complexity.

#### Accessibility: {None|WCAG AA|WCAG AAA|Custom Requirements}
* **Accessibility:** As a core principle, any future UI should be designed with accessibility in mind, with the understanding that good design will naturally lead to improved functionality.

#### Branding
* **Branding:** The UI should be clean, minimalist, and professional, to ensure the focus remains on the accuracy and content of the transcripts themselves.

#### Target Device and Platforms: {Web Responsive|Mobile Only|Desktop Only|Cross-Platform}
* **Target Platforms:** Any future UI should be web-responsive to support access from various devices for flexible review.

## Technical Assumptions

#### Repository Structure: {Monorepo|Polyrepo|Multi-repo}
* **Repository Structure:** Since this project involves a headless post-processing system with a potential future UI, a **Monorepo** structure with separate packages for the core logic, UI, and shared components would be a good fit. This would allow for easy code sharing and streamlined dependency management.

#### Service Architecture
* **Service Architecture:** The core of the MVP will be a **Monolith** service that houses all the post-processing logic, including the lexicon management, hybrid language identification, and QA metrics. This will simplify development and deployment for the MVP. We will design this monolith to be progressively complex, allowing for future refactoring into a microservice or serverless architecture if needed.

#### Testing Requirements
* **Testing Requirements:** To ensure the accuracy and reliability of the post-processing pipeline, we will require a **Full Testing Pyramid** approach, including unit tests for individual functions and integration tests for the full pipeline.
* **Additional Technical Assumptions and Requests**: Since you'll be using the Claude Code IDE, we'll design the system to be compatible with its specific capabilities. All post-processing will respect the integrity of the original SRT timestamps and the tone of the speech. A fallback will be in place for segments that span multiple words. A single, version-controlled configuration file will be the single source of truth for all post-processing parameters.

## Data Models

#### Transcript Segment
* **Purpose:** To represent an individual, timestamped segment of the transcript.
* **Key Attributes:**
    * `text`: `string` - The transcribed text of the segment.
    * `start_time`: `float` - The start time of the segment in seconds.
    * `end_time`: `float` - The end time of the segment in seconds.
    * `confidence_score`: `float` - The ASR confidence score for the segment.
    * `is_flagged`: `boolean` - A flag to indicate if the segment requires human review.
    * `flag_reason`: `string` - The reason the segment was flagged (e.g., `low_confidence`, `high_oov`).
    * `correction_history`: `array` - A log of all human corrections made to the segment.

#### Lexicon Entry
* **Purpose:** To represent a single entry in the domain-specific lexicon.
* **Key Attributes:**
    * `original_term`: `string` - The original term (e.g., "Dharma").
    * `variations`: `array` - A list of common misrecognized variations (e.g., "Therma", "Drama").
    * `transliteration`: `string` - The correct IAST transliteration.
    * `is_proper_noun`: `boolean` - A flag to indicate if the term should be capitalized.
    * `is_verse`: `boolean` - A flag to indicate if the term is part of a scriptural verse.
    * `canonical_text`: `string` - The full canonical text for a scriptural verse.

## Epic List

* **Epic 1: Foundation & Pre-processing Pipeline:** Establish the core processing framework, externalize lexicons, and implement foundational pre-processing corrections for punctuation, numbers, and filler words. This will ensure a clean, consistent input for later stages of the workflow.
* **Epic 2: Sanskrit & Hindi Identification & Correction:** Develop and integrate hybrid language modeling, implement the IAST transliteration standard, and enable verse and scripture identification.
* **Epic 3: Semantic Refinement & QA Framework:** Implement advanced semantic refinements, including Named Entity Recognition, and establish the robust, multi-tiered automated and human review system.
* **Epic 4: Deployment & Scalability:** Design the system for robust batch processing, implement version control, and set up benchmarking for continuous evaluation.

## Epic Details

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

**Epic Goal**: Develop and integrate hybrid language modeling, implement the IAST transliteration standard, and enable verse and scripture identification to ensure the academic integrity of the transcripts.

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
3.  The system provides a feedback loop to incorporate human corrections from the review process back into the post-processing script and external lexicons.

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

## Next Steps

After completing the architecture, the next logical steps are to:

1.  **Review with Product Owner:** The Product Owner will review and validate the architecture against the project's goals and requirements.
2.  **Begin story implementation with Dev agent:** With the architecture in place, the development team can begin implementing the stories.
3.  **Set up infrastructure:** The DevOps agent can set up the necessary infrastructure for the project.

#### Developer Handoff

The following prompt can be used to hand off the project to the development team:

> "I have a new project for you. All the project documentation, including the PRD and architecture, are in the `docs/` folder. The project is a post-processing workflow for ASR-generated transcripts. The core logic is in Python, and the project uses a monorepo architecture. The first epic is focused on setting up the project and implementing the foundational post-processing corrections. Please review the documentation and begin with the first story."