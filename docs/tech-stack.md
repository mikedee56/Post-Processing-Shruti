# Technology Stack

This is the DEFINITIVE technology selection for the entire project. All development must use these exact versions.

## Cloud Infrastructure
* **Provider:** Local development initially, with a future plan for integration with Google Cloud.
* **Key Services:** Not applicable for the MVP.
* **Deployment Host and Regions:** Not applicable for the MVP.

## Technology Stack Table
| Category | Technology | Version | Purpose | Rationale |
|---|---|---|---|---|
| Language | Python | 3.10 | Core development language | Wide adoption in NLP, existing script is in Python, extensive library support for AI/ML. |
| Framework | Not Applicable | | Headless service | The MVP will be a script-based monolith, not a web application framework. |
| Data Processing | pandas | | Data manipulation | For efficient handling and transformation of transcript data. |
| NLP Libraries | iNLTK, IndicNLP Library | | Foundational NLP tasks | These libraries provide out-of-the-box support for Indic languages, which is essential for transliteration, normalization, and other foundational NLP tasks. |
| Specialized Model | ByT5-Sanskrit (optional) | | Advanced corrections | This pretrained model can be used for more complex tasks like phonetic matching and OCR post-correction, offering significant performance gains for Sanskrit. |
| Database | Not Applicable | | Data storage | The MVP will use a file-based approach with JSON/YAML lexicons. |
| Version Control | Git | | Code and document versioning | Standard for collaborative development. |