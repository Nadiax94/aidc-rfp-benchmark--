# RFP AI Extractor & Model Evaluation

## Overvie
RFP AI is a project focused on structured information extraction from Request for Proposal (RFP) documents.
It evaluates and compares different AI models using a consistent set of five RFP documents and 24 target fields


## Team
- Layan Alharbi
- Nadia Al-Mahyawi
- Shumua Althobaiti
- Renad Asiri

## Models
| Model | Type |
|---|---|
| Granite-4.1-8B-AWQ-INT4 | Open-weight |
| Qwen3.5-4B-AWQ-4bit | Open-weight |
| GPT-5.6 Luna | Commercial |

## Architecture
RFP Document → Text Processing → Chunking → AI Extraction → Validation → Structured JSON → Evaluation

## Benchmark
- 5 RFP documents
- 24 fields
- 360 model-field comparisons

## Evaluation Metrics
- Accuracy
- Completeness
- Hallucination Rate
- JSON Validity
-  Latency
- Token Usage


## User Interface
The interface allows users to upload RFP documents, extract the required information, and review model evaluation results.
(https://rfp-demo-kyieztneqv8xhhfdugxefh.streamlit.app/)


1.## RFP Extraction
<img width="626" height="449" alt="Screenshot 2026-09-27 133749" src="https://github.com/user-attachments/assets/a5d3d7de-0cf6-45e4-841f-ef8c10762f7f" />

## Evaluation Overview
<img width="599" height="428" alt="Screenshot 2026-09-27 133802" src="https://github.com/user-attachments/assets/4952e25c-c54e-40b6-853c-8cb0a149a33c" />
<img width="575" height="341" alt="Screenshot 2026-09-27 133823" src="https://github.com/user-attachments/assets/27c63fc8-6c1d-4cf8-a123-cff53e51db58" />

## Model Comparison
<img width="595" height="464" alt="Screenshot 2026-09-27 134104" src="https://github.com/user-attachments/assets/3292a219-eb46-4be5-828d-12786e83676d" />
<img width="593" height="361" alt="Screenshot 2026-09-27 134124" src="https://github.com/user-attachments/assets/1cc9f2a8-f854-41fa-9d9a-fe54d17d41a6" />


## Detailed RFP Analsis
<img width="519" height="325" alt="Screenshot 2026-09-27 134232" src="https://github.com/user-attachments/assets/9114b030-9ea0-4a32-a105-d71b66e6ad45" />

## Technology Stack
- Python
- vLLM
- Docker
- Kubernetes
- OpenAI API
