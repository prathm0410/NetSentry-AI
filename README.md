# NetSentry AI

**An ML-powered Network Intrusion Detection and Security Data Engineering Platform.**

NetSentry AI is a portfolio project that aims to build an end-to-end network intrusion detection pipeline using data engineering, machine learning, and cybersecurity technologies.

## Project Goals

- Capture and analyze network traffic.
- Build reliable network data ingestion pipelines.
- Process and store network connection records.
- Train machine learning models to identify suspicious network activity.
- Provide explainable detection results through a dashboard.

## Technology Stack

- Python
- Zeek
- Apache Kafka
- Apache Spark (PySpark)
- MinIO
- PostgreSQL
- Apache Airflow
- dbt
- Scikit-learn and XGBoost
- SHAP
- FastAPI
- Streamlit and Plotly
- Docker

## Architecture

Network Traffic / PCAP → Zeek → Kafka → PySpark → Data Lake / PostgreSQL → ML Detection → API / Dashboard

## Current Progress

### Phase 1 — Foundation and Environment
- [x] Python environment setup
- [x] Git initialization
- [x] Docker setup

### Phase 2 — Network Data Collection
- [x] Zeek installation and verification
- [x] Sample PCAP acquisition
- [x] Zeek TSV log generation
- [x] Zeek JSON log generation
- [x] Python-based JSON ingestion
- [x] Record validation and normalization

### Upcoming Phases
- [ ] Phase 3: Dataset Engineering
- [ ] Phase 4: Kafka Streaming Pipeline
- [ ] Phase 5: PySpark Data Processing
- [ ] Phase 6: Data Lake and Warehouse
- [ ] Phase 7: Machine Learning Engine
- [ ] Phase 8: Orchestration and Analytics
- [ ] Phase 9: Backend and Dashboard
- [ ] Phase 10: Testing and Deployment

## Project Status

Currently under active development.

## Disclaimer

This project is intended for educational and authorized cybersecurity testing environments only.