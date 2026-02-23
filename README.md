# Credit Risk Decision & Monitoring System

End-to-end credit risk scoring system built with:

- XGBoost (profit-optimized training)
- FastAPI (production inference)
- Docker (containerized deployment)
- Streamlit (dashboard UI)
- AWS EC2 deployment

## Features

- Profit-based threshold optimization
- Model registry pattern
- Drift monitoring utilities
- Stress testing simulation
- Real-time API inference
- Batch + Single applicant UI

## Architecture

User → Streamlit → FastAPI → XGBoost  
                     ↓  
               Model Registry  
                     ↓  
              Monitoring & Drift  

## Run Locally

Build API:

docker build -t credit-risk-api .
docker run -p 8000:8000 credit-risk-api

API docs:
http://localhost:8000/docs
