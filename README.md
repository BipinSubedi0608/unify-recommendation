# UniFy Recommender API

Production-ready FastAPI recommendation service for university suggestions and semantic search.

## What this repo contains

- API service: [api/main.py](http://_vscodecontentref_/8)
- Inference logic: [ml/recommender_service.py](http://_vscodecontentref_/9)
- Trained model: [ml/models/recommender.pkl](http://_vscodecontentref_/10)
- Dependencies: [requirements.txt](http://_vscodecontentref_/11)
- Container config: [Dockerfile](http://_vscodecontentref_/12)
- Railway process entry: [Procfile](http://_vscodecontentref_/13)

## Endpoints

- GET /health
- POST /api/recommendations
- POST /api/search

## Local run

1. Install dependencies:
   pip install -r [requirements.txt](http://_vscodecontentref_/14)

2. Start API:
   uvicorn api.main:app --host 0.0.0.0 --port 8000

3. Test:
   curl http://localhost:8000/health

## Docker run

1. Build:
   docker build -t unify-recommender:latest .

2. Run:
   docker run -p 8000:8000 unify-recommender:latest

## Deploy to Railway

1. Push this repo to GitHub
2. Create a new Railway project from the repo
3. Set root directory to this project folder
4. Deploy (Railway reads [Procfile](http://_vscodecontentref_/15))
5. Verify:
   GET /health returns {"status":"ok"}

## Notes

- The model file is required at runtime.
- Do not remove [ml/models/recommender.pkl](http://_vscodecontentref_/16) unless you replace loading with remote model download.
