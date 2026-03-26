from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ml.recommender_service import get_recommendations, semantic_search


app = FastAPI(
    title="UniFy Recommender API",
    description="Recommendation engine and semantic search for universities",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RecommendationRequest(BaseModel):
    studentId: str
    topN: int = 10


class RecommendationResponse(BaseModel):
    universityId: str
    name: str
    location: str
    score: float
    contentScore: float
    collabScore: float


class SemanticSearchRequest(BaseModel):
    query: str
    searchType: str = "both"
    topN: int = 10


class SearchResult(BaseModel):
    type: str
    id: str
    name: str
    score: float


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/api/recommendations", response_model=dict)
async def get_user_recommendations(req: RecommendationRequest):
    """Get personalized university recommendations for a student."""
    try:
        recommendations = get_recommendations(req.studentId, req.topN)
        return {"recommendations": recommendations}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.post("/api/search", response_model=dict)
async def search(req: SemanticSearchRequest):
    """Semantic search across universities and courses."""
    try:
        results = semantic_search(req.query, req.searchType, req.topN)
        return {
            "query": req.query,
            "results": results,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")


@app.get("/api/recommendations/{student_id}")
async def get_recommendations_by_id(student_id: str, top_n: int = 10):
    """Alternative GET endpoint for recommendations."""
    try:
        recommendations = get_recommendations(student_id, top_n)
        return {"recommendations": recommendations}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
