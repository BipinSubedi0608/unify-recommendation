import pickle
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


class RecommenderModel:
    """Singleton recommender model loader with in-memory caching."""

    _instance: Optional["RecommenderModel"] = None
    _model = None
    _vectorizer = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_model()
        return cls._instance

    def _load_model(self) -> None:
        """Load model from pickle file into memory."""
        model_path = Path(__file__).parent.parent / "ml" / "models" / "recommender.pkl"

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")

        with model_path.open("rb") as f:
            bundle = pickle.load(f)

        self._model = bundle
        self._vectorizer = bundle.get("vectorizer")

    @property
    def model(self):
        if self._model is None:
            raise RuntimeError("Model not loaded")
        return self._model

    @property
    def vectorizer(self) -> TfidfVectorizer:
        if self._vectorizer is None:
            raise RuntimeError("Vectorizer not loaded")
        return self._vectorizer


def normalize(scores: np.ndarray) -> np.ndarray:
    """Normalize scores to [0, 1]."""
    if scores.size == 0:
        return scores
    min_v = float(scores.min())
    max_v = float(scores.max())
    if max_v - min_v < 1e-9:
        return np.zeros_like(scores)
    return (scores - min_v) / (max_v - min_v)


def get_recommendations(student_id: str, top_n: int = 10) -> list[dict]:
    """Get top-N university recommendations for a student."""
    loader = RecommenderModel()
    model = loader.model

    uni_ids = model["uni_ids"]
    universities = model["universities"]
    uni_matrix = model["uni_matrix"]

    if student_id not in model["student_map"]:
        raise ValueError(f"Unknown student id: {student_id}")

    student_vec = model["student_vectors"][student_id]
    content_scores = cosine_similarity(student_vec, uni_matrix).flatten()
    content_scores = normalize(content_scores)

    collab_map = model.get("collab", {}).get(student_id, {})
    collab_scores = np.array(
        [float(collab_map.get(uid, 0.0)) for uid in uni_ids], dtype=np.float32
    )
    collab_scores = normalize(collab_scores)

    combined = (
        model.get("content_weight", 0.8) * content_scores
        + model.get("collab_weight", 0.2) * collab_scores
    )

    ranked_indices = np.argsort(combined)[::-1][:top_n]

    out = []
    for idx in ranked_indices:
        uni = universities[idx]
        out.append(
            {
                "universityId": uni["id"],
                "name": uni["name"],
                "location": uni.get("location", ""),
                "score": float(combined[idx]),
                "contentScore": float(content_scores[idx]),
                "collabScore": float(collab_scores[idx]),
            }
        )

    return out


def semantic_search(query: str, search_type: str = "both", top_n: int = 10) -> list[dict]:
    """Search for universities or courses using semantic similarity."""
    loader = RecommenderModel()
    model = loader.model
    vectorizer = loader.vectorizer

    universities = model["universities"]
    query_vec = vectorizer.transform([query]).toarray()[0]

    results = []

    if search_type in ("both", "universities"):
        uni_matrix = model["uni_matrix"].toarray()
        uni_scores = cosine_similarity([query_vec], uni_matrix)[0]

        ranked_uni_indices = np.argsort(uni_scores)[::-1][:top_n]
        for idx in ranked_uni_indices:
            if uni_scores[idx] > 0.01:
                uni = universities[idx]
                results.append(
                    {
                        "type": "university",
                        "id": uni["id"],
                        "name": uni["name"],
                        "location": uni.get("location", ""),
                        "description": uni.get("description", "")[:200],
                        "score": float(uni_scores[idx]),
                    }
                )

    if search_type in ("both", "courses"):
        for uni in universities[:100]:
            for course in uni.get("courses", []):
                course_text = f"{course['title']} {course.get('code', '')}"
                course_vec = vectorizer.transform([course_text]).toarray()[0]
                score = float(cosine_similarity([query_vec], [course_vec])[0][0])

                if score > 0.01:
                    results.append(
                        {
                            "type": "course",
                            "id": course["id"],
                            "name": course["title"],
                            "university": uni["name"],
                            "universityId": uni["id"],
                            "degreeType": course.get("degreeType"),
                            "score": score,
                        }
                    )

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_n]
