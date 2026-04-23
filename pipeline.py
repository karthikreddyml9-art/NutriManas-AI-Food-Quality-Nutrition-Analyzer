"""
LangGraph multi-agent pipeline
State flows: classify → nutrition → quality → recommend → explain
"""

from typing import Any, TypedDict
from langgraph.graph import StateGraph, END

from agents.food_classifier import classify_food
from agents.rag_nutrition_agent import calculate_nutrition_rag
from agents.quality_analyzer import analyze_quality
from agents.health_recommender import get_health_recommendations
from agents.llm_explainer import explain_results


class NutriState(TypedDict):
    image_bytes: bytes
    user_profile: dict
    classification: dict
    nutrition: dict
    quality: dict
    health: dict
    explanation: dict
    error: str | None


async def agent_classify(state: NutriState) -> dict:
    try:
        result = classify_food(state["image_bytes"])
        return {"classification": result}
    except Exception as e:
        return {"classification": {}, "error": f"Classifier failed: {e}"}


async def agent_nutrition(state: NutriState) -> dict:
    try:
        cls = state.get("classification", {})
        detections = cls.get("detected_objects", [])
        area_ratio = detections[0]["area_ratio"] if detections else 0.2
        result = await calculate_nutrition_rag(
            food_name=cls.get("food_name", "unknown food"),
            ingredients=cls.get("ingredients", []),
            portion_label=cls.get("portion_size", "medium (150-250g)"),
            area_ratio=area_ratio,
        )
        return {"nutrition": result}
    except Exception as e:
        return {"nutrition": {}, "error": f"Nutrition calc failed: {e}"}


async def agent_quality(state: NutriState) -> dict:
    try:
        result = await analyze_quality(state["image_bytes"])
        return {"quality": result}
    except Exception as e:
        return {"quality": {}, "error": f"Quality analysis failed: {e}"}


async def agent_recommend(state: NutriState) -> dict:
    try:
        cls = state.get("classification", {})
        nutrition = state.get("nutrition", {}).get("nutrition", {})
        result = await get_health_recommendations(
            food_name=cls.get("food_name", "unknown food"),
            nutrition=nutrition,
            profile=state.get("user_profile"),
        )
        return {"health": result}
    except Exception as e:
        return {"health": {}, "error": f"Recommender failed: {e}"}


async def agent_explain(state: NutriState) -> dict:
    try:
        cls = state.get("classification", {})
        result = await explain_results(
            food_name=cls.get("food_name", "unknown food"),
            classification=cls,
            nutrition=state.get("nutrition", {}),
            quality=state.get("quality", {}),
            health=state.get("health", {}),
        )
        return {"explanation": result}
    except Exception as e:
        return {"explanation": {}, "error": f"Explainer failed: {e}"}


def build_pipeline() -> StateGraph:
    graph = StateGraph(NutriState)

    graph.add_node("classify", agent_classify)
    graph.add_node("nutrition", agent_nutrition)
    graph.add_node("quality", agent_quality)
    graph.add_node("recommend", agent_recommend)
    graph.add_node("explain", agent_explain)

    graph.set_entry_point("classify")
    graph.add_edge("classify", "nutrition")
    graph.add_edge("nutrition", "quality")
    graph.add_edge("quality", "recommend")
    graph.add_edge("recommend", "explain")
    graph.add_edge("explain", END)

    return graph.compile()


nutrimanas_pipeline = build_pipeline()


async def run_analysis(image_bytes: bytes, user_profile: dict | None = None) -> dict[str, Any]:
    initial_state: NutriState = {
        "image_bytes": image_bytes,
        "user_profile": user_profile or {},
        "classification": {},
        "nutrition": {},
        "quality": {},
        "health": {},
        "explanation": {},
        "error": None,
    }

    final_state = await nutrimanas_pipeline.ainvoke(initial_state)

    return {
        "food_name": final_state["classification"].get("food_name", "Unknown"),
        "classification": final_state["classification"],
        "nutrition": final_state["nutrition"],
        "quality": final_state["quality"],
        "health": final_state["health"],
        "explanation": final_state["explanation"],
        "error": final_state.get("error"),
    }
