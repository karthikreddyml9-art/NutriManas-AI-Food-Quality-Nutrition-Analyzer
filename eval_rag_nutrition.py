"""
=============================================================
NutriManas — RAG Nutrition Evaluation Script
=============================================================

WHAT IS RAG EVAL?
  We test whether our Indian Food Knowledge Base returns
  ACCURATE nutrition data for known dishes.

  We compare our KB values against a ground truth table
  (manually verified values from IFCT 2017 / NIN India).

HOW IT WORKS:
  1. For each test dish → generate embedding → search KB
  2. Get returned nutrition values
  3. Compare against ground truth
  4. Calculate Mean Absolute Error (MAE) per nutrient
  5. Score each result as GOOD / ACCEPTABLE / POOR

METRICS:
  - Retrieval Hit Rate: Did KB find the right dish?
  - Similarity Score: How confident was the vector search?
  - Calorie MAE: Average calorie error in kcal
  - Protein MAE: Average protein error in grams
  - Overall RAG Score: Combined quality score

HOW TO RUN:
  venv\\Scripts\\python.exe eval_rag_nutrition.py
  (requires Ollama running + Supabase connected)
=============================================================
"""

import sys
import json
import math
from typing import Optional

# ── Ground Truth Table ─────────────────────────────────────
# Source: IFCT 2017 (Indian Food Composition Tables)
# National Institute of Nutrition, India
# Values are per 100g unless noted
GROUND_TRUTH = [
    {
        "dish": "biryani",
        "calories": 150, "protein": 6.5, "carbs": 21.0,
        "fat": 4.5, "fiber": 0.8, "sodium": 420,
    },
    {
        "dish": "idli",
        "calories": 58, "protein": 2.0, "carbs": 12.0,
        "fat": 0.4, "fiber": 0.5, "sodium": 310,
    },
    {
        "dish": "dosa",
        "calories": 133, "protein": 3.5, "carbs": 22.0,
        "fat": 3.7, "fiber": 0.9, "sodium": 280,
    },
    {
        "dish": "samosa",
        "calories": 262, "protein": 4.5, "carbs": 32.0,
        "fat": 13.0, "fiber": 2.0, "sodium": 420,
    },
    {
        "dish": "dal tadka",
        "calories": 118, "protein": 7.5, "carbs": 16.0,
        "fat": 3.2, "fiber": 4.0, "sodium": 380,
    },
    {
        "dish": "butter chicken",
        "calories": 164, "protein": 16.0, "carbs": 7.0,
        "fat": 8.5, "fiber": 1.0, "sodium": 490,
    },
    {
        "dish": "palak paneer",
        "calories": 168, "protein": 8.5, "carbs": 8.0,
        "fat": 11.5, "fiber": 2.5, "sodium": 350,
    },
    {
        "dish": "chole",
        "calories": 164, "protein": 8.0, "carbs": 24.0,
        "fat": 4.5, "fiber": 7.0, "sodium": 400,
    },
    {
        "dish": "aloo gobi",
        "calories": 97, "protein": 2.5, "carbs": 14.0,
        "fat": 3.8, "fiber": 3.0, "sodium": 310,
    },
    {
        "dish": "rajma",
        "calories": 144, "protein": 9.0, "carbs": 22.0,
        "fat": 2.5, "fiber": 7.5, "sodium": 350,
    },
]

# ── KB Search (reuse RAG agent logic) ─────────────────────
sys.path.insert(0, ".")

def search_kb(query: str, threshold: float = 0.78) -> Optional[dict]:
    """Search the Indian Food KB and return best match."""
    try:
        from agents.rag_nutrition_agent import _search_kb
        results = _search_kb(query, threshold=threshold, limit=1)
        if results:
            return results[0]
    except Exception as e:
        print(f"  [ERROR] KB search failed: {e}")
    return None


def pct_error(predicted, truth) -> float:
    """Percentage error between predicted and ground truth."""
    if truth == 0:
        return 0.0
    return abs(predicted - truth) / truth * 100


def grade(pct: float) -> str:
    """Grade the accuracy of a prediction."""
    if pct <= 10:
        return "✅ GOOD"
    elif pct <= 25:
        return "⚠️  ACCEPTABLE"
    else:
        return "❌ POOR"


# ── Run Evaluation ─────────────────────────────────────────
print("=" * 60)
print("  NutriManas — RAG Nutrition Evaluation")
print("=" * 60)
print(f"\n  Testing {len(GROUND_TRUTH)} dishes against Indian Food KB\n")

results = []
hit_count = 0
total_cal_err = []
total_protein_err = []
total_carb_err = []
total_fat_err = []

for gt in GROUND_TRUTH:
    dish = gt["dish"]
    print(f"  🔍 Searching: '{dish}'")

    kb_result = search_kb(dish)

    if kb_result is None:
        print(f"     ❌ NOT FOUND in KB\n")
        results.append({"dish": dish, "found": False})
        continue

    similarity = kb_result.get("similarity", 0)
    kb_name    = kb_result.get("dish_name", "unknown")
    hit_count += 1

    # Get nutrition values from KB
    kb_cal     = kb_result.get("calories_kcal") or 0
    kb_protein = kb_result.get("protein_g") or 0
    kb_carbs   = kb_result.get("carbs_g") or 0
    kb_fat     = kb_result.get("fat_g") or 0

    # Calculate errors
    cal_err     = pct_error(kb_cal, gt["calories"])
    protein_err = pct_error(kb_protein, gt["protein"])
    carb_err    = pct_error(kb_carbs, gt["carbs"])
    fat_err     = pct_error(kb_fat, gt["fat"])

    total_cal_err.append(cal_err)
    total_protein_err.append(protein_err)
    total_carb_err.append(carb_err)
    total_fat_err.append(fat_err)

    print(f"     ✅ Matched: '{kb_name}' (similarity={similarity:.2f})")
    print(f"     {'Nutrient':<12} {'KB Value':>10} {'Ground Truth':>14} {'Error':>8} {'Grade':>15}")
    print(f"     {'-'*12} {'-'*10} {'-'*14} {'-'*8} {'-'*15}")
    print(f"     {'Calories':<12} {kb_cal:>9.1f} {gt['calories']:>13.1f} {cal_err:>7.1f}% {grade(cal_err):>15}")
    print(f"     {'Protein':<12} {kb_protein:>9.1f} {gt['protein']:>13.1f} {protein_err:>7.1f}% {grade(protein_err):>15}")
    print(f"     {'Carbs':<12} {kb_carbs:>9.1f} {gt['carbs']:>13.1f} {carb_err:>7.1f}% {grade(carb_err):>15}")
    print(f"     {'Fat':<12} {kb_fat:>9.1f} {gt['fat']:>13.1f} {fat_err:>7.1f}% {grade(fat_err):>15}")
    print()

    results.append({
        "dish": dish, "found": True,
        "kb_name": kb_name, "similarity": similarity,
        "cal_err": cal_err, "protein_err": protein_err,
    })

# ── Summary ────────────────────────────────────────────────
print("=" * 60)
print("  OVERALL RAG EVALUATION SUMMARY")
print("=" * 60)

hit_rate = hit_count / len(GROUND_TRUTH) * 100
avg_cal_err     = sum(total_cal_err) / len(total_cal_err) if total_cal_err else 0
avg_protein_err = sum(total_protein_err) / len(total_protein_err) if total_protein_err else 0
avg_carb_err    = sum(total_carb_err) / len(total_carb_err) if total_carb_err else 0
avg_fat_err     = sum(total_fat_err) / len(total_fat_err) if total_fat_err else 0

print(f"\n  Retrieval Hit Rate : {hit_rate:.1f}%  ({hit_count}/{len(GROUND_TRUTH)} dishes found)")
print(f"\n  Average Errors (lower is better):")
print(f"    Calories : {avg_cal_err:.1f}%  {grade(avg_cal_err)}")
print(f"    Protein  : {avg_protein_err:.1f}%  {grade(avg_protein_err)}")
print(f"    Carbs    : {avg_carb_err:.1f}%  {grade(avg_carb_err)}")
print(f"    Fat      : {avg_fat_err:.1f}%  {grade(avg_fat_err)}")

overall = (hit_rate + (100 - avg_cal_err) + (100 - avg_protein_err)) / 3
print(f"\n  Overall RAG Score  : {overall:.1f}/100")
print(f"\n  {'✅ PRODUCTION READY' if overall > 75 else '⚠️  NEEDS IMPROVEMENT'}")
print("\n  Tip: Low hit rate = add more dishes to KB")
print("  Tip: High error = KB values may need cleaning\n")
