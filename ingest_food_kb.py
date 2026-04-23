"""
Ingestion script: merges Indian food datasets, generates embeddings, stores in Supabase.
Run once: python ingest_food_kb.py
"""

import csv
import json
import time
import ollama
from supabase import create_client
from dotenv import load_dotenv
import os

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_ANON_KEY")
EMBED_MODEL = "nomic-embed-text"

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

NUTRITION_CSV = "data/dataset2/Indian_Food_Nutrition_Processed.csv"
FOOD_CSV = "data/dataset1/indian_food.csv"


def load_nutrition_data() -> dict[str, dict]:
    nutrition = {}
    with open(NUTRITION_CSV, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["Dish Name"].strip().lower()
            nutrition[name] = {
                "calories_kcal": float(row.get("Calories (kcal)") or 0),
                "carbs_g": float(row.get("Carbohydrates (g)") or 0),
                "protein_g": float(row.get("Protein (g)") or 0),
                "fat_g": float(row.get("Fats (g)") or 0),
                "fiber_g": float(row.get("Fibre (g)") or 0),
                "sodium_mg": float(row.get("Sodium (mg)") or 0),
                "calcium_mg": float(row.get("Calcium (mg)") or 0),
                "iron_mg": float(row.get("Iron (mg)") or 0),
                "vitamin_c_mg": float(row.get("Vitamin C (mg)") or 0),
            }
    print(f"Loaded {len(nutrition)} nutrition entries")
    return nutrition


def load_food_data() -> list[dict]:
    foods = []
    with open(FOOD_CSV, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            foods.append({
                "dish_name": row["name"].strip(),
                "ingredients": [i.strip() for i in row.get("ingredients", "").split(",")],
                "diet": row.get("diet", "").strip(),
                "course": row.get("course", "").strip(),
                "flavor_profile": row.get("flavor_profile", "").strip(),
                "region": row.get("region", "").strip(),
            })
    print(f"Loaded {len(foods)} food entries")
    return foods


def build_description(dish: dict, nutrition: dict) -> str:
    ingr = ", ".join(dish["ingredients"][:8]) if dish["ingredients"] else "various ingredients"
    n = nutrition
    parts = [
        f"{dish['dish_name']} is a {dish.get('diet', '')} {dish.get('course', 'dish')} from {dish.get('region', 'India')}.",
        f"Ingredients: {ingr}.",
        f"Flavor: {dish.get('flavor_profile', 'mixed')}.",
    ]
    if n.get("calories_kcal"):
        parts.append(
            f"Nutrition per serving: {n['calories_kcal']} kcal, "
            f"{n['protein_g']}g protein, {n['carbs_g']}g carbs, {n['fat_g']}g fat, {n['fiber_g']}g fiber."
        )
    return " ".join(parts)


def get_embedding(text: str) -> list[float]:
    response = ollama.embeddings(model=EMBED_MODEL, prompt=text)
    return response["embedding"]


def load_nutrition_as_dishes(nutrition_map: dict, food_metadata: dict) -> list[dict]:
    """Build full dish list — all from nutrition CSV, enriched with metadata where available."""
    dishes = []
    for name, n in nutrition_map.items():
        meta = food_metadata.get(name, {})
        dishes.append({
            "dish_name": meta.get("dish_name") or name.title(),
            "ingredients": meta.get("ingredients", []),
            "diet": meta.get("diet", ""),
            "course": meta.get("course", ""),
            "flavor_profile": meta.get("flavor_profile", ""),
            "region": meta.get("region", "India"),
            **n,
        })
    return dishes


def ingest():
    nutrition_map = load_nutrition_data()
    foods_raw = load_food_data()

    food_metadata = {f["dish_name"].lower(): f for f in foods_raw}

    all_dishes = load_nutrition_as_dishes(nutrition_map, food_metadata)

    for f in foods_raw:
        if f["dish_name"].lower() not in nutrition_map:
            all_dishes.append({**f, **{k: None for k in ["calories_kcal","protein_g","carbs_g","fat_g","fiber_g","sodium_mg","calcium_mg","iron_mg","vitamin_c_mg"]}})

    print(f"\nStarting ingestion of {len(all_dishes)} dishes...\n")
    success = 0
    skipped = 0

    for i, dish in enumerate(all_dishes):
        description = build_description(dish, dish)

        try:
            embedding = get_embedding(description)
        except Exception as e:
            print(f"  [{i+1}] Embedding failed for {dish['dish_name']}: {e}")
            skipped += 1
            continue

        record = {
            "dish_name": dish["dish_name"],
            "region": dish.get("region"),
            "ingredients": dish.get("ingredients") or [],
            "diet": dish.get("diet"),
            "course": dish.get("course"),
            "flavor_profile": dish.get("flavor_profile"),
            "calories_kcal": dish.get("calories_kcal"),
            "protein_g": dish.get("protein_g"),
            "carbs_g": dish.get("carbs_g"),
            "fat_g": dish.get("fat_g"),
            "fiber_g": dish.get("fiber_g"),
            "sodium_mg": dish.get("sodium_mg"),
            "calcium_mg": dish.get("calcium_mg"),
            "iron_mg": dish.get("iron_mg"),
            "vitamin_c_mg": dish.get("vitamin_c_mg"),
            "description": description,
            "embedding": embedding,
        }

        try:
            supabase.table("indian_food_kb").insert(record).execute()
            success += 1
            print(f"  [{i+1}/{len(all_dishes)}] ✓ {dish['dish_name']}")
        except Exception as e:
            print(f"  [{i+1}] DB insert failed for {dish['dish_name']}: {e}")
            skipped += 1

        time.sleep(0.05)

    print(f"\n✅ Done! {success} dishes ingested, {skipped} skipped.")


if __name__ == "__main__":
    ingest()
