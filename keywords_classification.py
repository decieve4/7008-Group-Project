import pandas as pd
import numpy as np
import re
CATEGORY_KEYWORDS = {
    "satisfaction": [
        "satisfaction", "satisfied", "dissatisfied", "unsatisfied",
        "overall satisfaction", "overall experience",
        "happy", "unhappy", "pleased", "disappointed",
        "overall impression", "overall rating", "experience"
    ],
    "service": [
        "service", "services", "staff", "employees", "personnel",
        "reception", "front desk", "concierge",
        "check in", "check-in", "check out", "check-out",
        "housekeeping", "cleaning service",
        "friendliness", "helpful", "attentive", "rude", "polite"
    ],
    "room": [
        "room", "rooms", "suite", "bedroom", "bed", "beds",
        "room size", "spacious", "small room", "quiet room",
        "view", "sea view", "city view"
    ],
    "tour": [
        "tour", "tours", "tour guide", "guided tour",
        "excursion", "excursions", "trip", "day trip",
        "itinerary", "schedule", "route", "program"
    ],
    "recommendation": [
        "recommend", "would you recommend", "likely to recommend",
        "likelihood to recommend", "nps", "net promoter",
        "how likely", "recommend to friends", "recommend to family"
    ],
    "location": [
        "location", "located", "area", "neighborhood", "surroundings",
        "distance", "walking distance", "far", "near", "nearby",
        "close to", "convenient", "central", "downtown", "city center",
        "access", "accessibility"
    ],
    "price": [
        "price", "cost", "fee", "charge", "fare",
        "expensive", "cheap", "low price", "high price",
        "budget", "value for money", "good value", "worth the money",
        "affordable", "overpriced", "price level", "pricing"
    ],
    "food": [
        "food", "meal", "meals", "breakfast", "lunch", "dinner",
        "restaurant", "cafe", "cafeteria", "bar",
        "buffet", "menu", "cuisine",
        "taste", "quality of food", "variety of food"
    ],
    "attraction": [
        "attraction", "attractions", "sightseeing", "sights",
        "landmark", "landmarks", "point of interest", "points of interest",
        "museum", "park", "monument"
    ],
    "transportation": [
        "transportation", "transport", "public transport",
        "bus", "train", "metro", "subway",
        "airport", "shuttle", "airport shuttle", "transfer",
        "taxi", "cab", "parking", "car park"
    ]
}


def reclassify_general_by_keywords(
    df: pd.DataFrame,
    category_keywords: dict,
    text_cols=("question_text", "options_text")
) -> pd.DataFrame:

    df = df.copy()

    if "category" not in df.columns:
        raise KeyError("DataFrame 中缺少 'category' 列")

    mask_general = df["category"] == "general"
    general_indices = np.flatnonzero(mask_general.to_numpy())


    for idx in general_indices:
        row = df.iloc[idx]

        parts = []
        for col in text_cols:
            if col in df.columns:
                val = row[col]
                if isinstance(val, list):
                    parts.append(" ".join(map(str, val)))
                elif pd.notna(val):
                    parts.append(str(val))
        full_text = " ".join(parts).lower()

        cat_hits = {}

        for cat, keywords in category_keywords.items():
            hits = 0
            for kw in keywords:
                kw = kw.lower()
                if " " in kw:
                    pattern = r"\b" + re.escape(kw) + r"\b"
                    if re.search(pattern, full_text):
                        hits += 1
                else:
                    if kw in full_text:
                        hits += 1
            if hits > 0:
                cat_hits[cat] = hits

        if cat_hits:
            best_cat = max(cat_hits, key=cat_hits.get)
            df.at[idx, "category"] = best_cat

    return df


if __name__ == "__main__":
    df = pd.read_json("cleaned_questions_all.json")
    df_reclassified = reclassify_general_by_keywords(df, CATEGORY_KEYWORDS)
    if "question_type" not in df_reclassified.columns:
        raise KeyError("DataFrame is missing 'question_type' column")

    remove_types = ["yes_no", "multiple_choice"]
    mask_keep = ~df_reclassified["question_type"].isin(remove_types)
    df_final = df_reclassified[mask_keep].reset_index(drop=True)
    output_path = "questions_reclassified.json"
    df_final.to_json(output_path, orient="records", force_ascii=False, indent=2)

