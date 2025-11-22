import json
import random
import re
from typing import List, Dict, Any



def load_questions(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["questions"]


def parse_options(options_raw: str):
    parts = re.split(r"[\/;,]", options_raw)
    return [p.strip() for p in parts if p.strip()]



def simulate_answers_for_users(
    questions: List[Dict[str, Any]],
    num_users: int = 100,
    min_q: int = 10,
    max_q: int = 15,
) -> List[Dict[str, Any]]:

    open_qs = [q for q in questions if q["question_type"] == "open_ended"]
    non_open_qs = [q for q in questions if q["question_type"] != "open_ended"]



    categories = {q["category"] for q in questions}
    cat_effless_prob = {cat: random.uniform(0.2, 0.6) for cat in categories}

    all_responses: List[Dict[str, Any]] = []

    for user_id in range(1, num_users + 1):
        n_total = random.randint(min_q, max_q)

        if open_qs:
            n_open = round(0.1 * n_total)
            n_open = max(0, min(n_open, len(open_qs), n_total))
        else:
            n_open = 0

        n_non_open = n_total - n_open

        sampled_open = random.sample(open_qs, n_open) if n_open > 0 else []
        if n_non_open <= len(non_open_qs):
            sampled_non_open = random.sample(non_open_qs, n_non_open)
        else:
            sampled_non_open = random.choices(non_open_qs, k=n_non_open)

        sampled_questions = sampled_open + sampled_non_open
        random.shuffle(sampled_questions)

        for q in sampled_questions:
            qid = q["id"]
            qtype = q["question_type"]
            cat = q["category"]
            options = parse_options(q.get("options", ""))

            if qtype == "open_ended":
                answer_prob = 0.7
            else:
                answer_prob = random.uniform(0.8, 0.95)

            if random.random() > answer_prob:
                answer = "not_answered"
            else:
                if qtype == "single_choice":
                    answer = random.choice(options) if options else "N/A"

                elif qtype == "multiple_choice":
                    if options:
                        k = random.randint(1, len(options))
                        chosen = random.sample(options, k)
                        answer = " / ".join(chosen)
                    else:
                        answer = "N/A"

                elif qtype == "yes_no":
                    answer = random.choice(["Yes", "No"])

                elif qtype == "rating":
                    answer = str(random.randint(0, 10))

                elif qtype == "open_ended":
                    answer = random.choice(["Yes", "No"])

                else:
                    answer = ""

            eff_prob = cat_effless_prob[cat]
            effective_flag = random.random() > eff_prob

            all_responses.append(
                {
                    "user_id": user_id,
                    "question_id": qid,
                    "answer": answer,
                    "effective": effective_flag,
                }
            )

    return all_responses



def main():
    questions = load_questions("convert_data.json")
    responses = simulate_answers_for_users(
        questions,
        num_users=8466,
        min_q=10,
        max_q=20,
    )

    with open("simulated_responses.json", "w", encoding="utf-8") as f:
        json.dump({"responses": responses}, f, ensure_ascii=False, indent=2)

    print(f"{len(responses)} generated saved as simulated_responses.json")


if __name__ == "__main__":
    main()
