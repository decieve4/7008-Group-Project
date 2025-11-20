import json
import random
import re
from typing import List, Dict, Any


def load_questions(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["questions"]


def parse_options(options_raw: str) -> List[str]:
    parts = re.split(r"[\/;,]", options_raw)
    return [p.strip() for p in parts if p.strip()]


def simulate_answers_for_users(
    questions: List[Dict[str, Any]],
    num_users: int = 100,
    min_q: int = 10,
    max_q: int = 15,
) -> List[Dict[str, Any]]:
    all_answers: List[Dict[str, Any]] = []

    for user_id in range(1, num_users + 1):
        n_questions = random.randint(min_q, max_q)
        sampled_questions = random.sample(questions, n_questions)

        for q in sampled_questions:
            q_id = q["id"]
            q_type = q["question_type"]
            options_raw = q.get("options", "")
            if q_type == "single_choice":
                options = parse_options(options_raw)
                if options:
                    answer = random.choice(options)
                else:
                    answer = "N/A"
            elif q_type == "open_ended":
                answer = random.choice(["Yes", "No"])
            elif q_type == "rating":
                answer = str(random.randint(0, 10))
            else:
                answer = ""

            all_answers.append(
                {
                    "user_id": user_id,
                    "question_id": q_id,
                    "answer": answer,
                }
            )

    return all_answers


def main():
    questions = load_questions("convert_data.json")
    # num_users: simulated number of users, min_q,max_q, range of number of questions each user answers
    answers = simulate_answers_for_users(questions, num_users=10000, min_q=10, max_q=15)
    out_data = {"responses": answers}
    with open("simulated_responses.json", "w", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False, indent=2)

    print(f" {len(answers)} answers generated，save to simulated_responses.json")

if __name__ == "__main__":
    main()