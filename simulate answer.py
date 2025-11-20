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


def sample_questions_for_user(
    questions: List[Dict[str, Any]],
    n_questions: int,
    open_ended_accept_prob: float = 0.1,
) -> List[Dict[str, Any]]:
    selected = []
    used_ids = set()

    while len(selected) < n_questions:
        q = random.choice(questions)
        qid = q["id"]
        if qid in used_ids:
            continue

        if q["question_type"] == "open_ended":
            if random.random() > open_ended_accept_prob:
                continue

        selected.append(q)
        used_ids.add(qid)

    return selected


def simulate_answers_for_users(
    questions: List[Dict[str, Any]],
    num_users: int = 100,
    min_q: int = 10,
    max_q: int = 15,
) -> List[Dict[str, Any]]:

    all_answers = []

    for user_id in range(1, num_users + 1):
        n_questions = random.randint(min_q, max_q)
        sampled_questions = sample_questions_for_user(
            questions, n_questions, open_ended_accept_prob=0.2
        )

        for q in sampled_questions:
            qid = q["id"]
            q_type = q["question_type"]
            options_raw = q.get("options", "")
            if q_type == "open_ended":
                skip_prob = 0.30
            else:
                skip_prob = 0.05

            if random.random() < skip_prob:
                answer = "not_answered"
            else:

                if q_type == "single_choice":
                    options = parse_options(options_raw)
                    answer = random.choice(options) if options else "N/A"

                elif q_type == "multiple_choice":
                    options = parse_options(options_raw)
                    if options:
                        k = random.randint(1, len(options))
                        chosen = random.sample(options, k)
                        answer = " / ".join(chosen)
                    else:
                        answer = "N/A"

                elif q_type == "yes_no":
                    answer = random.choice(["Yes", "No"])

                elif q_type == "open_ended":
                    answer = random.choice(["Yes", "No"])

                elif q_type == "rating":
                    answer = str(random.randint(0, 10))

                else:
                    answer = ""

            all_answers.append(
                {
                    "user_id": user_id,
                    "question_id": qid,
                    "answer": answer,
                }
            )

    return all_answers


def main():
    questions = load_questions("convert_data.json")

    answers = simulate_answers_for_users(
        questions,
        num_users=100,
        min_q=10,
        max_q=15,
    )

    out_data = {"responses": answers}
    with open("simulated_responses.json", "w", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False, indent=2)

    print(f"Generated {len(answers)} responses.")
    print("Saved to simulated_responses.json")


if __name__ == "__main__":
    main()
