from flask import Flask, render_template, request, send_file, abort, url_for
import json
import os

from Q5_v2 import generate_questionnaire, is_chinese

app = Flask(__name__)

OUTPUT_JSON_PATH = "output_questionaire.json"
OUTPUT_TXT_PATH = "output_questionaire.txt"

# Example categories, you may want to extract them dynamically from your question bank
AVAILABLE_CATEGORIES = [
    "General", "Satisfaction", "Location", "Service", "Price", "Quality", "Experience"
]

def build_requirements_json(
    user_input: str, 
    question_count: int, 
    selected_category: str,
    req_path: str = "questionaire_requirements.json"
) -> None:
    """
    Build a minimal requirements JSON file based on the user input
    from the web form, so we can reuse generate_questionnaire()
    without changing its internal logic.
    """
    # Simple language detection: CJK => zh, otherwise en
    language = "zh" if is_chinese(user_input) else "en"

    requirements = {
        "requirements": {
            "title": "Web Generated Questionnaire",
            "description": user_input,
            "topic": user_input,
            "language": language,
            "categories": [selected_category] if selected_category else [],
            "question_count": question_count
        }
    }

    with open(req_path, "w", encoding="utf-8") as f:
        json.dump(requirements, f, ensure_ascii=False, indent=2)


@app.route("/", methods=["GET", "POST"])
def index():
    """
    Main page:
    - GET: show the form
    - POST: build requirements, call generate_questionnaire, and render results
    """
    generated_questions = []
    user_input = ""
    question_count = 5  # Default question count
    selected_category = ""

    if request.method == "POST":
        user_input = request.form.get("user_input", "").strip()
        question_count = int(request.form.get("question_count", 5))  # Capture the number of questions
        selected_category = request.form.get("category", "").strip()  # Capture selected category

        if user_input:
            # 1) Write requirements JSON for Q5_v2 to consume
            build_requirements_json(user_input, question_count, selected_category)

            # 2) Call your existing questionnaire generator
            out_obj = generate_questionnaire(
                req_path="questionaire_requirements.json",
                db_path="convert_data.json"
            )

            # 3) Take ordered_questions and pass them to the template
            generated_questions = out_obj.get("ordered_questions", [])

    return render_template(
        "index.html",
        user_input=user_input,
        question_count=question_count,
        selected_category=selected_category,
        categories=AVAILABLE_CATEGORIES,  # Pass available categories to the template
        generated_questions=generated_questions,
    )


@app.route("/download/json")
def download_json():
    """
    Download the latest generated questionnaire as JSON.
    """
    if not os.path.exists(OUTPUT_JSON_PATH):
        abort(404, description="Questionnaire JSON not found. Please generate it first.")
    return send_file(
        OUTPUT_JSON_PATH,
        as_attachment=True,
        download_name="questionnaire.json",
        mimetype="application/json",
    )


@app.route("/download/txt")
def download_txt():
    """
    Download the latest generated questionnaire as a human-readable TXT file.
    """
    if not os.path.exists(OUTPUT_TXT_PATH):
        abort(404, description="Questionnaire TXT not found. Please generate it first.")
    return send_file(
        OUTPUT_TXT_PATH,
        as_attachment=True,
        download_name="questionnaire.txt",
        mimetype="text/plain",
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
