# ==========================================
# Project 8 - Question Bank & Survey System
# Database Simulation
# ==========================================

from datetime import datetime
from typing import List, Dict


# ------------------------------
# 1. 用户类 (UserProfile)
# ------------------------------
class User:
    """Represents a user profile in the survey system."""

    def __init__(self, user_id: int, name: str, age: int, gender: str, preferences: List[str]):
        self.user_id = user_id
        self.name = name
        self.age = age
        self.gender = gender
        self.preferences = preferences
        self.response_history = []  # list of Response objects
        self.created_at = datetime.now()

    def add_response(self, response):
        self.response_history.append(response)

    def __repr__(self):
        return f"User({self.name}, {self.age}, {self.gender})"


# ------------------------------
# 2. 题目类 (Question)
# ------------------------------
class Question:
    """Represents a single question in the question bank."""

    def __init__(self, question_id: int, content: str, category: str, difficulty: str):
        self.question_id = question_id
        self.content = content
        self.category = category
        self.difficulty = difficulty
        self.created_at = datetime.now()

    def get_text(self):
        return self.content

    def __repr__(self):
        return f"Question({self.category}, {self.difficulty})"


# ------------------------------
# 3. 问卷类 (Survey)
# ------------------------------
class Survey:
    """Represents a survey composed of multiple questions."""

    def __init__(self, survey_id: int, title: str, description: str, created_by: User):
        self.survey_id = survey_id
        self.title = title
        self.description = description
        self.created_by = created_by
        self.questions = []  # list of Question objects
        self.created_at = datetime.now()

    def add_question(self, question: Question):
        self.questions.append(question)

    def list_questions(self):
        return [q.content for q in self.questions]

    def __repr__(self):
        return f"Survey({self.title}, {len(self.questions)} questions)"


# ------------------------------
# 4. 用户答题记录 (Response)
# ------------------------------
class Response:
    """Represents a user's answer to a specific question."""

    def __init__(self, user: User, survey: Survey, question: Question, answer: str):
        self.user = user
        self.survey = survey
        self.question = question
        self.answer = answer
        self.timestamp = datetime.now()

    def __repr__(self):
        return f"Response(User={self.user.name}, Q={self.question.question_id}, A={self.answer})"


# ------------------------------
# 5. 模拟数据库 (Database)
# ------------------------------
class Database:
    """Simulates the storage of users, questions, surveys, and responses."""

    def __init__(self):
        self.users: Dict[int, User] = {}
        self.questions: Dict[int, Question] = {}
        self.surveys: Dict[int, Survey] = {}
        self.responses: List[Response] = []

    # --- User operations ---
    def add_user(self, user: User):
        self.users[user.user_id] = user

    # --- Question operations ---
    def add_question(self, question: Question):
        self.questions[question.question_id] = question

    # --- Survey operations ---
    def add_survey(self, survey: Survey):
        self.surveys[survey.survey_id] = survey

    # --- Response operations ---
    def add_response(self, response: Response):
        self.responses.append(response)
        response.user.add_response(response)

    # --- Helper functions ---
    def get_user_responses(self, user_id: int):
        return [r for r in self.responses if r.user.user_id == user_id]

    def get_survey_questions(self, survey_id: int):
        survey = self.surveys.get(survey_id)
        return survey.questions if survey else []

    def show_summary(self):
        print(f"Users: {len(self.users)}, Questions: {len(self.questions)}, Surveys: {len(self.surveys)}")
        print(f"Responses recorded: {len(self.responses)}")


# ------------------------------
# 6. 测试示例 (Database Simulation)
# ------------------------------
if __name__ == "__main__":
    # 创建数据库实例
    db = Database()

    # 添加用户
    alice = User(1, "Alice", 25, "F", ["travel", "food"])
    bob = User(2, "Bob", 30, "M", ["tech", "sports"])
    db.add_user(alice)
    db.add_user(bob)

    # 添加题目
    q1 = Question(101, "What is your favorite travel destination?", "travel", "easy")
    q2 = Question(102, "How often do you travel abroad?", "travel", "medium")
    q3 = Question(103, "Rate your satisfaction with online booking apps.", "tech", "medium")
    db.add_question(q1)
    db.add_question(q2)
    db.add_question(q3)

    # 创建问卷
    survey1 = Survey(201, "Travel Experience Survey", "A survey about travel preferences.", alice)
    survey1.add_question(q1)
    survey1.add_question(q2)
    db.add_survey(survey1)

    # 用户作答
    r1 = Response(alice, survey1, q1, "Japan")
    r2 = Response(alice, survey1, q2, "3 times per year")
    r3 = Response(bob, survey1, q1, "France")

    db.add_response(r1)
    db.add_response(r2)
    db.add_response(r3)

    # 打印结果
    db.show_summary()
    print("Survey Questions:", survey1.list_questions())
    print("Alice's Responses:", db.get_user_responses(1))
