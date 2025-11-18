import json
import os
from typing import List, Dict, Any, Optional

class SurveyDatabase:
    def __init__(self, data_file: str = "convert_data.json"):
        self.data_file = data_file
        self.data = self._load_data()
    
    def _load_data(self) -> Dict[str, Any]:
        """Load data from JSON file"""
        try:
            with open(self.data_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            # Create base data structure if file doesn't exist
            base_data = {
                "questions": [],
                "questionnaires": [],
                "users": [],
                "survey_results": [],
                "analysis_results": []
            }
            self._save_data(base_data)
            return base_data
    
    def _save_data(self, data: Dict[str, Any] = None):
        """Save data to JSON file"""
        if data is None:
            data = self.data
        
        with open(self.data_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    # Question management methods
    def get_all_questions(self) -> List[Dict]:
        """Get all questions"""
        return self.data.get("questions", [])
    
    def get_question_by_id(self, question_id: int) -> Optional[Dict]:
        """Get question by ID"""
        for question in self.data.get("questions", []):
            if question["id"] == question_id:
                return question
        return None
    
    def get_questions_by_category(self, category: str) -> List[Dict]:
        """Get questions by category"""
        return [q for q in self.data.get("questions", []) 
                if q.get("category") == category]
    
    def add_question(self, question_data: Dict) -> int:
        """Add new question"""
        questions = self.data.get("questions", [])
        
        # Generate new ID
        new_id = max([q["id"] for q in questions], default=0) + 1
        question_data["id"] = new_id
        question_data["usage_count"] = 0
        
        questions.append(question_data)
        self.data["questions"] = questions
        self._save_data()
        return new_id
    
    # Questionnaire management methods
    def get_all_questionnaires(self) -> List[Dict]:
        """Get all questionnaires"""
        return self.data.get("questionnaires", [])
    
    def get_questionnaire_by_id(self, questionnaire_id: int) -> Optional[Dict]:
        """Get questionnaire by ID"""
        for questionnaire in self.data.get("questionnaires", []):
            if questionnaire["id"] == questionnaire_id:
                return questionnaire
        return None
    
    def create_questionnaire(self, title: str, description: str, 
                           target_audience: str, requirements: Dict) -> int:
        """Create new questionnaire"""
        questionnaires = self.data.get("questionnaires", [])
        
        new_id = max([q["id"] for q in questionnaires], default=0) + 1
        
        questionnaire = {
            "id": new_id,
            "title": title,
            "description": description,
            "target_audience": target_audience,
            "requirements": requirements,
            "question_ids": [],
            "status": "draft"
        }
        
        questionnaires.append(questionnaire)
        self.data["questionnaires"] = questionnaires
        self._save_data()
        return new_id
    
    def add_question_to_questionnaire(self, questionnaire_id: int, question_id: int) -> bool:
        """Add question to questionnaire"""
        questionnaire = self.get_questionnaire_by_id(questionnaire_id)
        question = self.get_question_by_id(question_id)
        
        if not questionnaire or not question:
            return False
        
        if question_id not in questionnaire["question_ids"]:
            questionnaire["question_ids"].append(question_id)
            # Update question usage count
            question["usage_count"] = question.get("usage_count", 0) + 1
            self._save_data()
        
        return True
    
    # User management methods
    def get_all_users(self) -> List[Dict]:
        """Get all users"""
        return self.data.get("users", [])
    
    def get_user_by_id(self, user_id: int) -> Optional[Dict]:
        """Get user by ID"""
        for user in self.data.get("users", []):
            if user["id"] == user_id:
                return user
        return None
    
    # Data statistics methods
    def get_statistics(self) -> Dict[str, Any]:
        """Get data statistics"""
        questions = self.data.get("questions", [])
        questionnaires = self.data.get("questionnaires", [])
        users = self.data.get("users", [])
        
        # Count questions by category
        categories = {}
        for question in questions:
            category = question.get("category", "Uncategorized")
            categories[category] = categories.get(category, 0) + 1
        
        # Count questions by type
        question_types = {}
        for question in questions:
            q_type = question.get("question_type", "Unknown")
            question_types[q_type] = question_types.get(q_type, 0) + 1
        
        return {
            "total_questions": len(questions),
            "total_questionnaires": len(questionnaires),
            "total_users": len(users),
            "categories_distribution": categories,
            "question_types_distribution": question_types

        }
