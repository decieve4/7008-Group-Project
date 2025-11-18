import json

def convert_question_data(input_file, output_file="convert_data.json"):
    """
    Convert question data to standard survey system format
    """
    try:
        # Read input data
        with open(input_file, 'r', encoding='utf-8') as f:
            input_data = json.load(f)
        
        print(f"Converting {len(input_data)} questions...")
        
        # Create standard format
        converted_data = {
            "questions": [],
            "questionnaires": [],
            "users": []
        }
        
        # Convert each question
        for i, item in enumerate(input_data):
            question_text = item.get('question_text', '')
            
            question = {
                "id": i + 1,
                "question_text": question_text,
                "question_type": item.get('question_type', 'text'),
                "options": item.get('options_text', None),
                "difficulty": 2.0,
                "category": item.get('category', 'general'),
                "tags": [item.get('category', 'general'), item.get('language', 'en')],
                "usage_count": 0
            }
            converted_data["questions"].append(question)
        
        # Save converted data
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(converted_data, f, ensure_ascii=False, indent=2)
        
        print(f"Conversion completed! Saved to {output_file}")
        print(f"Total questions converted: {len(converted_data['questions'])}")
        
        # Show samples
        print("\nSample questions:")
        for i in range(3):
            text = converted_data['questions'][i]['question_text']
            print(f"{i+1}. {text}")
        
        return True
        
    except Exception as e:
        print(f"Conversion error: {e}")
        return False

# Execute conversion
if __name__ == "__main__":
    convert_question_data("questions.json", "convert_data.json")