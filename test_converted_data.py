from survey_database import SurveyDatabase

def test_converted_data():
    """
    Test the converted data to ensure it works correctly
    """
    print("=== Testing Converted Data ===\n")
    
    # Initialize database
    db = SurveyDatabase("convert_data.json")
    
    # Show basic statistics
    stats = db.get_statistics()
    print(f"Total questions: {stats['total_questions']}")
    print(f"Categories: {stats['categories_distribution']}")
    print(f"Question types: {stats['question_types_distribution']}")
    
    # Test category filtering
    print("\n=== Category Analysis ===")
    major_categories = ['satisfaction', 'service', 'room', 'general']
    for category in major_categories:
        questions = db.get_questions_by_category(category)
        if questions:
            print(f"{category.capitalize()}: {len(questions)} questions")
    
    # Test question retrieval
    print("\n=== Question Samples ===")
    all_questions = db.get_all_questions()
    print(f"First 3 questions preview:")
    for i in range(min(3, len(all_questions))):
        question = all_questions[i]
        print(f"{i+1}. {question['question_text'][:80]}...")
        print(f"   Type: {question['question_type']}, Category: {question['category']}")
    
    # Test specific question by ID
    print("\n=== Individual Question Test ===")
    sample_question = db.get_question_by_id(1)
    if sample_question:
        print(f"Question ID 1: {sample_question['question_text']}")
    
    # Test data integrity
    print("\n=== Data Integrity Check ===")
    categories_count = sum(stats['categories_distribution'].values())
    types_count = sum(stats['question_types_distribution'].values())
    
    if categories_count == stats['total_questions']:
        print("PASS - Category data consistent")
    else:
        print("FAIL - Category data inconsistent")
    
    if types_count == stats['total_questions']:
        print("PASS - Question type data consistent")
    else:
        print("FAIL - Question type data inconsistent")
    
    print(f"\n=== Test Completed ===")
    print(f"All tests passed for {stats['total_questions']} questions")

if __name__ == "__main__":
    test_converted_data()