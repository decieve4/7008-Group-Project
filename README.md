# Project 8: A question bank generator and survey system

Customer survey question bank generator - upload questions and generate customized questionnaires.

## Test Steps:
1. **Start the service**:  
   `python app.py`
2. **Access via browser**:  
   `http://localhost:5000`
3. **Enter questionnaire requirements**:  
   - requirements(like topic)
   - Number of questions
   - Other relevant parameters
4. **Click the "Generate Questionnaire" button**
5. **Verify page display**:  
   - Check if the generated questionnaire is displayed correctly
6. **Download results**:  
   - Download in JSON format  
   - Download in TXT format
7. **Validate output files**:  
   - Check if `output_questionaire.json` is correctly generated
### Independent Module Testing
- **Data Crawling Module**: Run `Q3-code.py` to test data crawling functionality
- **Questionnaire Generation Module**: Run `python Q5_v2.py` to test command-line generation (requires configuration of `questionaire_requirements.json`)
- **Visualization Module**: Run `plots.py` to generate data analysis charts
> **Note**: Ensure all required Python dependencies are installed before running.

## Files
### Code Documentation
- `Q3-code.py` - Travel Questionnaire Crawler System: Automatically collects, parses, categorizes, and stores travel survey questions.
- `convert_data.py`- Data Processing
- `survey_database.py` - Database Management
- `Part4.py` - Survey Intelligence Engine: Clean, Analyze, Generate
- `Q5_v2.py` - Questionnaire Auto-Generator: Creates structured surveys based on requirements
- `app.py` - A Flask based web questionnaire generation tool that automatically generates customized survey questionnaires based on user needs.
- `Visualization.py`  - Data Visualization Module: Multi-dimensional Visualization for Travel Questionnaire Dataset
- `plots.py` - Visualization Pipeline Script
- `keywords_classification.py` - Intelligent Classification System Based on Keyword Matching
### Data Files
- `functional_oo_diagram.mmd` - System functions OO diagram
- `convert_data.json` - Converted survey data (6000 questions)
- `survey_data.json` - Test data (only 5 questions)
- `output_questionaire.json` - output questionaire based on users requirement
- `output_questionaire.txt` - output questionaire based on users requirement
- `questionaire_requirements.json` - Questionnaire generator input configuration.
- `questionaire_requirements_default.json` - Questionnaire generator's fallback/default configuration file.
- `questions_reclassified/v2.json` - Reclassified travel survey questions dataset

## Class Description 
### Part 1: Data Crawling
- `FinalSurveyCrawler`(Class)- Crawl multi-platform travel questionnaires, parse questions/options, and categorize stored data.
### Part 2: Data PreProcessing
- `convert_question_data`(Core Function) - Standardized Conversion and Basic Data Cleaning of Tourism Survey Data
- `SurveyDatabase`(Class) - Manages system data (questions/questionnaires/users); supports questionnaire creation, question association, and data statistics; persists data in JSON format.
### Part 3: Data Deep Processing & Intelligent Generation
- `clean_question_text`(Core Function) - Cleans question text
- `normalize_for_dedup`(Core Function) - Standardizes text (synonym replacement, word segmentation/lemmatization) for deduplication and similarity matching
- `calculate_difficulty`(Core Function) - Calculates difficulty score (1-5) based on question type, length, number of options, and keywords
- `load_and_process_data`(Core Function) - Data processing pipeline (loading → cleaning → deduplication → difficulty scoring → Chinese/English splitting → saving)
- `generate_survey`(Core Function) - Generates personalized Chinese/English surveys via TF-IDF cosine similarity matching based on user requirements
### Part 4 : Questionnaire Generation Based on Requirements
- `generate_questionnaire`(Core Function) - Load user requirements, filter/rank questions from the question bank (TF-IDF for English/keyword scoring for Chinese), select non-duplicate questions, and generate structured/human-readable questionnaire outputs.
- `Flask Web Interface`(Application) - Web-based interactive interface for questionnaire generation (supports requirement input, question count/category selection, and JSON/TXT download of results).
### Part 5 : Data Visualization
- `DataVisualization`(Class) - data visualization  for survey questions: generate various charts (distribution/relation/correlation) to analyze difficulty, category, question type and text length of questions.
- `Visualization Execution Script` - Batch call DataVisualization methods to generate multi-dimensional visual reports for questions.
