# %%
import json
import pandas as pd
import numpy as np
import jieba  
import re
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# 0. AUTO-INSTALLER 
# ==========================================
def install_and_import_spacy():

    try:
        import spacy
        # Try loading the model to ensure it exists
        try:
            spacy.load("en_core_web_sm")
        except OSError:
            print("Spacy found, but 'en_core_web_sm' model missing. Downloading...")
            subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
            print("Model downloaded.")
        return spacy
    except ImportError:
        print("Spacy library not found. Installing now...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "spacy"])
            subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
            print("Installation complete.")
            import spacy
            return spacy
        except Exception as e:
            print(f"Automatic installation failed: {e}")
            print("Please run these commands in your terminal/command prompt:")
            print("  pip install spacy")
            print("  python -m spacy download en_core_web_sm")
            sys.exit(1)

# Perform the check/install BEFORE importing
spacy_nlp = install_and_import_spacy() 
# load the model
nlp_en = spacy_nlp.load("en_core_web_sm")

# ==========================================
# 1. CLEANING FUNCTIONS 
# ==========================================

def clean_question_text(text):

    if not isinstance(text, str): return ""
    
    # A. Standard prefix cleaning
    # Removes "Ask " at the beginning (case insensitive)
    text = re.sub(r'^\s*ask\s+', '', text, flags=re.IGNORECASE)
    # Removes numbering like "Q1.", "1.", "1 " at the beginning
    text = re.sub(r'^\s*(?:Q\s*)?\d+[\.:\s]\s*', '', text, flags=re.IGNORECASE)

    # B. Start with first valid character
    # This locates the first English letter or Chinese character
    # It cuts off any weird symbols at the start like "..." or "-"
    start_match = re.search(r'[a-zA-Z\u4e00-\u9fa5]', text)
    if start_match:
        text = text[start_match.start():]
    
    return text.strip()

def normalize_for_dedup(text, lang):
    """
    Standardizes text for deduplication (lemmatization + synonym mapping).
    """
    if not isinstance(text, str): return ""
    
    text = text.lower()
    
    # Synonym Map: Defining words that mean the same thing
    synonym_map = {
        "hotel": "accommodation", "hotels": "accommodation",
        "inn": "accommodation", "resort": "accommodation",
        "surveys": "survey", "travels": "travel",
        "trip": "travel", "journey": "travel"
    }
    
    if lang == 'en':
        # Process English text with Spacy
        doc = nlp_en(text)
        tokens = []
        for token in doc:
            # Skip punctuation and common words (stopwords) like "the", "is"
            if token.is_punct or token.is_stop: continue
            
            # Get the root word (e.g., "Traveling" -> "Travel")
            word = token.lemma_ 
            
            # Check if this word is in our synonym list and swap it
            word = synonym_map.get(word, word)
            tokens.append(word)
        return " ".join(tokens)

    elif lang == 'zh':
        # Process Chinese text
        # Remove non-Chinese characters
        text = re.sub(r'[^\w\s\u4e00-\u9fa5]', '', text)
        # Use Jieba to cut the sentence into words
        words = jieba.lcut(text)
        # Chinese Synonym Map
        zh_map = {"宾馆": "酒店", "饭店": "酒店", "住宿": "酒店", "游览": "旅游"}
        return " ".join([zh_map.get(w, w) for w in words if w.strip()])
    
    return text

# ==========================================
# 2. DIFFICULTY SCORING LOGIC
# ==========================================

def calculate_difficulty(row):

    score = 1 # Start with base score 1
    
    text = str(row.get('question_text', ''))
    q_type = str(row.get('question_type', '')).lower()
    options = str(row.get('options_text', ''))
    lang = row.get('detected_lang', 'en')

    # A. TYPE FACTOR
    # Open-ended questions are harder because users have to type an answer
    if 'open_ended' in q_type:
        score += 2
    # Multiple choice is slightly harder than simple Yes/No (Base 1)
    elif 'multiple_choice' in q_type or 'single_choice' in q_type:
        score += 1

    # B. LENGTH FACTOR 
    # Longer questions take more effort to read
    if lang == 'en':
        word_count = len(text.split())
        if word_count > 30: score += 2   
        elif word_count > 15: score += 1 
    else: 
        char_count = len(text)
        if char_count > 50: score += 2
        elif char_count > 20: score += 1

    # C. OPTIONS FACTOR
    if options:
        # estimate number of options by counting separators '/'
        option_count = options.count('/') + 1
        if option_count > 6: score += 1

    # D. WORDING FACTOR
    # Check for "Hard" keywords that require complex thinking
    text_lower = text.lower()
    
    en_hard_words = ["describe", "explain", "comprehensive", "evaluate", "perspective", "why"]
    zh_hard_words = ["描述", "解释", "详细", "评估", "看法", "为什么"]
    
    if lang == 'en':
        if any(w in text_lower for w in en_hard_words): score += 1
    else:
        if any(w in text_lower for w in zh_hard_words): score += 1

    # Ensure the score stays between 1 and 5
    final_score = max(1, min(5, score))
    return final_score

# ==========================================
# 3. ANALYSIS FUNCTIONS 
# ==========================================

def analyze_dataset_content(df, lang):
    print(f"\n>>> ANALYSIS FOR {lang.upper()} DATASET <<<")
    
    # [1] Question Types Analysis
    print(f"[1] Question Type Distribution:")
    if 'question_type' in df.columns:
        # Count unique values in 'question_type' column
        type_counts = df['question_type'].fillna('Unknown').value_counts()
        for q_type, count in type_counts.items():
            print(f"    - {q_type}: {count}")
    else:
        print("    (No data)")

    # [2] Topic Coverage Analysis
    print(f"[2] Topic Coverage (Overlapping):")
    
    # Define keywords for different topics
    if lang == 'en':
        topic_keywords = {
            "Hotel/Accommodation": ["hotel", "accommodation", "room", "stay", "inn"],
            "Travel/General":      ["travel", "trip", "journey", "tour", "tourism"],
            "Flight/Transport":    ["flight", "airline", "plane", "transport", "bus"],
            "Food/Dining":         ["food", "meal", "dining", "restaurant", "eat"],
            "Service/Satisfaction":["service", "staff", "satisfaction", "quality"]
        }
    else:
        topic_keywords = {
            "Hotel/Accommodation": ["酒店", "住宿", "房间", "宾馆", "饭店"],
            "Travel/General":      ["旅游", "旅行", "行程", "度假"],
            "Flight/Transport":    ["航班", "飞机", "交通", "机场"],
            "Food/Dining":         ["餐饮", "食物", "吃饭", "餐厅"],
            "Service/Satisfaction":["服务", "满意", "推荐", "态度"]
        }

    # Initialize counters to 0
    topic_counts = {k: 0 for k in topic_keywords}
    
    # Loop through every question
    for text in df['question_text']:
        text_lower = str(text).lower()
        # Check if the question belongs to a topic (can be multiple topics)
        for topic, keywords in topic_keywords.items():
            if any(k in text_lower for k in keywords):
                topic_counts[topic] += 1

    # Print results
    for topic, count in topic_counts.items():
        print(f"    - {topic}: {count}")

    # [3] Difficulty Analysis
    print(f"[3] Difficulty Level Distribution (1=Easy, 5=Hard):")
    
    # Count how many questions are Level 1, Level 2, etc.
    difficulty_counts = df['difficulty_score'].value_counts().sort_index()
    
    # Loop from 1 to 5 
    for i in range(1, 6):
        count = difficulty_counts.get(i, 0)
        print(f"    - Level {i}: {count}")
    print("-" * 40)

# ==========================================
# 4. LOADING & PROCESSING PIPELINE
# ==========================================

def load_and_process_data(filepath):
    try:
        # Load JSON file
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Convert to DataFrame (Table format)
        if isinstance(data, dict) and 'fullContent' in data:
            df = pd.DataFrame(data['fullContent'])
        else:
            df = pd.DataFrame(data)

        raw_count = len(df)

        # 1. Clean text
        # Apply the cleaning function to every row
        df['question_text'] = df['question_text'].apply(clean_question_text)
        # Remove rows that are too short (empty or junk)
        df = df[df['question_text'].str.len() > 2].reset_index(drop=True)

        # 2. Detect Language
        def detect_lang(text):
            # Check for Chinese characters using Unicode range
            if re.search(r'[\u4e00-\u9fa5]', str(text)): return 'zh'
            return 'en'
        df['detected_lang'] = df['question_text'].apply(detect_lang)

        # 3. Deduplicate
        print("Normalizing for duplicate detection...")
        # Create a special 'key' that looks effectively the same for similar questions
        df['dedup_key'] = df.apply(
            lambda x: normalize_for_dedup(str(x['question_text']) + " " + str(x.get('options_text', '')), x['detected_lang']), 
            axis=1
        )
        # Drop exact duplicates based on this key
        df.drop_duplicates(subset=['dedup_key'], keep='first', inplace=True)
        
        # 4. Calculate Difficulty 
        # We calculate this BEFORE splitting so the logic applies to all data
        df['difficulty_score'] = df.apply(calculate_difficulty, axis=1)

        # 5. Split into English and Chinese tables
        df_en = df[df['detected_lang'] == 'en'].copy().reset_index(drop=True)
        df_zh = df[df['detected_lang'] == 'zh'].copy().reset_index(drop=True)
        
        # --- DATA REPORT ---
        print("\n" + "="*50)
        print("          DATA CLEANING REPORT")
        print("="*50)
        print(f"1. Raw Questions Loaded:     {raw_count}")
        print(f"2. Final Valid Questions:    {len(df)}")
        print(f"   > English Pool:           {len(df_en)}")
        print(f"   > Chinese Pool:           {len(df_zh)}")
        print("="*50)

        # --- RUN ANALYSIS ---
        if not df_en.empty: analyze_dataset_content(df_en, 'en')
        if not df_zh.empty: analyze_dataset_content(df_zh, 'zh')

        return df_en, df_zh

    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame(), pd.DataFrame()

# ==========================================
# 5. GENERATION ENGINE (Main Logic)
# ==========================================

def generate_survey(df, lang, target_count=20):
    # Set the input for the normalized text
    df['model_input'] = df['dedup_key']

    # Initialize the TF-IDF Vectorizer
    if lang == 'en':
        vectorizer = TfidfVectorizer(stop_words='english')
    else:
        # Chinese requires a custom stopword list
        zh_stops = ["的", "了", "是", "我", "你", "在", "和", "有", "去", "吗", "我们", "什么"]
        vectorizer = TfidfVectorizer(stop_words=zh_stops)

    try:
        # Train the model on our data
        tfidf_matrix = vectorizer.fit_transform(df['model_input'])
    except ValueError:
        print("Error: Not enough text data.")
        return

    # Loop to keep asking for user input
    while True:
        print(f"\n--- Generate {lang.upper()} Survey (Target: {target_count}) ")
        print("Enter requirement (or type 'exit'):")
        
        req = input(">> ")
        
        # Check for Exit command
        if req.strip().lower() == 'exit':
            print("Terminating program.")
            sys.exit(0)
        
        if not req.strip(): continue

        # Process user input (Same cleaning as dataset)
        processed_req = normalize_for_dedup(req, lang)
        
        # Convert user input to numbers
        req_vector = vectorizer.transform([processed_req])
        
        # Calculate similarity scores
        cosine_sim = cosine_similarity(req_vector, tfidf_matrix).flatten()
        
        # Sort results from highest score to lowest
        sorted_indices = cosine_sim.argsort()[::-1]
        
        print(f"\nResults for: '{req}'\n" + "-"*40)
        
        count = 0
        seen_text = set()
        
        # Iterate through the best matches
        for idx in sorted_indices:
            if count >= target_count: break
            
            score = cosine_sim[idx]
            # Only show results with > 5% similarity
            if score > 0.05: 
                q_text = df.iloc[idx]['question_text']
                
                # Ensure do not show the same question twice in this list
                if q_text not in seen_text:
                    # Show Question + Difficulty Score
                    diff_score = df.iloc[idx]['difficulty_score']
                    
                    print(f"{count + 1}. [Match: {int(score*100)}%] [Diff: {diff_score} {diff_label}]")
                    print(f"    {q_text}")
                    
                    # Show Options if they exist
                    opts = df.iloc[idx].get('options_text')
                    if pd.notna(opts) and str(opts).strip() != "":
                        print(f"    (Options: {opts})")
                    
                    print("-" * 40)
                    seen_text.add(q_text)
                    count += 1
                    
        if count == 0: print("No matches found.")

# ==========================================
# MAIN EXECUTION BLOCK (Just for demo what the code doing)
# ==========================================

if __name__ == "__main__":
    # Path of input file
    file_path = r"C:\Users\User\Desktop\questions.json"
    
    # Step 1: Load and Process Data
    english_df, chinese_df = load_and_process_data(file_path)
    
    if english_df.empty and chinese_df.empty:
        print("No data loaded. Exiting.")
        sys.exit(1)
    
    # Step 2: User Menu Loop
    while True:
        print("\n=========================================")
        print("   SURVEY GENERATOR SYSTEM")
        print("=========================================")
        print("1. Generate English Survey")
        print("2. Generate Chinese Survey")
        print("Type 'exit' to close program.")
        
        choice = input("Select: ").strip().lower()
        
        if choice == '1':
            if not english_df.empty: generate_survey(english_df, 'en')
            else: print("No English questions.")
        elif choice == '2':
            if not chinese_df.empty: generate_survey(chinese_df, 'zh')
            else: print("No Chinese questions.")
        elif choice == 'exit':
            print("Exiting.")
            sys.exit(0)
        else:
            print("Invalid selection.")


