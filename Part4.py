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
# 0. AUTO-INSTALLER & SETUP
# ==========================================
def install_and_import_spacy():
    try:
        import spacy
        try:
            spacy.load("en_core_web_sm")
        except OSError:
            print("Downloading language model 'en_core_web_sm'...")
            subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        return spacy
    except ImportError:
        print("Installing library 'spacy'...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "spacy"])
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        import spacy
        return spacy

warnings.filterwarnings("ignore")
spacy_nlp = install_and_import_spacy() 
nlp_en = spacy_nlp.load("en_core_web_sm")

import pandas as pd
import numpy as np
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# 1. CLEANING FUNCTIONS 
# ==========================================

def clean_question_text(text):
    if not isinstance(text, str): return ""
    
    # A. Standard prefix cleaning
    text = re.sub(r'^\s*ask\s+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^\s*(?:Q\s*)?\d+[\.:\s]\s*', '', text, flags=re.IGNORECASE)

    # B. Start with first valid character
    start_match = re.search(r'[a-zA-Z\u4e00-\u9fa5]', text)
    if start_match:
        text = text[start_match.start():]

    # C. Strict Question Mark Filter (Depends on whether strengthen the cleansing requirement)
    if not re.search(r'[?？]\s*$', text):
        return "" 

    return text.strip()

def normalize_for_dedup(text, lang):
    if not isinstance(text, str): return ""
    text = text.lower()
    
    synonym_map = {
        "hotel": "accommodation", "hotels": "accommodation",
        "inn": "accommodation", "resort": "accommodation",
        "surveys": "survey", "travels": "travel", "service": "hospitality",
        "services": "hospitalitys", "hotel": "property",
        "trip": "travel", "journey": "travel"
    }
    
    if lang == 'en':
        doc = nlp_en(text)
        tokens = []
        for token in doc:
            if token.is_punct or token.is_stop: continue
            word = token.lemma_ 
            word = synonym_map.get(word, word)
            tokens.append(word)
        return " ".join(tokens)

    elif lang == 'zh':
        text = re.sub(r'[^\w\s\u4e00-\u9fa5]', '', text)
        words = jieba.lcut(text)
        zh_map = {"宾馆": "酒店", "饭店": "酒店", "住宿": "酒店", "游览": "旅游"}
        return " ".join([zh_map.get(w, w) for w in words if w.strip()])
    
    return text

# ==========================================
# 2. DIFFICULTY SCORING LOGIC
# ==========================================

def calculate_difficulty(row):
    score = 1 
    
    text = str(row.get('question_text', ''))
    q_type = str(row.get('question_type', '')).lower()
    options = str(row.get('options_text', ''))
    
    # --- CHANGED: Use 'language' instead of 'detected_lang' ---
    lang = row.get('language', 'en') 

    # A. TYPE FACTOR
    if 'open_ended' in q_type:
        score += 2
    elif 'multiple_choice' in q_type or 'single_choice' in q_type:
        score += 1

    # B. LENGTH FACTOR
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
        option_count = options.count('/') + 1
        if option_count > 6: score += 1

    # D. WORDING FACTOR
    text_lower = text.lower()
    en_hard_words = ["describe", "explain", "comprehensive", "evaluate", "perspective", "why"]
    zh_hard_words = ["描述", "解释", "详细", "评估", "看法", "为什么"]
    
    if lang == 'en':
        if any(w in text_lower for w in en_hard_words): score += 1
    else:
        if any(w in text_lower for w in zh_hard_words): score += 1

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
        type_counts = df['question_type'].fillna('Unknown').value_counts()
        for q_type, count in type_counts.items():
            print(f"    - {q_type}: {count}")
    else:
        print("    (No data)")

    # [2] Topic Coverage Analysis
    print(f"[2] Topic Coverage:")
    
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

    topic_counts = {k: 0 for k in topic_keywords}
    
    for text in df['question_text']:
        text_lower = str(text).lower()
        for topic, keywords in topic_keywords.items():
            if any(k in text_lower for k in keywords):
                topic_counts[topic] += 1

    for topic, count in topic_counts.items():
        print(f"    - {topic}: {count}")

    # [3] Difficulty Analysis
    print(f"[3] Difficulty Level Distribution (1=Easy, 5=Hard):")
    difficulty_counts = df['difficulty_score'].value_counts().sort_index()
    
    for i in range(1, 6):
        count = difficulty_counts.get(i, 0)
        print(f"    - Level {i}: {count}")
    print("-" * 40)

# ==========================================
# 4. LOADING & PROCESSING PIPELINE
# ==========================================

def load_and_process_data(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, dict) and 'fullContent' in data:
            df = pd.DataFrame(data['fullContent'])
        else:
            df = pd.DataFrame(data)

        raw_count = len(df)

        # 1. Clean text
        df['question_text'] = df['question_text'].apply(clean_question_text)
        # Remove rows that became empty after cleaning
        df = df[df['question_text'].str.len() > 2].reset_index(drop=True)

        # 3. Deduplicate
        print("Normalizing for duplicate detection...")
        df['dedup_key'] = df.apply(
            # Use x['language'] directly
            lambda x: normalize_for_dedup(str(x['question_text']) + " " + str(x.get('options_text', '')), x['language']), 
            axis=1
        )
        df.drop_duplicates(subset=['dedup_key'], keep='first', inplace=True)
        
        # 4. Calculate Difficulty 
        df['difficulty_score'] = df.apply(calculate_difficulty, axis=1)
        
        # ============================================================
        # SAVE COMBINED CLEANED DATA TO JSON
        # ============================================================
        print("Saving cleaned dataset (cleaned_questions_all.json)...")
        df.to_json('./travel_survey_final/cleaned_questions_all.json', orient='records', force_ascii=False, indent=4)
        # ============================================================

        # 5. Split into English and Chinese tables using 'language' column
        df_en = df[df['language'] == 'en'].copy().reset_index(drop=True)
        df_zh = df[df['language'] == 'zh'].copy().reset_index(drop=True)
        
        # --- DATA REPORT ---
        print("\n" + "="*50)
        print("          DATA CLEANING REPORT")
        print("="*50)
        print(f"1. Raw Questions Loaded:     {raw_count}")
        print(f"2. Final Valid Questions:    {len(df)}")
        print(f"   > English Pool:           {len(df_en)}")
        print(f"   > Chinese Pool:           {len(df_zh)}")
        print("="*50)

        if not df_en.empty: analyze_dataset_content(df_en, 'en')
        if not df_zh.empty: analyze_dataset_content(df_zh, 'zh')

        return df_en, df_zh

    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame(), pd.DataFrame()

# ==========================================
# 5. GENERATION ENGINE 
# ==========================================

def generate_survey(df, lang, target_count=10):
    # Set the input for the normalized text
    df['model_input'] = df['dedup_key']

    # Initialize the TF-IDF Vectorizer
    if lang == 'en':
        vectorizer = TfidfVectorizer(stop_words='english')
    else:
        zh_stops = ["的", "了", "是", "我", "你", "在", "和", "有", "去", "吗", "我们", "什么"]
        vectorizer = TfidfVectorizer(stop_words=zh_stops)

    try:
        tfidf_matrix = vectorizer.fit_transform(df['model_input'])
    except ValueError:
        print("Error: Not enough text data.")
        return

    while True:
        print(f"\n--- Generate {lang.upper()} Survey (Target: {target_count}) ")
        print("Enter requirement (or type 'exit'):")
        
        req = input(">> ")
        
        if req.strip().lower() == 'exit':
            print("Terminating program.")
            sys.exit(0)
        
        if not req.strip(): continue

        # Process user input 
        processed_req = normalize_for_dedup(req, lang)
        req_vector = vectorizer.transform([processed_req])
        
        # Calculate similarity scores
        cosine_sim = cosine_similarity(req_vector, tfidf_matrix).flatten()
        
        # Sort results from highest score to lowest
        sorted_indices = cosine_sim.argsort()[::-1]
        
        print(f"\nResults for: '{req}'\n" + "-"*40)
        
        count = 0
        seen_text = set()
        
        for idx in sorted_indices:
            if count >= target_count: break
            
            score = cosine_sim[idx]
            
            # Threshold for matching
            if score > 0.05: 
                q_text = df.iloc[idx]['question_text']
                
                if q_text not in seen_text:
                    # Show Question + Difficulty Score
                    diff_score = df.iloc[idx]['difficulty_score']
                    # Create star rating string (e.g. ★★★☆☆)
                    diff_label = "★" * diff_score + "☆" * (5 - diff_score)

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
# MAIN EXECUTION BLOCK
# ==========================================

if __name__ == "__main__":
    # Path of input file 
    file_path = r"./travel_survey_final/questions.json"
    
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











