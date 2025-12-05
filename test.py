import csv
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
import numpy as np

def preprocess_text(text):
    # 1. Remove content inside square brackets, parentheses, and curly braces
    text = re.sub(r'\([^)]*\)|\[[^]]*\]|\{[^}]*\}', '', text)
    
    # --- 💡 조건부 영어 제거 로직 시작 ---
    
    # 2a. 텍스트에 한글이 포함되어 있는지 확인
    # 're.search'는 패턴이 문자열 내에 있는지 확인합니다.
    has_korean = re.search(r'[가-힣]', text)
    
    if has_korean:
        # 2b-1. 한글이 있으면: 한글과 공백만 남기고 (영어도 제거)
        # 한글([^가-힣\s])을 제외한 모든 문자(영어, 숫자, 특수문자) 제거
        text = re.sub(r'[^가-힣\s]', '', text)
    else:
        # 2b-2. 한글이 없으면: 영어, 숫자, 한글을 제외한 특수 문자만 제거 (영어 보존)
        # 특수 문자/숫자([^가-힣a-zA-Z\s])만 제거
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # 참고: 현재 1단계에서 한글은 이미 제거되었으므로, [^a-zA-Z\s]만으로도 충분합니다.
        
    # --- 조건부 영어 제거 로직 끝 ---

    # 3. Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

def create_map_from_csv(file_path):
    result_map = {}
    
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        
        # Get the first row as keys
        try:
            keys = next(reader)
        except StopIteration:
            print("Error: File is empty")
            return {}
            
        # Initialize lists for each key
        for key in keys:
            result_map[key] = []
            
        # Iterate through the rest of the rows
        for row in reader:
            for i, value in enumerate(row):
                if i < len(keys):
                    # Store RAW value here to preserve alignment and codes
                    # Preprocessing will be done during clustering
                    result_map[keys[i]].append(value)
                    
    return result_map

from sklearn.metrics.pairwise import cosine_similarity

def classify_data_by_targets(codes, names, targets):
    # Preprocess names for classification
    cleaned_names = [preprocess_text(name) for name in names]
    
    # Combine targets and cleaned names for vectorization to ensure same feature space
    all_texts = targets + cleaned_names
    
    # Vectorize
    # Use char n-grams to capture partial matches (e.g. "에이드" inside "레몬에이드")
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 3))
    try:
        tfidf_matrix = vectorizer.fit_transform(all_texts)
    except ValueError:
        print("Error: No valid data to classify.")
        return {}, {}, None

    # Split matrix back into targets and items
    target_vectors = tfidf_matrix[:len(targets)]
    item_vectors = tfidf_matrix[len(targets):]
    
    # Calculate similarity between items and targets
    # Shape: (n_items, n_targets)
    similarity_matrix = cosine_similarity(item_vectors, target_vectors)
    
    # Classify
    clusters = {}
    cluster_indices = {}
    
    # Initialize clusters for all targets
    for target in targets:
        clusters[target] = []
        cluster_indices[target] = []
    clusters['Unclassified'] = []
    cluster_indices['Unclassified'] = []
    
    threshold = 0.1 # Minimum similarity to be classified
    
    for idx, (code, name, cleaned_name) in enumerate(zip(codes, names, cleaned_names)):
        if not cleaned_name.strip():
            continue
            
        # Find best matching target
        similarities = similarity_matrix[idx]
        best_target_idx = np.argmax(similarities)
        best_score = similarities[best_target_idx]
        
        if best_score >= threshold:
            target = targets[best_target_idx]
            clusters[target].append((code, name, cleaned_name, best_score))
            cluster_indices[target].append(idx)
        else:
            clusters['Unclassified'].append((code, name, cleaned_name, 0.0))
            cluster_indices['Unclassified'].append(idx)
            
    return clusters, cluster_indices

if __name__ == "__main__":
    file_path = 'waldpos_public_base_goods.csv'
    data_map = create_map_from_csv(file_path)
    
    keys = list(data_map.keys())
    print(f"Keys found: {keys}")
    
    if len(keys) >= 2:
        # Assume Col 0 is Code, Col 1 is Name
        col_code = keys[0]
        col_name = keys[1]
        
        codes = data_map[col_code]
        names = data_map[col_name]
        
        # Verify alignment
        if len(codes) != len(names):
            print(f"Warning: Column lengths mismatch! Codes: {len(codes)}, Names: {len(names)}")
            min_len = min(len(codes), len(names))
            codes = codes[:min_len]
            names = names[:min_len]
        
        # Define Targets
        targets = [
            "아메리카노", "라떼", "프라페", "스무디", "에이드", "티", "차", 
            "쥬스", "요거트", "버블티", "디저트", "케이크", "빵", "베이글", 
            "핫도그", "쿠키", "마카롱", "세트"
        ]
        
        print(f"Classifying {len(codes)} items into {len(targets)} targets...")
        clusters, cluster_indices = classify_data_by_targets(codes, names, targets)
        
        if clusters:
            output_file = 'clustering_results.txt'
            print(f"Writing results to {output_file}...")
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("=== Target-based Classification Results ===\n")
                
                # Sort targets by count
                sorted_targets = sorted(clusters.keys(), key=lambda k: len(clusters[k]), reverse=True)
                
                for target in sorted_targets:
                    items = clusters[target]
                    
                    f.write(f"\n[Target: {target}] (Count: {len(items)})\n")
                    
                    # Print items in cluster
                    # Sort by score desc
                    items.sort(key=lambda x: x[3], reverse=True)
                    
                    for code, name, cleaned, score in items:
                        f.write(f"  - [{code}] {name} -> (Cleaned: {cleaned}) [Sim: {score:.4f}]\n")
            
            print("Done.")
    else:
        print("Error: Need at least 2 columns (Code and Name) to cluster.")


