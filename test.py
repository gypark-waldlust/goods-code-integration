import csv
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
import numpy as np

def preprocess_text(text):
    # 1. Remove content inside square brackets
    text = re.sub(r'\[.*?\]', '', text)
    
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

def cluster_data(codes, names):
    # Preprocess names for clustering using the user's function
    cleaned_names = [preprocess_text(name) for name in names]
    
    # Filter out empty names to avoid noise, but we need to keep indices aligned with codes
    # So we'll just cluster everything, and empty strings will likely form their own cluster or be noise
    
    # Vectorize
    vectorizer = TfidfVectorizer()
    try:
        tfidf_matrix = vectorizer.fit_transform(cleaned_names)
    except ValueError:
        print("Error: No valid data to cluster.")
        return {}

    # Cluster using DBSCAN
    # eps: maximum distance between two samples for one to be considered as in the neighborhood of the other
    # min_samples: The number of samples (or total weight) in a neighborhood for a point to be considered as a core point
    # Lower eps = stricter similarity (smaller distance required)
    dbscan = DBSCAN(eps=0.3, min_samples=2, metric='cosine')
    labels = dbscan.fit_predict(tfidf_matrix)
    
    # Group by cluster
    clusters = {}
    for code, name, cleaned_name, label in zip(codes, names, cleaned_names, labels):
        # Optional: Skip items that became empty after preprocessing if desired
        if not cleaned_name.strip():
            continue
            
        if label not in clusters:
            clusters[label] = []
        clusters[label].append((code, name, cleaned_name))
        
    return clusters

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
            # Truncate to shorter length
            min_len = min(len(codes), len(names))
            codes = codes[:min_len]
            names = names[:min_len]
        
        print(f"Clustering {len(codes)} items...")
        clusters = cluster_data(codes, names)
        
        if clusters:
            # Sort clusters by size
            sorted_labels = sorted(clusters.keys(), key=lambda k: len(clusters[k]), reverse=True)
            
            print("\n=== Clustering Results ===")
            for label in sorted_labels:
                items = clusters[label]
                if label == -1:
                    print(f"\n[Noise] (Count: {len(items)})")
                else:
                    print(f"\n[Cluster {label}] (Count: {len(items)})")
                
                # Print first 5 items in cluster
                for code, name, cleaned in items[:5]:
                    print(f"  - [{code}] {name} -> (Cleaned: {cleaned})")
                if len(items) > 5:
                    print(f"  ... and {len(items)-5} more")
    else:
        print("Error: Need at least 2 columns (Code and Name) to cluster.")
