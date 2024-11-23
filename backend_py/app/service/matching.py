from typing import List, Dict, Optional
from app.model import db_handler
import numpy as np
from tqdm import tqdm  

def cos_sim(A, B):
    """두 벡터 간 코사인 유사도를 계산합니다."""
    A = np.array(A, dtype=float)  
    B = np.array(B, dtype=float)  
    similarity = np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))
    print(f"Cosine similarity 계산 완료: {similarity}")
    return similarity

def matching(user_vector) -> Optional[List[Dict]]:
    """
    유저의 user_id를 기반으로 가장 유사한 변호사 5명의 전체 정보를 반환합니다.
    
    Args:
        user_id (str): 유저의 user_id.
        
    Returns:
        Optional[List[Dict]]: 추천 변호사 목록 (변호사 정보 전체 포함).
    """
    print("유저 벡터 준비 완료.")
    user_vector = user_vector[0]

    print("변호사 벡터를 DB에서 가져오는 중...")
    lawyer_vectors = db_handler.get_all_vectors()
    if lawyer_vectors is None:
        print("변호사 벡터를 찾을 수 없습니다.")
        return None
    print(f"{len(lawyer_vectors)}명의 변호사 벡터를 성공적으로 로드했습니다.")

    print("유사도 계산 및 정렬 진행 중...")
    sorted_lawyers = sorted(
        tqdm(lawyer_vectors, desc="코사인 유사도 계산", unit="vector"),
        key=lambda x: cos_sim(x[1], user_vector.tolist()),
        reverse=True
    )

    print("가장 유사한 상위 5명의 변호사 정보를 가져오는 중...")
    recommend_lawyers = []
    for i in tqdm(range(5), desc="추천 변호사 로드", unit="lawyer"):
        lawyer_id = sorted_lawyers[i][0]
        lawyer_info = db_handler.get_user_by_id(lawyer_id)
        if lawyer_info is None:
            print(f"유저를 찾을 수 없습니다: user_id={lawyer_id}")
            continue

        lawyer_dict = {
            "id": lawyer_info[0],
            "email": lawyer_info[1],
            "username": lawyer_info[2],
            "description": lawyer_info[3]
        }
        recommend_lawyers.append(lawyer_dict)
        print(f"추천 변호사 추가: {lawyer_dict['username']}")

    print("추천 변호사 리스트 작성 완료.")
    return recommend_lawyers