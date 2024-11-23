from fastapi import APIRouter
from app.router.DTO import text_DTO, Calculator_DTO, Macthing_DTO
from app.service import *
from psycopg2 import pool, DatabaseError
import uuid


AI_router = APIRouter(prefix= '/ai')

connection_string = ""
try:
    connection_pool = pool.SimpleConnectionPool(1, 10, connection_string)
    if connection_pool:
        print("연결 풀이 성공적으로 생성되었습니다.")
except DatabaseError as e:
    print("연결 풀을 생성하는 데 실패했습니다:", e)

@AI_router.post("/vec")
async def classify_word(param:text_DTO): 
    """
    Request: sentence, user_id

    Role: sentence를 분류벡터로 예측하여 변경, 분류벡터와 sentence를 각각 Vector table과 User table에 저장 및 반환
    """

    classifier = Maching_Classifier(model_path= "/Users/dhewl/Desktop/hanbyeol/Source/best_Koelectra_8.pt" )
    sentence = param.sentence
    user_id = param.user_id
    vector = classifier.predict(predict_string=sentence).tolist()

    try:
        conn = connection_pool.getconn()
        with conn.cursor() as cur:
            vector_id = str(uuid.uuid4())
            cur.execute("""
                INSERT INTO "Vector" (id, user_id, vector)
                VALUES (%s, %s, %s::float8[])
                ON CONFLICT (user_id) DO UPDATE
                SET vector = EXCLUDED.vector;
            """, (vector_id, user_id, vector))

            cur.execute("""
                UPDATE "User"
                SET description = %s
                WHERE id = %s;
            """, (sentence, user_id))
            
            conn.commit()
    except Exception as e:
        return {"status": "error", "message": str(e)}
    finally:
        if 'conn' in locals():
            connection_pool.putconn(conn)

    return {
        "status": "success",
        "message": vector
    }


@AI_router.post("/matching")
async def matching_lawyer(param: Macthing_DTO):
    """
    유저의 user_id를 기반으로 가장 유사한 변호사 5명의 전체 정보를 반환
    """
    print("매칭 프로세스 시작...")
    sentence = param.sentence
    print("입력 문장:", sentence)
    
    print("분류기 로드 중...")
    classifier = Maching_Classifier(model_path="/Users/dhewl/Desktop/hanbyeol/Source/best_Koelectra_8.pt")
    print("분류기 로드 완료. 추론 시작...")
    user_vector = classifier.predict(sentence)
    print("유저 벡터 추론 완료.")

    recommended_lawyers = matching(user_vector)
    
    if recommended_lawyers is None:
        print("추천 변호사를 찾을 수 없습니다.")
        return {"status": "error", "message": "추천 변호사를 찾을 수 없습니다."}
    
    print("추천 결과:", recommended_lawyers)
    
    return {
        "status": "success",
        "message": recommended_lawyers
    }


@AI_router.post("/revival")
async def calculate_revival(param:Calculator_DTO):

    calculater =  Revival_Calculator(model_path='/Users/dhewl/Desktop/hanbyeol/Source/Revival Cal acc_99.74 f1_99.80.pt', data_path="/Users/dhewl/Desktop/hanbyeol/Source/Train/data/revival_data.csv")
    result, explanation = calculater.predict(**param.model_dump())
    print(f"회생가능성 : {result}%")
    print(f"설명: {explanation}")
    return {
        "status":"success",
        "message":{
            "score": result,
            "explanation": explanation
        }
    }

    


