from psycopg2 import pool, DatabaseError
import uuid
from typing import List, Optional, Tuple

class DatabaseHandler:
    def __init__(self, connection_string: str):
        try:
            self.connection_pool = pool.SimpleConnectionPool(1, 10, connection_string)
            if self.connection_pool:
                print("데이터베이스 연결 풀이 성공적으로 생성되었습니다.")
        except DatabaseError as e:
            print("데이터베이스 연결 풀을 생성하는 데 실패했습니다:", e)
            self.connection_pool = None

    def _get_connection(self):
        return self.connection_pool.getconn()

    def _return_connection(self, conn):
        if conn:
            self.connection_pool.putconn(conn)

    def get_all_vectors(self) -> Optional[List[Tuple[str, List[float]]]]:

        """Vector 테이블에서 모든 벡터와 해당 user_id를 반환합니다."""
        
        conn = None
        try:
            conn = self._get_connection()
            with conn.cursor() as cur:
                cur.execute("SELECT user_id, vector FROM \"Vector\";")
                vectors = cur.fetchall()
            return vectors
        except Exception as e:
            print("Vector 테이블에서 모든 벡터를 가져오는 데 실패했습니다:", e)
            return None
        finally:
            self._return_connection(conn)

    def add_or_update_vector(self, user_id: str, vector: List[float]) -> None:
        """user_id와 vector를 받아 Vector 테이블에 저장 또는 업데이트합니다."""
        conn = None
        try:
            conn = self._get_connection()
            with conn.cursor() as cur:
                vector_id = str(uuid.uuid4()) 
                cur.execute("""
                    INSERT INTO "Vector" (id, user_id, vector)
                    VALUES (%s, %s, %s::float8[])
                    ON CONFLICT (user_id) DO UPDATE
                    SET vector = EXCLUDED.vector;
                """, (vector_id, user_id, vector))
                conn.commit()
        except Exception as e:
            print("Vector 테이블에 데이터를 추가/업데이트하는 데 실패했습니다:", e)
        finally:
            self._return_connection(conn)

    def get_lawyers(self) -> Optional[List[tuple]]:
        """User 테이블에서 Role이 'lawyer'인 모든 데이터를 반환합니다."""
        conn = None
        try:
            conn = self._get_connection()
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM \"User\" WHERE role = 'lawyer';")
                lawyers = cur.fetchall()
            return lawyers
        except Exception as e:
            print("User 테이블에서 'lawyer' 데이터를 가져오는 데 실패했습니다:", e)
            return None
        finally:
            self._return_connection(conn)

    def get_vector_by_user_id(self, user_id: str) -> Optional[List[float]]:
        """user_id를 이용해 Vector 테이블의 벡터를 반환합니다."""
        conn = None
        try:
            conn = self._get_connection()
            with conn.cursor() as cur:
                cur.execute("SELECT vector FROM \"Vector\" WHERE user_id = %s;", (user_id,))
                result = cur.fetchone()
                if result:
                    return result[0]
                else:
                    return None
        except Exception as e:
            print("[ERROR] Vector 테이블에서 벡터를 가져오는 데 실패했습니다:", e)
            return None
        finally:
            if conn:
                self._return_connection(conn)

        
    def get_user_by_id(self, user_id:str) -> Optional[tuple]:
        """user_id를 이용해 User 테이블의 유저데이터를 반환합니다."""
        conn = None
        try:
            conn = self._get_connection()
            with conn.cursor() as cur:
                cur.execute("SELECT id, email, username, description FROM \"User\" WHERE id = %s;", (user_id,))
                result = cur.fetchone()
                if result:
                    return result
                else:
                    print("해당 user_id로 유저를 찾을 수 없습니다.")
        except Exception as e:
            print("User 테이블에서 유저를 가져오는데 실패하였습니다.", e)
            return None
        finally:
            self._return_connection(conn)
    def get_bnpl_amount(self, user_id: str) -> Optional[int]:
        """
        Bnpl 테이블에서 해당 user_id의 현재 amount를 조회합니다.
        """
        conn = None
        try:
            conn = self._get_connection()
            with conn.cursor() as cur:
                cur.execute("SELECT amount FROM \"Bnpl\" WHERE user_id = %s;", (user_id,))
                result = cur.fetchone()
                return result[0] if result else None
        except Exception as e:
            print("Bnpl 테이블에서 amount를 가져오는 데 실패했습니다:", e)
            return None
        finally:
            self._return_connection(conn)

    def update_bnpl_amount(self ,user_id: str, new_amount: int) -> bool:
        """
        Bnpl 테이블에서 해당 user_id의 amount를 업데이트합니다.
        """
        conn = None
        try:
            conn = self._get_connection()
            with conn.cursor() as cur:
                cur.execute("UPDATE \"Bnpl\" SET amount = %s WHERE user_id = %s;", (new_amount, user_id))
                conn.commit()
                return True
        except Exception as e:
            print("Bnpl 테이블에서 amount를 업데이트하는 데 실패했습니다:", e)
            return False
        finally:
            self._return_connection(conn)
    



if __name__ == "__main__":
    db_handler = DatabaseHandler(connection_string="")
    print(db_handler.get_all_vectors())