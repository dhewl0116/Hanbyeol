import { FaSearch } from "react-icons/fa";
import Lawyer from "../../components/Lawyer";

import { useLocation } from "react-router";

interface JsonResponse {
  id: string;
  email: string;
  username: string;
  description: string;
}

const LawyerPage = () => {
  const location = useLocation();
  const state: JsonResponse[] = location.state.data.message;

  console.log(state);

  /*
  json 응답 방식:
   {
        "lawyer_info": {
            "id": "uuid",
            "email": "lawyer2@example.com",
            "username": "Lawyer 2",
            "role": "lawyer",
            "description": "성명",
            "created_at": "2024-01-01T00:00:00Z",
            // 기타 변호사 정보
        },
        "similarity": 0.87
    },
  */

  return (
    <>
      {state && (
        <>
          {" "}
          <div className="w-full h-[511px] bg-primary shadow-[0_4px_30px_0_rgba(0,0,0,0.2)] flex justify-center gap-x-16 items-center pt-24">
            <h1 className="font-bold text-white text-[64px]">
              당신에게 맞는
              <br />
              변호사를 찾으세요
            </h1>
            <div className="w-[451px] flex items-center bg-transparent border-2 border-white rounded-3xl relative top-10">
              <input
                placeholder="검색하기"
                className="h-16 pr-4 ml-4 text-2xl font-bold text-white bg-transparent w-96 placeholder:text-white focus:outline-none"
              />
              <FaSearch size={25} color="white" />
            </div>
          </div>
          <div className="flex flex-col items-center w-full py-24 gap-y-16 bg-secondary">
            <h1 className="mb-12 text-6xl font-bold">가장 추천하는 변호사</h1>
            <Lawyer
              username={state[0].username}
              bio={state[0].email}
              id={state[0].id}
              className="mb-20 scale-125"
            />
            {state.slice(1).map((v, i) => {
              const { id, username, email } = v;
              return (
                <div key={i}>
                  <Lawyer username={username} bio={email} id={id} />
                </div>
              );
            })}
          </div>
        </>
      )}
    </>
  );
};

export default LawyerPage;
