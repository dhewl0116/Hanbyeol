import { useLocation, useNavigate, useParams } from "react-router";
import nestClient from "../../lib/api/nestClient";
import toast from "react-hot-toast";
import HumanPhoto from "../../assets/HumanPhoto.png";
import { useEffect, useState } from "react";
import { LawyerType } from "../../types";

function LawyerDetailPage() {
  const { id } = useParams();
  const location = useLocation();
  const isAccepted: boolean | null = location.state?.isAccepted;
  const [lawyer, setLawyer] = useState<LawyerType | null>(null);
  const navigate = useNavigate();
  useEffect(() => {
    const fetchData = async () => {
      const res = await nestClient.get(`/user/${id}`);
      setLawyer(res.data);
    };
    fetchData();
  }, [id]);

  const HandleRequest = async () => {
    const res = await nestClient.post("/match/send_request", {
      lawyerId: id,
    });
    if (res.data) {
      navigate("../success");
      console.log(res.data);
      return res.data;
    }

    toast.error("Something Went Wrong...");
  };
  return (
    <div className="grid w-full h-screen bg-primary place-content-center">
      <div className="flex justify-center">
        <div className="flex flex-col w-[900px] gap-5 p-10 bg-white rounded-2xl">
          <div className="flex items-center justify-between mb-5">
            <div className="flex flex-col gap-2">
              <h1 className="text-2xl font-semibold">{lawyer?.username}</h1>
              <p className="font-bold">{lawyer?.email}</p>
            </div>
            <img
              src={HumanPhoto}
              alt="profile"
              className="w-32 h-auto rounded-full"
            />
          </div>

          <div className="p-6 mt-4 bg-gray-200 border-2 border-gray-300 rounded-lg">
            <p>{lawyer?.description}</p>
          </div>

          <div className="flex justify-between gap-5 mt-5">
            {isAccepted === null || isAccepted === undefined ? (
              <>
                <button
                  onClick={() => navigate(-1)}
                  className="w-full py-3 text-lg text-white transition-all transform rounded-full shadow-lg bg-primary hover:bg-blue-500 hover:scale-105 active:scale-95"
                >
                  뒤로가기
                </button>
                <button
                  onClick={HandleRequest}
                  className="w-full py-3 text-lg text-white transition-all transform rounded-full shadow-lg bg-primary hover:bg-blue-500 hover:scale-105 active:scale-95"
                >
                  신청하기
                </button>
              </>
            ) : isAccepted === false ? (
              <button
                onClick={() => navigate("/")}
                className="w-full py-3 text-lg text-white transition-all transform rounded-full shadow-lg bg-primary hover:bg-blue-500 hover:scale-105 active:scale-95"
              >
                변호사의 승인을 대기 중입니다.
              </button>
            ) : (
              <>
                <button
                  onClick={() => navigate("./chat")}
                  className="w-full py-3 text-lg text-white transition-all transform rounded-full shadow-lg bg-primary hover:bg-blue-500 hover:scale-105 active:scale-95"
                >
                  채팅하기
                </button>
                <button
                  onClick={() => navigate("/pay")}
                  className="w-full py-3 text-lg text-white transition-all transform rounded-full shadow-lg bg-primary hover:bg-blue-500 hover:scale-105 active:scale-95"
                >
                  BNPL
                </button>
                <button
                  onClick={() =>
                    navigate("./recoverystatus", {
                      state: {
                        isLawyer: false,
                      },
                    })
                  }
                  className="w-full py-3 text-lg text-white transition-all transform rounded-full shadow-lg bg-primary hover:bg-blue-500 hover:scale-105 active:scale-95"
                >
                  진척도 확인하기
                </button>
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default LawyerDetailPage;
