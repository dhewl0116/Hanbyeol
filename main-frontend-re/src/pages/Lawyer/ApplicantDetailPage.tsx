import { useEffect, useState } from "react";
import { useLocation, useNavigate, useParams } from "react-router";
import nestClient from "../../lib/api/nestClient";
import HumanPhoto from "../../assets/HumanPhoto.png";
import toast from "react-hot-toast";

interface JsonResponse {
  id: string;
  username: string;
  role: string;
  description: string;
  created_at: string;
}

function ApplicantDetailPage() {
  const { id } = useParams();
  const navigate = useNavigate();
  const location = useLocation();
  const { accepted } = location.state;
  const [applicant, setApplicant] = useState<JsonResponse | null>(null);

  const HandleAccept = async () => {
    const res = await nestClient.patch(`match/edit_progress/${id}`, {
      status: "document submission",
    });
    toast.success("성공적으로 수락했습니다!");
    navigate(-1);
    console.log(res.data);
  };

  const HandleReject = async () => {
    const res = await nestClient.delete(`match/reject_request/${id}`);
    toast.success("성공적으로 거절했습니다!");
    navigate(-1);
    console.log(res.data);
  };

  useEffect(() => {
    const fetchData = async () => {
      const res = await nestClient.get(`/user/${id}`);
      setApplicant(res.data);
    };
    fetchData();
  }, [id]);

  return (
    <div className="grid w-full h-screen bg-primary place-content-center">
      <div className="flex flex-col w-[900px] gap-5 p-10 bg-white rounded-2xl">
        <div className="flex items-center justify-between mb-5">
          <div className="flex flex-col gap-2">
            <h1 className="text-4xl font-semibold">{applicant?.username}</h1>
          </div>
          <img
            src={HumanPhoto}
            alt="profile"
            className="w-32 h-auto rounded-full"
          />
        </div>

        <div className="p-6 mt-4 bg-gray-200 border-2 border-gray-300 rounded-lg">
          <p>{applicant?.description}</p>
        </div>

        {accepted ? (
          <div className="flex justify-between gap-5 mt-5">
            <button
              onClick={() => navigate("./chat")}
              className="w-full py-3 text-lg text-white transition-all transform rounded-full shadow-lg bg-primary hover:bg-blue-500 hover:scale-105 active:scale-95"
            >
              채팅하기
            </button>
            <button
              onClick={() =>
                navigate("./recoverystatus", {
                  state: {
                    isLawyer: true,
                  },
                })
              }
              className="w-full py-3 text-lg text-white transition-all transform rounded-full shadow-lg bg-primary hover:bg-blue-500 hover:scale-105 active:scale-95"
            >
              진척도 체크하기
            </button>
          </div>
        ) : (
          <div className="flex justify-between gap-5 mt-5">
            <button
              onClick={HandleAccept}
              className="w-full py-3 text-lg text-white transition-all transform rounded-full shadow-lg bg-primary hover:bg-blue-500 hover:scale-105 active:scale-95"
            >
              수락하기
            </button>
            <button
              onClick={HandleReject}
              className="w-full py-3 text-lg text-white transition-all transform rounded-full shadow-lg bg-primary hover:bg-blue-500 hover:scale-105 active:scale-95"
            >
              거절하기
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

export default ApplicantDetailPage;
