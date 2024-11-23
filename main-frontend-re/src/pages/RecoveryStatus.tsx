import { FC, useEffect, useState } from "react";
import Progress1 from "../assets/progress1.png";
import Progress2 from "../assets/progress2.png";
import Progress3 from "../assets/progress3.png";
import Progress4 from "../assets/progress4.png";
import { useLocation, useNavigate, useParams } from "react-router";
import nestClient from "../lib/api/nestClient";
import toast from "react-hot-toast";

type Status =
  | ""
  | "not accepted now"
  | "document submission"
  | "correction response"
  | "creditor meeting"
  | "repayment plan approval";

const StatusToKorean: FC<{ status: string }> = ({ status }) => {
  switch (status) {
    case "document submission":
      return <h1 className="mt-4 text-4xl font-bold">필요서류 제출</h1>;
    case "correction response":
      return <h1 className="mt-4 text-4xl font-bold">보정권고 대응</h1>;
    case "creditor meeting":
      return <h1 className="mt-4 text-4xl font-bold">채권자 집회</h1>;
    case "repayment plan approval":
      return <h1 className="mt-4 text-4xl font-bold">변제계획 인가</h1>;
  }
};

function RecoveryStatus() {
  const { id } = useParams();
  const location = useLocation();
  const { isLawyer } = location.state;
  const navigate = useNavigate();
  const [status, setStatus] = useState<Status>("");

  const HandleChange = (value: Status) => {
    setStatus(value);
  };

  const HandleSave = async () => {
    const res = await nestClient.patch(`/match/edit_progress/${id}`, {
      status,
    });
    if (res.data) {
      toast.success("변경사항을 저장했습니다!");
      navigate(-1);
    }
  };

  useEffect(() => {
    const fetchData = async () => {
      const res = await nestClient.get(`match/${id}`);
      if (res.status >= 200 && res.status < 300) {
        setStatus(res.data.status);
      }
    };
    fetchData();
  }, [id]);

  return (
    <div className="grid w-full h-screen bg-primary place-content-center">
      <div className="w-[1200px] p-10 text-center bg-white  rounded-2xl">
        <h2 className="text-2xl font-bold">나의 회생 상황</h2>
        <StatusToKorean status={status} />

        <div className="flex flex-col mt-6">
          <ul className="flex justify-around mt-4 text-lg font-bold">
            <li>필요 서류 제출</li>
            <li>보정권고 대응</li>
            <li>채권자 집회</li>
            <li>변제계획 인가</li>
          </ul>

          <div className="mt-6">
            <div className="w-full h-1 bg-blue-600"></div>
            <ul className="flex justify-around mt-2">
              <li className="w-3 h-3 rounded-full bg-blue-600 -mt-1.5"></li>
              <li className="w-3 h-3 rounded-full bg-blue-600 -mt-1.5"></li>
              <li className="w-3 h-3 rounded-full bg-blue-600 -mt-1.5"></li>
              <li className="w-3 h-3 rounded-full bg-blue-600 -mt-1.5"></li>
            </ul>
          </div>

          <div className="flex justify-around mt-6">
            <div className="text-lg font-bold text-center">
              <img
                src={Progress1}
                alt=""
                className="mt-4 w-36 h-36"
                onClick={() => HandleChange("document submission")}
              />
            </div>
            <div className="text-lg font-bold text-center">
              <img
                src={Progress2}
                alt=""
                className="mt-4 w-36 h-36"
                onClick={() => HandleChange("correction response")}
              />
            </div>
            <div className="text-lg font-bold text-center">
              <img
                src={Progress3}
                alt=""
                className="mt-4 w-36 h-36"
                onClick={() => HandleChange("creditor meeting")}
              />
            </div>
            <div className="text-lg font-bold text-center">
              <img
                src={Progress4}
                alt=""
                className="mt-4 w-36 h-36"
                onClick={() => HandleChange("repayment plan approval")}
              />
            </div>
          </div>
        </div>

        <div className="flex justify-center mt-8 gap-x-12">
          {isLawyer && (
            <button
              onClick={HandleSave}
              className="px-8 py-3 mt-6 text-lg text-white transition-all transform rounded-full shadow-lg bg-primary hover:bg-blue-500 hover:scale-105 active:scale-95"
            >
              변경사항 저장
            </button>
          )}
          <button
            onClick={() => navigate(-1)}
            className="px-8 py-3 mt-6 text-lg text-white transition-all transform rounded-full shadow-lg bg-primary hover:bg-blue-500 hover:scale-105 active:scale-95"
          >
            뒤로가기
          </button>
        </div>
      </div>
    </div>
  );
}

export default RecoveryStatus;
