import { useNavigate } from "react-router";
import Button from "../components/Button/Button";
import { useUser } from "../hooks/useUser";
import Main1 from "../assets/main1.png";
import Main2 from "../assets/main2.png";
import Main3 from "../assets/main3.png";
import nestClient from "../lib/api/nestClient";
import toast from "react-hot-toast";
import { useEffect } from "react";

const MainPage = () => {
  const navigate = useNavigate();
  const { user, isLoading } = useUser();

  const checkLogIn = () => {
    if (!user && !isLoading) {
      navigate("/login");
      return;
    } else if (user) {
      navigate("/board");
    }
  };

  useEffect(() => {
    if (!isLoading && user && user.role === "applicant") {
      const checkLawyer = async () => {
        const res = await nestClient
          .get("/match/check_matched_lawyer")
          .then((r) => r.data);
        console.log(res);
        if (res) {
          toast("매칭된 변호사가 존재하여 변호사 페이지로 이동합니다.");

          navigate(`/lawyer_result/${res.lawyer.id}`, {
            state: {
              isAccepted: res.status !== "not accepted now" ? true : false,
            },
          });
        }
      };
      checkLawyer();
    }
  }, [isLoading, user]);

  return (
    <div className="bg-secondary">
      <div className="flex h-screen bg-primary justify-center items-center gap-[60px] shadow-[0_4px_30px_0_rgba(0,0,0,0.2)]">
        <div className="flex flex-col gap-[30px]">
          <h1 className="text-[64px] font-bold text-white mt-[50px] leading-tight">
            개인 회생 서비스와 <br />
            변호사 매칭 서비스로 <br />
            <span className="font-normal">한번에</span>
          </h1>
          <p className="text-[20px] text-white">
            BNPL 서비스와 AI 데이터 기반으로 통합 솔루션 제공
          </p>
          <div className="flex gap-[20px]">
            <Button onClick={() => navigate("/lawyer")}>시작하기</Button>
            <Button onClick={checkLogIn} secondary>
              신청하기
            </Button>
          </div>
        </div>

        <img src={Main1} />
      </div>
      <div className="flex flex-col items-center mt-[80px]">
        <h2 className="text-[48px] font-bold text-center">한별을 소개합니다</h2>
        <p className="text-[20px] text-fontlight text-center">
          한별은 AI 변호사 매칭과 BNPL 서비스를 통해 <br />
          청년들의 금융위기 극복을 도와주는 플랫폼입니다.
        </p>
      </div>
      <div className="flex flex-col items-center mt-[80px]">
        <div className="flex gap-[80px]">
          <img src={Main2} />
          <div className="flex flex-col gap-[10px] justify-center">
            <h3 className="text-[48px] font-bold text-right">
              AI 변호사 매칭 서비스
            </h3>
            <p className="text-[20px] text-fontlight text-right">
              AI가 매칭해주는 최적의 변호사를 통해 <br />
              개인 회생/파산을 손쉽게 신청해보세요
            </p>
          </div>
        </div>
        <div className="flex gap-[80px] mt-[50px]">
          <div className="flex flex-col gap-[10px] justify-center">
            <h3 className="text-[48px] font-bold">회생점수계산기</h3>
            <p className="text-[20px] text-fontlight">
              회생/파산이 가능한지, 나랑 비슷한 사람 중 몇 명이 <br />
              회생/파산 신청에 성공했는지를 한눈에 볼 수 있는 기능이에요
            </p>
          </div>
          <img src={Main3} />
        </div>
      </div>
      <div className="h-[500px]" />
    </div>
  );
};

export default MainPage;
