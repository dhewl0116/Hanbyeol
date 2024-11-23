import { useState } from "react";
import PossibilityButton from "../../components/Button/PossibilityButton";
import Button from "../../components/Button/Button";
import { useNavigate } from "react-router";
import usePossibilityStore from "../../lib/store/usePossibilityStore";
import toast from "react-hot-toast";

const PossibilitySpouse = () => {
  const navigate = useNavigate();
  const [spouse, setSpouse] = useState<boolean | undefined>(undefined); // 배우자 여부
  const [spouseState, setSpouseState] = useState<string | undefined>(undefined); // 결혼 상태
  const { setMaritalstatus } = usePossibilityStore();
  const HandleClick = () => {
    if (spouse === undefined || (spouse && spouseState === undefined)) {
      toast.error("답변 형식이 올바르지 않습니다.");
      return;
    } else if (spouse === false) {
      setMaritalstatus("미혼");
    } else if (spouse && spouseState) {
      setMaritalstatus(spouseState);
    }
    navigate("../monthlyincome");
  };
  return (
    <>
      <div className="flex flex-col items-center w-full h-[75vh] bg-primary relative">
        <div className="flex flex-col px-20 w-[586px] h-[741px] bg-white rounded-3xl shadow-[0px_4px_30px_rgba(0,0,0,0.25)] relative top-1/4">
          <h1 className="mt-16 text-4xl font-bold">
            자녀를 같이 양육했던 <br />
            배우자가 계신가요?
          </h1>
          <p className="mt-2 font-bold text-fontlight">
            이혼하셨더라도 살아계신다면, "있어요" 라고 대답해주세요.
          </p>
          <div className="flex mt-4 gap-x-4">
            <PossibilityButton
              onClick={() => setSpouse(true)}
              activated={spouse === true}
            >
              있어요
            </PossibilityButton>
            <PossibilityButton
              onClick={() => setSpouse(false)}
              activated={
                spouse ===
                false /* spouse가 undefined 일때 활성화를 방지하기 위함 */
              }
            >
              없어요
            </PossibilityButton>
          </div>

          {spouse && (
            <>
              <h2 className="mt-12 text-2xl font-bold">
                현재 그 분과 결혼 중인 상태이신가요?
              </h2>
              <div className="flex mt-4 gap-x-4">
                <PossibilityButton
                  onClick={() => setSpouseState("기혼")}
                  activated={spouseState === "기혼"}
                >
                  결혼
                </PossibilityButton>
                <PossibilityButton
                  onClick={() => setSpouseState("이혼")}
                  activated={spouseState === "이혼"}
                >
                  이혼
                </PossibilityButton>
                <PossibilityButton
                  onClick={() => setSpouseState("미혼")}
                  activated={spouseState === "미혼"}
                >
                  미혼
                </PossibilityButton>
              </div>
            </>
          )}

          <div className="flex-1"></div>

          <Button
            onClick={HandleClick}
            className="w-[404px] text-sm self-center rounded-xl mb-5"
            secondary
          >
            다음으로
          </Button>
          <p
            onClick={() => navigate(-1)}
            className="self-center mb-10 text-sm font-bold cursor-pointer"
          >
            뒤로가기
          </p>
        </div>
      </div>

      <div className="w-full h-[25vh] bg-secondary"></div>
    </>
  );
};

export default PossibilitySpouse;
