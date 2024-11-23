import { useNavigate } from "react-router";
import Button from "../../components/Button/Button";
import PossibilityButton from "../../components/Button/PossibilityButton";
import { ChangeEvent, useState } from "react";
import Input from "../../components/Input/Input";
import toast from "react-hot-toast";
import usePossibilityStore from "../../lib/store/usePossibilityStore";

const PossibilityNumberOfDependents = () => {
  const navigate = useNavigate();
  const [answer, setAnswer] = useState<boolean | undefined>(undefined); // 미성년자 자녀 여부 (답변이 하나일 경우, answer 로 설정.)
  const [value, setValue] = useState<string>("");
  const { setNumberOfDependents } = usePossibilityStore();

  const HandleChange = (e: ChangeEvent<HTMLInputElement>) => {
    const { value } = e.target;
    if (/^\d*$/.test(value)) {
      setValue(value);
    }
  };

  const HandleClick = () => {
    if (answer === undefined || (answer && value === "")) {
      toast.error("답변 형식이 올바르지 않습니다.");
      return;
    } else if (answer === false) {
      setNumberOfDependents(0);
    } else {
      setNumberOfDependents(Number(value));
    }
    navigate("../spouse");
  };

  return (
    <>
      <div className="flex flex-col items-center w-full h-[75vh] bg-primary relative">
        <div className="flex flex-col px-20 w-[586px] h-[741px] bg-white rounded-3xl shadow-[0px_4px_30px_rgba(0,0,0,0.25)] relative top-1/4">
          <h1 className="mt-16 text-4xl font-bold">
            현재 가족을 부양하고 계신가요?
          </h1>
          <div className="flex mt-8 gap-x-4">
            <PossibilityButton
              onClick={() => setAnswer(true)}
              activated={answer === true}
            >
              네
            </PossibilityButton>
            <PossibilityButton
              onClick={() => setAnswer(false)}
              activated={
                answer ===
                false /* answer 가 undefined 일때 활성화를 방지하기 위함 */
              }
            >
              아니요
            </PossibilityButton>
          </div>

          {answer && (
            <>
              <h2 className="mt-16 text-2xl font-bold">
                몇 명을 부양하고 계신가요?
              </h2>
              <Input
                type="text"
                placeholder="자녀 수를 입력해주세요."
                className="mt-4 w-[404px] self-center"
                onChange={HandleChange}
                value={value}
                required
              />
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

export default PossibilityNumberOfDependents;
