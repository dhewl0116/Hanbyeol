import { useNavigate } from "react-router";
import Button from "../../components/Button/Button";
import { ChangeEvent, useState } from "react";
import PossibilityInput from "../../components/Input/PossibilityInput";
import usePossibilityStore from "../../lib/store/usePossibilityStore";
import toast from "react-hot-toast";

const PossibilityDebt = () => {
  const navigate = useNavigate();
  const [value, setValue] = useState("");
  const { setTotalDebt } = usePossibilityStore();

  const HandleChange = (e: ChangeEvent<HTMLInputElement>) => {
    const { value } = e.target;
    if (/^\d*$/.test(value)) {
      setValue(value);
    }
  };

  const HandleClick = () => {
    if (value === "") {
      toast.error("답변 형식이 올바르지 않습니다.");
      return;
    }
    setTotalDebt(Number(value));
    navigate("../result");
  };

  return (
    <>
      <div className="flex flex-col items-center w-full h-[75vh] bg-primary relative">
        <div className="flex flex-col px-20 w-[586px] h-[741px] bg-white rounded-3xl shadow-[0px_4px_30px_rgba(0,0,0,0.25)] relative top-1/4">
          <h1 className="my-16 text-[32px] font-bold">
            모든 채무의
            <br />
            원금 합계를 알려주세요.
          </h1>

          <PossibilityInput
            type="text"
            placeholder="금액을 입력해주세요."
            className="text-2xl border-b border-black w-72 focus:outline-none"
            onChange={HandleChange}
            value={value}
            required
          />

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

export default PossibilityDebt;
