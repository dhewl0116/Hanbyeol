import { useNavigate } from "react-router";
import Button from "../../components/Button/Button";
import { ChangeEvent, useState } from "react";
import useAssetValueModalStore from "../../lib/store/useAssetValueModalStore";
import PossibilityInput from "../../components/Input/PossibilityInput";
import usePossibilityStore from "../../lib/store/usePossibilityStore";
import toast from "react-hot-toast";

const PossibilityAssetValues = () => {
  const navigate = useNavigate();
  const { onOpen } = useAssetValueModalStore();
  const { setAssetValue } = usePossibilityStore();
  const [value, setValue] = useState<string>("");

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
    setAssetValue(Number(value));
    navigate("../debtor");
  };
  return (
    <>
      <div className="flex flex-col items-center w-full h-[75vh] bg-primary relative">
        <div className="flex flex-col px-20 w-[586px] h-[741px] bg-white rounded-3xl shadow-[0px_4px_30px_rgba(0,0,0,0.25)] relative top-1/4">
          <h1 className="mt-16 text-[32px] font-bold">
            소유하고 계신 재산의 <br /> 현재 시점 가치를 알려주세요.
          </h1>

          <div className="w-[414px] bg-[#ffc6c6] rounded-2xl flex items-center px-8 py-3 font-bold text-[#d64646] text-wrap mt-3">
            소유하고 계신 부동산에 담보가 설정되어 있거나 보증금처럼 돌려줘야
            하는 돈이 있는 경우 그 금액만큼은 가치에서 빼고 계산해 주세요.
          </div>

          <PossibilityInput
            placeholder="금액을 입력해주세요."
            className="text-2xl border-b border-black w-72 focus:outline-none"
            onChange={HandleChange}
            value={value}
            required
          />

          <button
            onClick={onOpen}
            className="px-8 py-4 mt-6 font-bold rounded-lg w-fit h-fit bg-secondary"
          >
            재산에는 어떤 것들이 들어가나요?
          </button>

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

export default PossibilityAssetValues;
