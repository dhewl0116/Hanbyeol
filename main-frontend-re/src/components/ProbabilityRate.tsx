import { FC } from "react";
import { FaRegSmile } from "react-icons/fa";

interface Props {
  rate: number;
  message: string;
}

const ProbabilityRate: FC<Props> = ({ rate, message }) => {
  return (
    <div className="w-[560px] h-[800px] bg-secondary flex flex-col items-center py-4 rounded-xl">
      <div className="bg-white w-[540px] h-96 rounded-2xl flex flex-col justify-center items-center">
        <FaRegSmile size={288} />
        <h3 className="mt-6 text-4xl font-bold">
          {rate >= 75
            ? "회생 가능성이 높아요!"
            : rate >= 40
            ? "어렵지만 가능할 수 있어요"
            : "회생이 불가능할 가능성이 높아요"}
        </h3>
      </div>
      <div className="h-12 bg-white w-[540px] rounded-xl mt-4 flex items-center px-6">
        <p className="text-2xl font-bold">회생 가능성: {rate}%</p>
      </div>
      <div className="bg-white w-[540px] flex-1 rounded-xl mt-4 p-4 text-xl">
        <p className="font-bold">{message}</p>
      </div>
    </div>
  );
};

export default ProbabilityRate;
