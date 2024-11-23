import { useEffect, useState } from "react";
import usePossibilityStore from "../../lib/store/usePossibilityStore";
import aiClient from "../../lib/api/aiClient";
import toast from "react-hot-toast";
import ProbabilityRate from "../../components/ProbabilityRate";

interface JsonType {
  status: string;
  message: {
    score: number;
    explanation: string;
  };
}

const PossibilityResult = () => {
  const {
    AssetValue,
    MaritalStatus,
    MonthlyIncome,
    NumberOfDependents,
    TotalDebt,
  } = usePossibilityStore();
  const [result, setResult] = useState<JsonType | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  useEffect(() => {
    console.log({
      NumberOfDependents,
      MaritalStatus,
      MonthlyIncome,
      AssetValue,
      TotalDebt,
    }); // test

    aiClient
      .post("/ai/revival", {
        NumberOfDependents,
        MaritalStatus,
        MonthlyIncome,
        AssetValue,
        TotalDebt,
      })
      .then((r) => {
        console.log(r.data);
        toast.success("결과가 나왔습니다!");
        setResult(r.data);
        setIsLoading(false);
      });
  }, []);
  if (isLoading)
    return (
      <div className="flex flex-col items-center justify-center w-full h-screen bg-primary">
        <div className="absolute -translate-x-1/2 -translate-y-1/2 top-1/2 left-1/2">
          <h1 className="text-6xl font-bold text-white">
            회생 가능성을 계산하고 있습니다
          </h1>
          <h3 className="mt-8 text-4xl font-bold text-white">
            잠시만 기다려 주세요...
          </h3>
        </div>
      </div>
    );
  else {
    return (
      <>
        <div className="flex flex-col items-center justify-center w-full h-screen bg-primary">
          {result && (
            <div className="absolute -translate-x-1/2 -translate-y-1/2 top-1/2 left-1/2">
              <ProbabilityRate
                rate={result?.message.score}
                message={result?.message.explanation}
              />
            </div>
          )}
        </div>
      </>
    );
  }
};

export default PossibilityResult;
