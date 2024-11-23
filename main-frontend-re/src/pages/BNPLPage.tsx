import { ChangeEvent, useState } from "react";
import Button from "../components/Button/Button";
import aiClient from "../lib/api/aiClient";
import { useUser } from "../hooks/useUser";
import toast from "react-hot-toast";
import { useQuery } from "@tanstack/react-query";

const BNPLPage = () => {
  const { user, isLoading } = useUser();
  const [name, setName] = useState("");
  const [account, setAccount] = useState("");
  const [amount, setAmount] = useState("");

  const HandleName = (e: ChangeEvent<HTMLInputElement>) => {
    setName(e.target.value);
  };

  const HandleAccount = (e: ChangeEvent<HTMLInputElement>) => {
    setAccount(e.target.value);
  };

  const HandleAmount = (e: ChangeEvent<HTMLInputElement>) => {
    const { value } = e.target;
    if (/^\d*$/.test(value)) {
      setAmount(value);
    }
  };

  const HandleBNPL = async () => {
    const res = await aiClient.post("/bnpl/call", {
      user_id: user.userId,
      account_num: account,
      money: Number(amount),
    });
    if (res.data.status === "fail") {
      toast.error("30만원 이상 빌릴 수 없습니다.");
      return;
    }
    toast.success(`${amount} 원을 성공적으로 빌렸습니다!`);
    console.log(res);
    refetch();
  };

  const {
    data,
    isLoading: isDataLoading,
    refetch,
  } = useQuery({
    queryKey: ["bnpl"],
    queryFn: async () => {
      if (user && !isLoading) {
        const res = await aiClient.post("/bnpl/user", {
          user_id: user.userId,
        });
        return res.data.amount;
      }
      return null;
    },
    refetchInterval: 1000,
  });
  return (
    <div className="grid w-full h-screen bg-primary place-content-center">
      <div>
        <h1 className="text-6xl font-bold text-white">한별 Pay</h1>
        <h3 className="mt-6 text-4xl font-bold text-white">
          개인회생자를 위한 간편결제 소액대출 서비스
        </h3>
        <p className="font-bold text-white">
          현재 빌린 금액: {isDataLoading ? "..." : data}원
        </p>
      </div>

      <div className="flex flex-col mt-8 space-y-4">
        <input
          value={name}
          onChange={HandleName}
          className="h-16 p-4 text-3xl font-bold text-white bg-transparent border-2 border-white w-72 focus:outline-none rounded-2xl placeholder:text-white/80"
          placeholder="이름"
        />
        <input
          value={account}
          onChange={HandleAccount}
          className="h-16 p-4 text-3xl font-bold text-white bg-transparent border-2 border-white w-72 focus:outline-none rounded-2xl placeholder:text-white/80"
          placeholder="계좌번호"
        />
        <input
          value={amount}
          onChange={HandleAmount}
          className="h-16 p-4 text-3xl font-bold text-white bg-transparent border-2 border-white w-72 focus:outline-none rounded-2xl placeholder:text-white/80"
          placeholder="금액"
        />
      </div>

      <Button
        onClick={HandleBNPL}
        className="w-20 h-16 mt-4 text-2xl rounded-2xl"
      >
        대출
      </Button>
    </div>
  );
};

export default BNPLPage;
