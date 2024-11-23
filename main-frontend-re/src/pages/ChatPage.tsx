import { ChangeEvent, useEffect, useRef, useState } from "react";
import Message from "../components/Message";
import { useParams } from "react-router";
import nestClient from "../lib/api/nestClient";
import { LawyerType } from "../types";
import { useUser } from "../hooks/useUser";
import { useQuery } from "@tanstack/react-query";

interface JsonResponse {
  id: string;
  content: string;
  match_id: string;
  user_id: string;
  created_at: string;
}

const ChatPage = () => {
  const { id } = useParams();
  const { user } = useUser();
  const [lawyer, setLawyer] = useState<LawyerType | null>(null);
  const [message, setMessage] = useState<string>("");
  const [matchId, setMatchId] = useState<string | null>(null);
  const sendRef = useRef<HTMLDivElement>(null);

  const {
    data: allMessages,
    isLoading,
    refetch,
  } = useQuery({
    queryKey: ["message"],
    queryFn: async () => {
      if (matchId) {
        const messageRes = await nestClient.get(
          `/chat/get_messages/${matchId}`
        );
        console.log(messageRes.data);
        return messageRes.data as JsonResponse[];
      }
      return [];
    },
    refetchInterval: 1000,
  });

  const SendMessage = async () => {
    try {
      await nestClient.post("/chat/send_message", {
        match_id: matchId,
        content: message,
      });
      setMessage("");
      refetch();
    } catch (err) {
      console.error("Something went wrong: ", err);
    }
  };

  useEffect(() => {
    if (matchId) {
      refetch(); // matchId가 설정된 직후 refetch 호출
    }
  }, [matchId, refetch]);

  useEffect(() => {
    const onEnter = (e: KeyboardEvent) => {
      if (e.key === "Enter") {
        sendRef.current?.click();
        SendMessage();
      }
    };
    window.addEventListener("keydown", onEnter);
    return () => {
      window.removeEventListener("keydown", onEnter);
    };
  }, []);

  useEffect(() => {
    const fetchData = async () => {
      const lawyerRes = await nestClient.get(`/user/${id}`);
      setLawyer(lawyerRes.data);

      const matchRes = await nestClient.get(`/match/${id}`).then((r) => r.data);
      setMatchId(matchRes.id);
    };
    fetchData();
  }, [id]);

  return (
    <>
      <div className="w-full h-[75vh] bg-primary">
        <div className="w-[550px] h-[800px] rounded-3xl p-[30px] bg-white absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2">
          <div className="grid w-full h-16 text-xl font-bold place-content-center rounded-2xl bg-[#efefef]">
            {lawyer?.username} 님과 대화중..
          </div>
          <div className="flex flex-col mt-12 h-[540px] overflow-y-scroll gap-y-4 scrollbar-hide">
            {!isLoading &&
              allMessages?.map((message) => (
                <Message
                  key={message.id}
                  content={message.content}
                  writerId={message.user_id}
                  myId={user.userId}
                />
              ))}
          </div>
          <div className="w-full h-[75px] py-4 pr-4 border-4 mt-4 border-primary shadow-[0_4px_4px_0_rgba(0,0,0,0.1)] rounded-2xl flex items-center">
            <input
              value={message}
              onChange={(e: ChangeEvent<HTMLInputElement>) =>
                setMessage(e.target.value)
              }
              className="flex-1 px-4 py-2 text-[17px] font-bold focus:outline-none"
            />
            <div
              onClick={SendMessage}
              className="grid w-12 h-12 rounded-2xl bg-primary place-content-center"
              ref={sendRef}
            >
              <p className="font-bold text-white">전송</p>
            </div>
          </div>
        </div>
      </div>
      <div className="w-full h-[25vh] bg-secondary"></div>
    </>
  );
};

export default ChatPage;
