import { FC } from "react";
import { twMerge } from "tailwind-merge";

interface Props {
  content: string;
  writerId: string;
  myId: string;
}

const Message: FC<Props> = ({ content, writerId, myId }) => {
  return (
    <div
      className={twMerge(
        "w-fit h-fit rounded-2xl px-4 py-3 font-bold text-[17px] shadow-[0_4px_4px_0_rgba(0,0,0,0.1)]",
        writerId === myId
          ? "bg-[#f1f1f1] text-fontgrey self-end"
          : "bg-primary text-white self-start"
      )}
    >
      {content}
    </div>
  );
};

export default Message;
