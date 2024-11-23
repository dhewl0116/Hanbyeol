import { FC } from "react";
import Button from "./Button/Button";
import { twMerge } from "tailwind-merge";
import { useNavigate } from "react-router";
import HumanPhoto from "../assets/HumanPhoto.png";

interface Props {
  id: string;
  username: string;
  bio: string;
  className?: string;
}

const Lawyer: FC<Props> = ({ id, username, bio, className }) => {
  const navigate = useNavigate();
  return (
    <div
      className={twMerge(
        "w-[840px] h-[300px] grid grid-cols-2 grid-rows-1",
        className
      )}
    >
      <img
        src={HumanPhoto}
        className="grid place-content-center w-[353px] h-[300px]"
      />

      <div className="flex flex-col justify-center pl-12">
        <h1 className="text-[40px] font-bold">{username}</h1>
        <p className="text-fontlight">{bio}</p>
        <Button
          className="text-2xl rounded-2xl w-60 h-[74px] border-none mt-8"
          secondary
          onClick={() => {
            navigate(`./${id}`);
          }}
        >
          보기
        </Button>
      </div>
    </div>
  );
};

export default Lawyer;
