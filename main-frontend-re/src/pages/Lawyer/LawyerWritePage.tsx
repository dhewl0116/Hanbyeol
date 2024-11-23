import { ChangeEvent, useState } from "react";
import Button from "../../components/Button/Button";
import { useNavigate } from "react-router";
import aiClient from "../../lib/api/aiClient";
import { useUser } from "../../hooks/useUser";

const LawyerWritePage = () => {
  const navigate = useNavigate();
  const [value, setValue] = useState("");
  const { user } = useUser();

  const onChange = (e: ChangeEvent<HTMLTextAreaElement>) => {
    setValue(e.target.value);
  };
  const HandleSend = async () => {
    const res = await aiClient.post("/ai/vec", {
      user_id: user.userId,
      sentence: value,
    });
    console.log(res);
    navigate("/lawyer");
    return res;
  };

  return (
    <div className="grid w-full h-screen bg-primary place-content-center">
      <form>
        <div className="flex flex-col items-center w-full px-16 py-8 h-fit bg-secondary rounded-3xl">
          <h1 className="text-3xl font-bold">
            유저들을 도와줄 수 있게 간단한 정보를 입력해주세요!
          </h1>
          <textarea
            className="w-[700px] h-[400px] resize-none focus:outline-none mt-8 p-4 text-xl"
            value={value}
            onChange={onChange}
          />
          <div className="flex self-end mt-8 gap-x-8">
            <Button
              onClick={() =>
                navigate(
                  "/lawyer/applicant_detail/01e0b08c-98bf-460b-a887-bd24b6ff1cea"
                )
              }
              className="border-none"
              secondary
            >
              뒤로가기
            </Button>
            <Button onClick={HandleSend}>사연 보내기</Button>
          </div>
        </div>
      </form>
    </div>
  );
};

export default LawyerWritePage;
