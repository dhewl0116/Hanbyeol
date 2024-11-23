import { ChangeEvent, useState } from "react";
import Button from "../../components/Button/Button";
import { useNavigate } from "react-router";
import aiClient from "../../lib/api/aiClient";

const WritePage = () => {
  const navigate = useNavigate();
  const [value, setValue] = useState<string>("");

  console.log(value);

  const onChange = (e: ChangeEvent<HTMLTextAreaElement>) => {
    setValue(e.target.value);
  };

  const onMatch = async () => {
    const res = await aiClient.post("/ai/matching", {
      sentence: value,
    });

    if (res.status === 200) {
      console.log(res.data);
      navigate("/lawyer_result", {
        state: {
          data: res.data,
        },
      });
    }
  };

  return (
    <div className="grid w-full h-screen bg-primary place-content-center">
      <form>
        <div className="flex flex-col items-center w-full px-16 py-8 h-fit bg-secondary rounded-3xl">
          <h1 className="text-3xl font-bold">
            변호사님들과 매칭 할 수 있게 간단한 정보를 입력해주세요!
          </h1>
          <textarea
            className="w-[700px] h-[400px] resize-none focus:outline-none mt-8 p-4 text-xl"
            value={value}
            onChange={onChange}
          />
          <div className="flex self-end mt-8 gap-x-8">
            <Button
              onClick={() => navigate("/board")}
              className="border-none"
              secondary
            >
              뒤로가기
            </Button>
            <Button onClick={onMatch}>사연 보내기</Button>
          </div>
        </div>
      </form>
    </div>
  );
};

export default WritePage;
