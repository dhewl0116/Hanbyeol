import { useNavigate } from "react-router";
import Button from "../components/Button/Button";
import Board1 from "../assets/board1.png";
import Board2 from "../assets/board2.png";
import Board3 from "../assets/board3.png";

const BoardPage = () => {
  const navigate = useNavigate();
  return (
    <>
      <div className="w-full h-[786px] bg-primary pt-64 flex flex-col items-center relative">
        <div className="w-fit">
          <h1 className="text-white font-bold text-[40px]">
            한별에서 도움을 받으세요!
          </h1>
          <div className="flex mt-8 gap-x-6">
            <div className="bg-white w-[353px] h-[245px] rounded-2xl grid place-items-center relative">
              <p className="relative text-2xl text-center bottom-4">
                나랑 비슷한 사람들은 <br />
                회생했을까?
              </p>
              <Button
                onClick={() => navigate("/possibility/start")}
                className="w-[303px] h-[60px] absolute bottom-4 text-xl font-normal rounded-2xl"
                secondary
              >
                회생가능성 계산하기
              </Button>
            </div>
            <div className="bg-white w-[353px] h-[245px] rounded-2xl grid place-items-center relative">
              <p className="relative text-2xl text-center bottom-4">
                간단한 질문에 답하고 <br />
                안심되고 나에게 딱 맞는 <br />
                변호사 추천받기
              </p>
              <Button
                onClick={() => navigate("/write")}
                className="w-[303px] h-[60px] absolute bottom-4 text-xl font-normal rounded-2xl"
                secondary
              >
                30초만에 변호사 매칭
              </Button>
            </div>
            {/* <div className="bg-white w-[353px] h-[245px] rounded-2xl grid place-items-center relative">
              <p className="relative text-2xl text-center bottom-4">
                AI로 부채 상환내역 기록하고 <br />
                불법사금융 정리하기
              </p>
              <Button
                onClick={() => navigate("/pay")}
                className="w-[303px] h-[60px] absolute bottom-4 text-xl font-normal rounded-2xl"
                secondary
              >
                불법 사채 끝장내기
              </Button>
            </div> */}
          </div>
        </div>
        <h1 className="absolute text-5xl font-bold text-white bottom-16">
          이런 경우에도 회생이 가능해요
        </h1>
      </div>
      <div className="flex flex-col items-center w-full py-24 h-fit bg-secondary gap-y-8">
        <div className="flex items-center gap-x-16">
          <img src={Board1} />
          <div className="flex flex-col">
            <h1 className="text-[40px] font-bold text-right">
              주식/코인 빛 회생 가능
            </h1>
            <p className="text-right text-fontlight">
              amet, consectetur adipiscing elit. Morbi placerat pellentesque
              <br />
              interdum. Ut varius est ac justo pulvinar, sed
              <br />
              posuere purus gravida. Mauris ac nunc q
            </p>
          </div>
        </div>
        <div className="flex items-center gap-x-16">
          <div className="flex flex-col">
            <h1 className="text-[40px] font-bold">도박/여행 빛 회생 가능</h1>
            <p className="text-fontlight">
              amet, consectetur adipiscing elit. Morbi placerat pellentesque
              <br />
              interdum. Ut varius est ac justo pulvinar, sed
              <br />
              posuere purus gravida. Mauris ac nunc q
            </p>
          </div>
          <img src={Board2} />
        </div>
      </div>
      <div className="w-full h-[900px] flex flex-col items-center bg-primary">
        <h1 className="mt-24 text-5xl font-bold text-center text-white">
          그 외의 경우에도 <br />
          한별에서는 다 가능!
        </h1>
        <p className="mt-8 text-center text-white">
          추가적인 요소들을 나열하는 설명 문구추가적인 요소들을 나열하는 설명
          문구추가적인 요소들을
          <br />
          나열하는 설명 문구추가적인 요소들을 나열하는 설명 문구
        </p>
        <h2 className="font-bold text-white text-[40px] mt-32 relative right-72">
          현재 한별에 등록중인 변호사
        </h2>
        <img src={Board3} className="mt-12" />
      </div>
    </>
  );
};

export default BoardPage;
