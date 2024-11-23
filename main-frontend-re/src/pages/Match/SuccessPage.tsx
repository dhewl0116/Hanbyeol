import { useNavigate } from "react-router";
import Success from "../../assets/success.png";
import Button from "../../components/Button/Button";

function SuccessPage() {
  const navigate = useNavigate();
  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-primary">
      <div className="flex flex-col items-center text-center text-white">
        <h1 className="text-3xl font-bold">신청이 완료되었습니다</h1>
        <p className="mt-2 text-2xl">
          변호사가 신청을 승인할때까지 기다려주세요
        </p>
        <img src={Success} alt="success" className="w-24 h-24 mt-6" />
      </div>

      <Button onClick={() => navigate("/")} className="mt-24">
        나가기
      </Button>
    </div>
  );
}

export default SuccessPage;
