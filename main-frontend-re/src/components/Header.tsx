import { useNavigate } from "react-router-dom";
import { memo } from "react";
import { useUser } from "../hooks/useUser";
import toast from "react-hot-toast";

const Header = () => {
  const navigate = useNavigate();
  const { user, isLoading, logout } = useUser();
  console.log(user, isLoading);
  const checkLogin = () => {
    if (user) {
      toast.success(user.username);
      return;
    }
    navigate("/login");
  };
  const HandleLogout = () => {
    logout();
    navigate("/");
  };
  return (
    <div className="flex items-center bg-primary h-[60px] z-10 absolute top-0 w-full">
      <div className="flex justify-between w-full mx-[300px]">
        <span
          className="text-[24px] text-white font-bold cursor-pointer"
          onClick={() => navigate("/")}
        >
          HanByul
        </span>
        <div className="flex justify-between items-center gap-[40px]">
          <span
            className="font-bold text-white cursor-pointer"
            onClick={checkLogin}
          >
            {user ? `${user.username}님` : "로그인"}
          </span>
          <span onClick={HandleLogout} className="font-bold text-white">
            {user ? "로그아웃" : "도움말"}
          </span>
        </div>
      </div>
    </div>
  );
};

export default memo(Header);
