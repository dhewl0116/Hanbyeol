import { useEffect, useState } from "react";
import nestClient from "../lib/api/nestClient";

export const useUser = () => {
  const [user, setUser] = useState<any>(null);
  const [isLoading, setIsLoading] = useState(true);
  useEffect(() => {
    setIsLoading(true);
    const checkUser = async () => {
      const token = localStorage.getItem("access_token");
      if (token) {
        try {
          const res = await nestClient.get("/auth/profile");

          setUser(res.data);
          return res.data;
        } catch (err) {
          console.error(err);
        }
      }
      setIsLoading(false);
      console.log(isLoading);
    };
    checkUser();
    setIsLoading(false);
  }, []);

  const logout = () => {
    localStorage.removeItem("access_token");
    setUser(null);
    window.location.reload();
  };

  return {
    user,
    isLoading,
    logout,
  };
};
