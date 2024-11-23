import { useEffect, useState } from "react";
import nestClient from "../../lib/api/nestClient";
import { useUser } from "../../hooks/useUser";
import Button from "../../components/Button/Button";
import Applicant from "../../components/Applicant";
import "../../index.css";

interface Applicant {
  id: string;
  lawyer_id: string;
  applicant_id: string;
  status: string;
  created_at: string;

  applicant: {
    id: string;
    email: string;
    username: string;
    password: string;
    role: string;
    description: string;
    created_at: string;
  };
}

const LawyerHome = () => {
  const [accepted, setAccepted] = useState<Applicant[]>([]);
  const [unaccepted, setUnaccepted] = useState<Applicant[]>([]);
  const [selected, setSelected] = useState<string>("accepted");
  const { user, isLoading } = useUser();

  const AcceptedClick = () => {
    setSelected("accepted");
  };

  const UnacceptedClick = () => {
    setSelected("unaccepted");
  };

  useEffect(() => {
    if (!isLoading && user) {
      const fetchData = async () => {
        const applicants = await nestClient
          .get("/user/get_applicants")
          .then((r) => r.data);

        const unAccepted = applicants.filter(
          (x: Applicant) => x.status === "not accepted now"
        );
        console.log(unAccepted);
        setUnaccepted(unAccepted);

        const accepted = applicants.filter(
          (x: Applicant) => x.status !== "not accepted now"
        );
        console.log(accepted);
        setAccepted(accepted);
      };
      fetchData();
    }
  }, [isLoading, user]);

  console.log(accepted);

  return (
    <div className="grid w-full h-screen bg-primary place-content-center">
      <div className="w-[1000px] h-[600px] bg-white flex flex-col rounded-3xl">
        <div className="self-center mt-8 space-x-8">
          <Button
            className="w-96 rounded-xl"
            secondary={selected === "accepted"}
            onClick={AcceptedClick}
          >
            회생을 도와주고 있는 사람들
          </Button>
          <Button
            className="w-96 rounded-xl"
            secondary={selected === "unaccepted"}
            onClick={UnacceptedClick}
          >
            나에게 회생을 신청한 사람들
          </Button>
        </div>
        <div className="flex flex-col items-center mt-4 gap-y-4 h-[480px] overflow-y-scroll scrollbar-hide">
          {selected === "accepted" ? (
            <>
              {" "}
              {accepted.map((value) => (
                <Applicant
                  accepted
                  id={value.applicant_id}
                  username={value.applicant.username}
                  created_at={value.created_at}
                  key={value.applicant_id}
                />
              ))}
            </>
          ) : (
            <>
              {unaccepted.map((value) => (
                <Applicant
                  accepted={false}
                  id={value.applicant_id}
                  username={value.applicant.username}
                  created_at={value.created_at}
                  key={value.applicant_id}
                />
              ))}
            </>
          )}
        </div>
      </div>
    </div>
  );
};

export default LawyerHome;
