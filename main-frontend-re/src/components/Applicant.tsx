import { FC } from "react";
import { useNavigate } from "react-router";

interface Props {
  id: string;
  username: string;
  created_at: string;
  accepted: boolean;
}

const Applicant: FC<Props> = ({
  id,
  username,
  created_at,
  accepted = false,
}) => {
  const navigate = useNavigate();
  return (
    <div
      onClick={() =>
        navigate(`./applicant_detail/${id}`, {
          state: {
            accepted,
          },
        })
      }
      className="w-[600px] h-32 bg-secondary rounded-2xl p-6"
    >
      <h1 className="text-4xl font-bold">{username}</h1>
      <p className="text-fontlight">{created_at}</p>
    </div>
  );
};

export default Applicant;
