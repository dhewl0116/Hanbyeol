import { Toaster } from "react-hot-toast";

const ToasterProvider = () => {
  return (
    <Toaster
      toastOptions={{
        duration: 2000,
      }}
    />
  );
};

export default ToasterProvider;
