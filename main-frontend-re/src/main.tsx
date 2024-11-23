import { createRoot } from "react-dom/client";
import App from "./App.tsx";
import "./index.css";
import ModalProvider from "./providers/ModalProvider.tsx";
import ToasterProvider from "./providers/ToasterProvider.tsx";

createRoot(document.getElementById("root")!).render(
  <>
    <ModalProvider />
    <ToasterProvider />
    <App />
  </>
);
