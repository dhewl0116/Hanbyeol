import { BrowserRouter, Outlet, Route, Router, Routes } from "react-router-dom";
import Header from "./components/Header";
import MainPage from "./pages/MainPage";
import LoginPage from "./pages/LoginPage";
import RegisterPage from "./pages/RegisterPage";
import BoardPage from "./pages/BoardPage";
import PossibilityStart from "./pages/Possibility/Start";
import PossibilitySpouse from "./pages/Possibility/Spouse";
import PossibilityMonthlyIncome from "./pages/Possibility/MonthlyIncome";
import PossibilityAssetValues from "./pages/Possibility/AssetValues";
import WritePage from "./pages/Match/WritePage";
import PossibilityDebt from "./pages/Possibility/Debt";
import PossibilityResult from "./pages/Possibility/Result";
import LawyerPage from "./pages/Match/LawyerPage";
import PossibilityNumberOfDependents from "./pages/Possibility/NumberOfDependents";
import ChatPage from "./pages/ChatPage";
import ChooseRolePage from "./pages/ChooseRolePage";
import ApplicantDetailPage from "./pages/Lawyer/ApplicantDetailPage";
import LawyerDetailPage from "./pages/Match/LawyerDetailPage";
import SuccessPage from "./pages/Match/SuccessPage";
import RecoveryStatus from "./pages/RecoveryStatus";
import LawyerWritePage from "./pages/Lawyer/LawyerWritePage";
import LawyerHome from "./pages/Lawyer/LawyerHome";
import BNPLPage from "./pages/BNPLPage";

const Layout = () => (
  <>
    <Header />
    <Outlet />
  </>
);

const AppRouter = () => (
  <BrowserRouter>
    <Routes>
      <Route element={<Layout />}>
        <Route path="/" element={<MainPage />} />
        <Route path="/chooserole" element={<ChooseRolePage />} />
        <Route path="/login" element={<LoginPage />} />
        <Route path="/register" element={<RegisterPage />} />
        <Route path="/board" element={<BoardPage />} />
        <Route path="/write" element={<WritePage />} />
        <Route path="/pay" element={<BNPLPage />} />

        <Route path="/lawyer_result/*">
          <Route path="" element={<LawyerPage />} />
          <Route path=":id" element={<LawyerDetailPage />} />
          <Route path="success" element={<SuccessPage />} />
          <Route path=":id/chat" element={<ChatPage />} />
          <Route path=":id/recoverystatus" element={<RecoveryStatus />} />
        </Route>

        <Route path="/lawyer/*">
          <Route path="" element={<LawyerHome />} />
          <Route path="write" element={<LawyerWritePage />} />
          <Route
            path="applicant_detail/:id"
            element={<ApplicantDetailPage />}
          />
          <Route
            path="applicant_detail/:id/recoverystatus"
            element={<RecoveryStatus />}
          />
          <Route path="applicant_detail/:id/chat" element={<ChatPage />} />
        </Route>

        <Route path="/possibility/*">
          <Route path="start" element={<PossibilityStart />} />
          <Route path="children" element={<PossibilityNumberOfDependents />} />
          <Route path="spouse" element={<PossibilitySpouse />} />
          <Route path="monthlyincome" element={<PossibilityMonthlyIncome />} />
          <Route path="assetvalues" element={<PossibilityAssetValues />} />
          <Route path="debtor" element={<PossibilityDebt />} />
          <Route path="result" element={<PossibilityResult />} />
        </Route>
      </Route>
    </Routes>
  </BrowserRouter>
);

export default AppRouter;
