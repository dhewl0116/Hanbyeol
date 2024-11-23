import { create } from "zustand";

interface PossibilityStore {
  AssetValue: number | null; // 총자산
  setAssetValue: (value: number) => void;

  MaritalStatus: string | null; // 결혼 여부 (결혼, 이혼, 미혼)
  setMaritalstatus: (value: string) => void;

  MonthlyIncome: number | null; // 월 평균 수입
  setMonthlyIncome: (value: number) => void;

  TotalDebt: number | null; // 모든 채무의 원금 합계
  setTotalDebt: (value: number) => void;

  NumberOfDependents: number | null; // 부양 가족수
  setNumberOfDependents: (value: number) => void;
}

const usePossibilityStore = create<PossibilityStore>((set) => ({
  AssetValue: null,
  setAssetValue: (value) => set({ AssetValue: value }),

  MaritalStatus: null,
  setMaritalstatus: (value) => set({ MaritalStatus: value }),

  MonthlyIncome: null,
  setMonthlyIncome: (value) => set({ MonthlyIncome: value }),

  TotalDebt: null,
  setTotalDebt: (value) => set({ TotalDebt: value }),

  NumberOfDependents: null,
  setNumberOfDependents: (value) => set({ NumberOfDependents: value }),
}));

export default usePossibilityStore;
