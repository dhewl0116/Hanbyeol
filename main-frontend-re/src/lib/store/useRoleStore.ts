import { create } from "zustand";

interface RoleStore {
  role: string | null;
  setRole: (role: string | null) => void;
}

const useRoleStore = create<RoleStore>((set) => ({
  role: null,
  setRole: (role) => set({ role }),
}));

export default useRoleStore;
