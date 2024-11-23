import { ButtonHTMLAttributes, forwardRef, ReactNode } from "react";
import { twMerge } from "tailwind-merge";

interface Props extends ButtonHTMLAttributes<HTMLButtonElement> {
  children: ReactNode;
  activated?: boolean;
}

const PossibilityButton = forwardRef<HTMLButtonElement, Props>(
  ({ children, activated, type = "button", onClick, ...props }, ref) => (
    <button
      type={type}
      onClick={onClick}
      className={twMerge(
        `${activated ? "border-primary" : "border-fontlight"}`,
        "w-[104px] h-[68px] border-2 rounded-xl transition font-bold"
      )}
      ref={ref}
      {...props}
    >
      {children}
    </button>
  )
);

PossibilityButton.displayName = "PossibilityButton";

export default PossibilityButton;
