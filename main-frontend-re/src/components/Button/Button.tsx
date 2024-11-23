import { ButtonHTMLAttributes, forwardRef, memo, ReactNode } from "react";
import { twMerge } from "tailwind-merge";

interface Props extends ButtonHTMLAttributes<HTMLButtonElement> {
  children?: ReactNode;
  className?: string;
  secondary?: boolean;
}

const Button = forwardRef<HTMLButtonElement, Props>(
  ({ children, className, type = "button", secondary, ...props }, ref) => (
    <button
      type={type}
      className={twMerge(
        `w-[175px] h-[50px] rounded-[25px] border-2 font-bold ${
          !secondary
            ? "bg-white text-primary border-primary"
            : "bg-primary text-white border-white"
        }`,
        className
      )}
      ref={ref}
      {...props}
    >
      {children}
    </button>
  )
);
Button.displayName = "Button";
export default memo(Button);
