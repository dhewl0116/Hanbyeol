import { forwardRef, InputHTMLAttributes } from "react";
import { twMerge } from "tailwind-merge";

interface Props extends InputHTMLAttributes<HTMLInputElement> {
  className?: string;
}

const Input = forwardRef<HTMLInputElement, Props>(
  ({ className, type = "text", ...props }, ref) => (
    <input
      type={type}
      className={twMerge(
        `border border-gray-300 rounded-lg p-3 text-sm font-bold w-full mx-auto`,
        className
      )}
      ref={ref}
      {...props}
    />
  )
);

export default Input;
