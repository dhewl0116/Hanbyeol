import { forwardRef, InputHTMLAttributes } from "react";
import { twMerge } from "tailwind-merge";

interface Props extends InputHTMLAttributes<HTMLInputElement> {
  className?: string;
}

const PossibilityInput = forwardRef<HTMLInputElement, Props>(
  ({ className, ...props }, ref) => (
    <div className="flex items-center mt-8 gap-x-2">
      <input
        type="text"
        placeholder="금액을 입력해주세요."
        className={twMerge(
          `text-2xl border-b border-black w-72 font-bold focus:outline-none`,
          className
        )}
        ref={ref}
        {...props}
      />
      <span className="text-2xl font-bold">만원</span>
    </div>
  )
);

PossibilityInput.displayName = "PossibilityInput";

export default PossibilityInput;
