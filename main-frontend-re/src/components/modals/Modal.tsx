import * as Dialog from "@radix-ui/react-dialog";
import { FC, ReactNode } from "react";
import { twMerge } from "tailwind-merge";

interface Props {
  title: string;
  children: ReactNode;
  onChange: (open: boolean) => void;
  isOpen: boolean;
  titleBig?: boolean;
}

const Modal: FC<Props> = ({
  title,
  onChange,
  isOpen,
  children,
  titleBig = false,
}) => {
  return (
    <Dialog.Root open={isOpen} defaultOpen={isOpen} onOpenChange={onChange}>
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 z-20 bg-fontlight" />
        <Dialog.Content className="fixed z-30 top-1/2 left-1/2 translate-x-[-50%] translate-y-[-50%] w-[500px] h-fit px-8 pt-8 pb-16 bg-secondary rounded-2xl">
          <Dialog.Title
            className={twMerge(
              `${titleBig ? "text-3xl" : "text-2xl"}`,
              "font-bold"
            )}
          >
            {title}
          </Dialog.Title>
          <div>{children}</div>
          <Dialog.Close asChild>
            <button className="absolute p-2 mt-4 font-bold top-4 right-8 text-fontgrey bg-slate-400">
              닫기
            </button>
          </Dialog.Close>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
};

export default Modal;
