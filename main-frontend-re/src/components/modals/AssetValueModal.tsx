import useAssetValueModalStore from "../../lib/store/useAssetValueModalStore";
import Modal from "./Modal";

const AssetValueModal = () => {
  const { isOpen, onClose } = useAssetValueModalStore();
  const onChange = (open: boolean) => {
    if (!open) {
      onClose();
    }
  };
  return (
    <Modal
      isOpen={isOpen}
      onChange={onChange}
      title="재산에는 다음과 같은 것들이 있어요"
    >
      <div className="ml-24">
        <ul className="mt-12 text-xl font-bold list-disc text-fontlight">
          <li>예금 잔고</li>
          <li>보험 해약 환급금</li>
          <li>전세/월세 보증금</li>
          <li>차량</li>
          <li>부동산</li>
          <li>예상 퇴직금</li>
          <li>빌려주고 못 받은 돈</li>
          <li>판매하고 못 받은 돈</li>
          <li>사업용 설비나 재고</li>
          <li>주식과 코인의 현재 시점 가치</li>
        </ul>
      </div>
    </Modal>
  );
};

export default AssetValueModal;
