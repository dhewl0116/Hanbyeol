import torch
from transformers import ElectraTokenizer, ElectraModel
from torch import nn
import torch
from torch.utils.data import Dataset

class ELECTRADataset(Dataset):
    def __init__(self, dataset):
        tokenizer = ElectraTokenizer.from_pretrained('monologg/koelectra-base-v3-discriminator')
        self.encodings = tokenizer([enc[0] for enc in dataset], truncation=True, padding=True)
        self.labels = [label[1] for label in dataset]

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


class ELECTRAClassifier(nn.Module):
    def __init__(self,  hidden_size=768, num_classes=10, dr_rate=None):
        super(ELECTRAClassifier, self).__init__()
        self.electra = ElectraModel.from_pretrained('monologg/koelectra-base-v3-discriminator', return_dict=False)
        self.dr_rate = dr_rate
        self.classifier = nn.Linear(hidden_size, num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)

    def forward(self, token_ids, attention_mask, segment_ids):
        out = self.electra(
            input_ids=token_ids,
            attention_mask=attention_mask,
            token_type_ids=segment_ids.long()
        )
        out = out[0][:, 0, :]  
        if self.dr_rate:
            out = self.dropout(out)
        return self.classifier(out)

class Maching_Classifier:
    def __init__(self, model_path) -> None:
        self.model = ELECTRAClassifier(dr_rate=0.5)
        self.model.load_state_dict(torch.load(model_path, weights_only=True))

    def predict(self, predict_string):
        data = [predict_string, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
        dataset_another = [data]

        another_test = ELECTRADataset(dataset_another)
        test_dataloader = torch.utils.data.DataLoader(another_test, batch_size = 32, num_workers = 0)

        with torch.no_grad():
            for batch_id, batch_data in enumerate(test_dataloader):
                input_ids = batch_data['input_ids']
                attention_mask = batch_data['attention_mask']
                token_type_ids = batch_data['token_type_ids']
                labels = batch_data['labels'].float()

                outputs = nn.Sigmoid()(self.model(input_ids, attention_mask, token_type_ids))
                
        outputs = torch.where(outputs < 0.1, torch.tensor(0.0), outputs)

        return outputs.cpu().detach().numpy()


# if __name__ == "__main__":
#     clasifier = Maching_Classifier()
#     print(clasifier.predict("안녕하세요 주식과 코인을 동시에 하다가 망해서 회생을 신청하려합니다."))