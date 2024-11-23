import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from transformers import ElectraTokenizer, ElectraModel
from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score


device = torch.device("mps")


tokenizer = ElectraTokenizer.from_pretrained('monologg/koelectra-base-v3-discriminator')
ELECTRAModel = ElectraModel.from_pretrained('monologg/koelectra-base-v3-discriminator', return_dict=False)


data = pd.read_csv('data/data.csv')
data_list = data.apply(lambda row: [row['story'], row[1:].tolist()], axis=1)
dataset_train, dataset_test = train_test_split(data_list, test_size=0.1, random_state=42)

class ELECTRADataset(Dataset):
    def __init__(self, dataset):
        self.encodings = tokenizer([enc[0] for enc in dataset], truncation=True, padding=True)
        self.labels = [label[1] for label in dataset]

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

max_len = 64
batch_size = 32
warmup_ratio = 0.1
num_epochs = 10
max_grad_norm = 1
learning_rate =  5e-5

data_train = ELECTRADataset(dataset_train)
data_test = ELECTRADataset(dataset_test)
train_dataloader = DataLoader(data_train, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(data_test, batch_size=batch_size, shuffle=False)


class ELECTRAClassifier(nn.Module):
    def __init__(self, electra, hidden_size=768, num_classes=10, dr_rate=None):
        super(ELECTRAClassifier, self).__init__()
        self.electra = electra
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

model = ELECTRAClassifier(ELECTRAModel, dr_rate=0.5).to(device)

no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)

pos_weight = torch.ones(10).to(device) * 9
loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)


t_total = len(train_dataloader) * num_epochs
warmup_step = int(t_total * warmup_ratio)
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=t_total)

def accuracy(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred > 0.5, dtype=float)
    acc_list = []
    for cate in range(y_pred.shape[1]):
        acc_list.append(accuracy_score(y_true[:, cate], y_pred[:, cate]))
    return sum(acc_list) / len(acc_list)

def get_f1_score(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred > 0.5, dtype=float)
    return f1_score(y_true, y_pred, average='micro')


train_history = {'accuracy': [], 'f1_score': []}
test_history = {'accuracy': [], 'f1_score': []}
loss_history = []

for e in range(num_epochs):
    model.train()
    for batch_id, batch_data in enumerate(train_dataloader):
        input_ids = batch_data['input_ids'].to(device)
        attention_mask = batch_data['attention_mask'].to(device)
        token_type_ids = batch_data['token_type_ids'].to(device)
        labels = batch_data['labels'].float().to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask, token_type_ids)
        
        loss = loss_fn(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()

        sigmoid = nn.Sigmoid()
        y_pred = sigmoid(outputs).cpu().detach().numpy()  # Sigmoid로 확률 변환
        y_pred_binary = (y_pred > 0.5).astype(float)  # 0.5 기준으로 이진화
        print(labels, y_pred_binary)
        # 정확도 및 F1 스코어 계산
        train_acc = accuracy(labels.cpu().detach().numpy(), y_pred_binary)
        train_f1 = get_f1_score(labels.cpu().detach().numpy(), y_pred_binary)

        print(f"[Train] Epoch {e+1}, Batch {batch_id+1}, Loss: {loss.item()}, Acc: {train_acc}, F1: {train_f1}")
        
        train_history['accuracy'].append(train_acc)
        train_history['f1_score'].append(train_f1)
        loss_history.append(loss.item())

    # 평가 루프
    model.eval()
    with torch.no_grad():
        for batch_id, batch_data in enumerate(test_dataloader):
            input_ids = batch_data['input_ids'].to(device)
            attention_mask = batch_data['attention_mask'].to(device)
            token_type_ids = batch_data['token_type_ids'].to(device)
            labels = batch_data['labels'].float().to(device)

            outputs = model(input_ids, attention_mask, token_type_ids)
            test_acc = accuracy(labels.cpu().detach().numpy(), sigmoid(outputs).cpu().detach().numpy())
            test_f1 = get_f1_score(labels.cpu().detach().numpy(), sigmoid(outputs).cpu().detach().numpy())

            print(f"[Test] Epoch {e+1}, Batch {batch_id+1}, Acc: {test_acc}, F1: {test_f1}")
            
            test_history['accuracy'].append(test_acc)
            test_history['f1_score'].append(test_f1)

torch.save(model.state_dict(), 'model/KOelectra.pt')

import matplotlib.pyplot as plt


fig, ax = plt.subplots(figsize=(5, 2.7))
ax.plot(np.arange(len(train_history['accuracy'])), train_history['accuracy'], label='Train Accuracy')
ax.plot(np.arange(len(train_history['f1_score'])), train_history['f1_score'], label='Train F1 Score')
ax.plot(np.arange(len(loss_history)), loss_history, label='Loss')
ax.legend()
plt.show()

def predict(predict_string):
    data = [predict_string, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    dataset_another = [data]

    another_test = ELECTRADataset(dataset_another)
    test_dataloader = torch.utils.data.DataLoader(another_test, batch_size = batch_size, num_workers = 0)

    model = ELECTRAClassifier(ELECTRAModel, dr_rate=0.5).to(device)
    model.load_state_dict(torch.load('model/electra_basic.pt', map_location=device))

    with torch.no_grad():
      for batch_id, batch_data in enumerate(test_dataloader):
        input_ids = batch_data['input_ids'].to(device)
        attention_mask = batch_data['attention_mask'].to(device)
        token_type_ids = batch_data['token_type_ids'].to(device)
        labels = batch_data['labels'].float().to(device)

        outputs = nn.Sigmoid()(model(input_ids, attention_mask, token_type_ids))

    labels = ["coin_fail","stock_fail","gambling_fail","over_consuming", "real_estate_investment","no_income", "crime_damage", "health_problem", "family_problem", "business_fail"]
    return outputs.cpu().detach().numpy()

while True:
  string = input("입력 : ")
  print(predict(string))
