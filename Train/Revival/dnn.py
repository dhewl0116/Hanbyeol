import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt

# Device 설정
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

# 데이터 로드
file_path = ''  # 실제 파일 경로로 변경
data = pd.read_csv(file_path)

# 레이블 인코딩 ('예' -> 1, '아니오' -> 0)
label_encoder = LabelEncoder()
data['EligibleForRehabilitation'] = label_encoder.fit_transform(data['EligibleForRehabilitation'])

# 학습할 때와 추론할 때 동일한 피처 순서를 유지하도록 정렬
feature_order = ['NumberOfDependents', 'MaritalStatus', 'MonthlyIncome', 'AssetValue', 'TotalDebt']

# 데이터 정렬
X = data[['NumberOfDependents', 'MaritalStatus', 'MonthlyIncome', 'AssetValue', 'TotalDebt']]
y = data['EligibleForRehabilitation']

# 범주형 변수 처리 (One-hot 인코딩 필요 시)
X.loc[:, 'MaritalStatus'] = X['MaritalStatus'].apply(lambda x: 1 if x == '예' else 0)

# 데이터 타입을 float으로 변환
X = X.astype(float)

# 학습 및 테스트 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 텐서로 변환
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1).to(device)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1).to(device)

# 모델 정의
class Revival_DNN(nn.Module):
    def __init__(self):
        super(Revival_DNN, self).__init__()
        self.fc1 = nn.Linear(X_train_tensor.shape[1], 512)  
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        x = self.fc4(x)
        return x

dnn_model = Revival_DNN().to(device)

# 손실 함수와 옵티마이저
criterion = nn.BCEWithLogitsLoss()  # 이진 분류에 적합한 손실 함수
optimizer = optim.Adam(dnn_model.parameters(), lr=0.001, weight_decay=1e-5)

# 학습 과정에서 기록할 데이터 초기화
train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []
train_f1_scores = []
test_f1_scores = []

# 학습 과정
epochs = 1000

for epoch in range(epochs):
    dnn_model.train()
    optimizer.zero_grad()
    
    # Forward
    outputs = dnn_model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    train_losses.append(loss.item())

    # 역전파
    loss.backward()
    optimizer.step()

    # 학습 데이터 정확도 및 F1 점수 계산
    outputs_cpu = torch.sigmoid(outputs).cpu().detach().numpy()
    y_train_pred = (outputs_cpu > 0.5).astype(float)
    train_accuracy = accuracy_score(y_train_tensor.cpu().numpy(), y_train_pred)
    train_f1 = f1_score(y_train_tensor.cpu().numpy(), y_train_pred)
    train_accuracies.append(train_accuracy)
    train_f1_scores.append(train_f1)

    # 테스트 데이터 검증
    dnn_model.eval()
    with torch.no_grad():
        test_outputs = dnn_model(X_test_tensor)
        test_loss = criterion(test_outputs, y_test_tensor)
        test_losses.append(test_loss.item())
        
        test_outputs = torch.sigmoid(test_outputs)
        test_outputs = (test_outputs > 0.5).float()
        
        test_accuracy = accuracy_score(y_test_tensor.cpu().numpy(), test_outputs.cpu().numpy())
        test_f1 = f1_score(y_test_tensor.cpu().numpy(), test_outputs.cpu().numpy())
        test_accuracies.append(test_accuracy)
        test_f1_scores.append(test_f1)

    # 학습 과정 출력
    if (epoch + 1) % 100 == 0 or epoch == 0:
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  Train Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}")
        print(f"  Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}")
        print(f"  Train F1 Score: {train_f1:.4f}, Test F1 Score: {test_f1:.4f}")

# 그래프 시각화
plt.figure(figsize=(15, 10))

# Loss 그래프
plt.subplot(2, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Accuracy 그래프
plt.subplot(2, 2, 2)
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(test_accuracies, label='Test Accuracy')
plt.title('Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# F1 Score 그래프
plt.subplot(2, 2, 3)
plt.plot(train_f1_scores, label='Train F1 Score')
plt.plot(test_f1_scores, label='Test F1 Score')
plt.title('F1 Score Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('F1 Score')
plt.legend()

plt.tight_layout()
plt.show()
