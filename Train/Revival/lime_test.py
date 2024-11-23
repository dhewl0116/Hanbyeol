import torch
from torch import nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from lime.lime_tabular import LimeTabularExplainer

# Revival_DNN 모델 정의
class Revival_DNN(nn.Module):
    def __init__(self):
        super(Revival_DNN, self).__init__()
        self.fc1 = nn.Linear(5, 512)
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

# 모델 로드
model = Revival_DNN()
model.load_state_dict(torch.load("model/Revival Cal acc:99.74 f1:99.80.pt", map_location=torch.device('cpu'), weights_only=True))
model.eval()

# 학습 데이터 불러오기 및 전처리
file_path = "data/df_cleaned5.csv"  # 실제 파일 경로
dataframe = pd.read_csv(file_path)

# 범주형 데이터 인코딩 (MaritalStatus)
label_encoder = LabelEncoder()
dataframe['MaritalStatus'] = label_encoder.fit_transform(dataframe['MaritalStatus'])

# 필요한 열만 선택 (레이블 제외)
features = ['NumberOfDependents', 'MaritalStatus', 'MonthlyIncome', 'AssetValue', 'TotalDebt']
processed_data = dataframe[features].to_numpy(dtype=np.float32)

# LIME 설명기 초기화를 위해 학습 데이터 사용
explainer = LimeTabularExplainer(
    processed_data,
    feature_names=features,
    class_names=['Success Probability'],
    discretize_continuous=True
)

# LIME 설명을 위한 예측 함수 정의
def predict_fn(x):
    x = torch.tensor(x, dtype=torch.float32)
    with torch.no_grad():
        logits = model(x).numpy()
    probs = torch.sigmoid(torch.tensor(logits)).numpy()  # 시그모이드 적용
    return np.hstack([1 - probs, probs])  # 두 클래스의 확률을 반환

# 예측 및 설명 함수 정의
def predict_and_explain(sample):
    # 모델에 입력하여 결과 예측
    sample_tensor = torch.tensor(sample, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        prediction = model(sample_tensor).item()
    probability = torch.sigmoid(torch.tensor(prediction)).item()
    
    # LIME 해석 생성
    exp = explainer.explain_instance(np.array(sample), predict_fn, num_features=5)
    explanation = exp.as_list()
    
    # 결과 및 해석 출력
    print(f"Input Sample: {np.array(sample, dtype=np.float32).tolist()}")
    print(f"Predicted Success Probability: {probability*100:.2f}")
    print("Explanation:")
    for feature, value in explanation:
        print(f"  {feature}: {value}")

# 사용 예시 - 임의의 샘플 데이터 입력
sample_input = [1, 0, 10, 300, 6000]
predict_and_explain(sample_input)
