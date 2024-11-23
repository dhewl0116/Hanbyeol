# 필요한 라이브러리 import
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd

# 데이터 불러오기
file_path = 'data/df_cleaned5.csv'  # 파일 경로에 맞게 수정
data = pd.read_csv(file_path)

# 데이터 전처리 (타겟 변수 인코딩 및 범주형 변수 원핫 인코딩)
label_encoder = LabelEncoder()
data['EligibleForRehabilitation'] = label_encoder.fit_transform(data['EligibleForRehabilitation'])

# 입력(X)과 출력(y) 분리
X = data.drop(columns=['EligibleForRehabilitation'])
y = data['EligibleForRehabilitation']

# 범주형 변수에 대해 원핫 인코딩 처리
X = pd.get_dummies(X, drop_first=True)

# 학습 및 테스트 데이터셋 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest 모델 정의
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# 모델 학습
rf_model.fit(X_train, y_train)

# 테스트 데이터로 예측
y_pred = rf_model.predict(X_test)

# 정확도 및 F1 점수 계산
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Random Forest Accuracy: {accuracy}")
print(f"Random Forest F1 Score: {f1}")
