# 필요한 라이브러리 import
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd

# 데이터 불러오기
file_path = 'data/df_cleaned5.csv'  # 데이터 파일 경로에 맞게 수정
data = pd.read_csv(file_path)

# 데이터 전처리 (범주형 변수 인코딩 및 one-hot 인코딩)
label_encoder = LabelEncoder()
data['EligibleForRehabilitation'] = label_encoder.fit_transform(data['EligibleForRehabilitation'])

# 입력(X)과 출력(y) 분리
X = data.drop(columns=['EligibleForRehabilitation'])
y = data['EligibleForRehabilitation']

# 범주형 변수에 대한 원핫 인코딩 처리
X = pd.get_dummies(X, drop_first=True)

# 데이터셋을 학습용과 테스트용으로 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# KNN 분류기 생성 (k=5 예시, 필요시 k값 조정 가능)
knn_model = KNeighborsClassifier(n_neighbors=5)

# 모델 학습
knn_model.fit(X_train, y_train)

# 테스트 데이터로 예측
y_pred = knn_model.predict(X_test)

# 정확도 계산
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"KNN Accuracy: {accuracy}")
print(f"KNN F1 Score: {f1}")
