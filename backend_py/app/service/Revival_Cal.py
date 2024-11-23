import torch
from torch import nn
import numpy as np
import pandas as pd
import random
from lime.lime_tabular import LimeTabularExplainer
from openai import OpenAI
import json

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


class Revival_Calculator:
    def __init__(self, model_path, data_path):
        np.set_printoptions(precision=6, suppress=True)
        torch.set_printoptions(precision=10)

        print("OpenAI 클라이언트 초기화 중...")
        self.client = OpenAI(api_key="")

        print("딥러닝 모델 초기화 중...")
        self.model = Revival_DNN()
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('mps'), weights_only=True))
        self.model.eval()
        print("모델 로드 완료.")

        print("데이터 로드 및 전처리 중...")
        dataframe = pd.read_csv(data_path)
        marital_status_map = {'미혼': 0, '기혼': 1}
        dataframe['MaritalStatus'] = dataframe['MaritalStatus'].map(marital_status_map)
        features = ['NumberOfDependents', 'MaritalStatus', 'MonthlyIncome', 'AssetValue', 'TotalDebt']
        processed_data = dataframe[features].to_numpy(dtype=np.float32)
        print("데이터 로드 및 전처리 완료.")

        print("LIME 설명자 초기화 중...")
        self.explainer = LimeTabularExplainer(
            processed_data,
            feature_names=features,
            class_names=['Success Probability'],
            discretize_continuous=True
        )
        print("LIME 설명자 초기화 완료.")

    def generate_prompt(self, lime_explanation, model_prediction):
        explanation_details = f"""
        LIME과 딥러닝 모델을 이용해서 유저의 회생가능성을 예측하고, 그 예측결과에 대한 해석을 편하게 설명해주는 기능을 개발하려고 해.\n
        딥러닝모델에 들어가는 input은 AssetValue, TotalDebt, MonthlyIncome, NumberOfDependents, MaritalStatus 5개야.\n
        AssetValue는 현재 재산의 가치, TotalDebt는 현재 가지고있는 대출, MonthlyIncome은 월급, NumberOfDependents는 부양가족 수, MaritalStatus는 결혼유무야.\n
        일단 변수간의 상관관계만 배경지식정도로 기억하라고 알려줄게. 총 빚이 보유자산 + 월금 몇달 모아서 갚아질것같으면 회생이 왠만하면 불가능할거야. 근데 그렇게 계산해도 가망없으면 아마 가능할거야.\n
        그리고 부양가족은 최저생계비랑 관련이 있는 변수고 미혼기혼도 약간 그런느낌이야. 그리고 보유자산은 낮을수록 회생가능성이 높아질거고, 빚은 많을수록 회생가능성이 높아질거야. 반대로는 당연히 낮아지고.\n
        그리고 월급이 너무 많으면 충분히 빚을 갚을수있다고 생각해서 회생에 안좋은 영향을 끼치기도해. 하지만 이것도 빚이 너무 많은경우에는 무용지물이야.
        이제부터 내가 딥러닝모델의 결과와 LIME결과를 줄게.\n
        회생결과는 {model_prediction}이 나왔어.
    """
        for feature, value in lime_explanation:
            if value > 0:
                explanation_details += f"{feature}는 {value}만큼 긍정적인 영향을 주어 회생 가능성을 높였어.\n"
            else:
                explanation_details += f"{feature}는 {value}만큼 부정적인 영향을 주어 회생 가능성을 낮췄어.\n"

        explanation_details += """
        LIME과 딥러닝모델의 결과는 이렇게 나왔어. 이걸 완전 하나하나 그대로 말하지말고, 중요한것만 골라서 일반인이 이해하기 쉽게 설명해줘. 그니까 뭐가 부족했고, 이런거 납득하기 쉽게 말해주면 됨.
        너무 길게는 말고 2~3줄정도로. 
        그리고 이거 알려줄 때, 너 입장에서 알려주지말고. 일반인에게 이 왜 이 결과가 나왔는지 그 당사자에게 설명한다는 느낌으로 말을해줘.
        
        Ex)
        "현재 자산 가치가 낮아 회생가능성에 긍정적인 영향을 주었지만, 총 대출이 상대적으로 적어 회생 가능성을 약간 낮췄습니다. 월급이 충분해 빚 상환에 일부 긍정적 작용을 했으며, 부양가족 수가 없고 미혼 상태가 부양 부담을 줄여 긍정적으로 작용했습니다. 전체적으로는 ~하면(여기에는 말이안되는 말이 들어가면 안됨. 그리고 명확한 방향성이 제시되면 좋음. 아주 명확하게) 더 나은 결과를 기대할 수 있습니다."
        예시는 예시일뿐이야. 참고만 해
        그리고 반드시 결과는 JSON으로 반환해줘.
        그리고 너무 말이 안되는것같은 표현은 배제해, 그니까 표현을 진짜 사람에게 전하는것처럼 사용하라는 뜻이야. 그리고 긍정적이었다. 부정적이었다. 이러한 표현을 반복하지마. 설명하듯이 말해. 예시는 예시일뿐이야

        JSON Structure
        {"result": your content}
    """
        

        return explanation_details

    

    def preprocess_data(self, NumberOfDependents, MaritalStatus, MonthlyIncome, AssetValue, TotalDebt):
        marital_status_map = {'미혼': 0, '기혼': 1}
        MaritalStatus_encoded = marital_status_map.get(MaritalStatus, 0)

        input_data = pd.DataFrame({
            'NumberOfDependents': [NumberOfDependents],
            'MaritalStatus': [MaritalStatus_encoded],
            'MonthlyIncome': [MonthlyIncome],
            'AssetValue': [AssetValue],
            'TotalDebt': [TotalDebt]
        })

        input_tensor = torch.tensor(input_data.values.astype(np.float32), dtype=torch.float32)
        return input_tensor

    def predict_fn(self, x):
        x = torch.tensor(x, dtype=torch.float32)
        with torch.no_grad():
            logits = self.model(x).numpy()
        probs = torch.sigmoid(torch.tensor(logits)).numpy()
        return np.hstack([1 - probs, probs])

    def predict(self, NumberOfDependents, MaritalStatus, MonthlyIncome, AssetValue, TotalDebt):
        print("========== 회생 가능성 예측 시작 ==========")
        print("입력 데이터를 준비 중...")
        processed_data = self.preprocess_data(NumberOfDependents, MaritalStatus, MonthlyIncome, AssetValue, TotalDebt)

        print("모델 추론 중...")
        with torch.no_grad():
            output = self.model(processed_data)
            probability = torch.sigmoid(output).item()
        print(f"모델 출력 확률: {probability:.6f}")

        output_percentage = int(probability * 100)

        if output_percentage < 5:
            output_percentage = random.randint(0, 5)
        if output_percentage > 95:
            output_percentage = random.randint(95, 99)
        print(f"최종 출력 확률: {output_percentage}%")

        print("LIME 설명 생성 중...")
        exp = self.explainer.explain_instance(
            processed_data.numpy().flatten(), self.predict_fn, num_features=5
        )
        explanation = exp.as_list()
        print("LIME 설명 생성 완료.")

        print("OpenAI를 이용한 결과 해석 생성 중...")
        prompt = self.generate_prompt(lime_explanation=explanation, model_prediction=output_percentage)

        completion = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.4
        )

        result = completion.choices[0].message.content
        result = result.strip('`').replace('json', '').strip()
        result = json.loads(result)
        result = result.get("result")
        print("========== 회생 가능성 예측 완료 ==========")

        return output_percentage, result


if __name__ == "__main__":
    cal = Revival_Calculator("model/Revival Cal acc:99.74 f1:99.80.pt", "data/df_cleaned5.csv")
    result, explanation = cal.predict(1, '미혼', 280, 1000, 100000)
    print(result, explanation)
