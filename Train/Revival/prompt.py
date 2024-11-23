
from openai import OpenAI
client = OpenAI(api_key="")

def generate_prompt(lime_explanation, model_prediction):
    # LIME 해석 결과를 요약하여 GPT에게 전달할 프롬프트 작성
    explanation_details = f"""
    LIME과 딥러닝 모델을 이용해서 유저의 회생가능성을 예측하고, 그 예측결과에 대한 해석을 편하게 설명해주는 기능을 개발하려고 해.\n
    딥러닝모델에 들어가는 input은 AssetValue, TotalDebt, MonthlyIncome, NumberOfDependents, MaritalStatus 5개야.\n
    AssetValue는 현재 재산의 가치, TotalDebt는 현재 가지고있는 대출, MonthlyIncome은 월급, NumberOfDependents는 부양가족 수, MaritalStatus는 결혼유무야.\n
    일단 변수간의 상관관계만 배경지식정도로 기억하라고 알려줄게. 총 빚이 보유자산 + 월금 몇달 모아서 갚아질것같으면 회생이 왠만하면 불가능할거야. 근데 그렇게 계산해도 가망없으면 아마 가능할거야.\n
    그리고 부양가족은 최저생계비랑 관련이 있는 변수고 미혼기혼도 약간 그런느낌이야. 그리고 보유자산은 낮을수록 회생가능성이 높아질거고, 빚은 많을수록 회생가능성이 높아질거야. 반대로는 당연히 낮아지고.\n
    그리고 월급이 너무 많으면 충분히 빚을 갚을수있다고 생각해서 회생에 안좋은 영향을 끼치기도해. 하지만 이것도 빚이 너무 많은경우에는 무용지물이지.
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

    JSON Structure
    {"result": your content}
"""
    

    return explanation_details


if __name__ == "__main__":
    # 예시 LIME 해석 결과와 모델 예측값
    lime_example = [
        ("AssetValue <= 1500", 0.5668),
        ("TotalDebt <= 9000", -0.3673),
        ("MonthlyIncome > 2000", 0.124),
        ("NumberOfDependents <= 1", 0.0112),
        ("MaritalStatus == '미혼'", 0.025)
    ]
    model_prediction = 82

    prompt = generate_prompt(lime_example, model_prediction)

    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": prompt }
        ],
        temperature=0.4
    )

    print(completion.choices[0].message.content)