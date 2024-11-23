import os
from google.cloud import translate_v2 as translate
import pandas as pd

# Google Cloud JSON 인증 파일 경로 설정
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/dhewl/Documents/google_api_key/gc_key.json"

client = translate.Client()

# 번역 함수
def translate_text(text, target_language="en"):
    if isinstance(text, str):
        result = client.translate(text, target_language=target_language)
        return result["translatedText"]
    else:
        return text

def back_translation(text, source_language="ko", intermediate_language="en"):
    translated_text = translate_text(text, target_language=intermediate_language)
    print(f"중간 번역된 문장: {translated_text}")

    back_translated_text = translate_text(translated_text, target_language=source_language)
    return back_translated_text

file_path = '/Users/dhewl/Desktop/han-star/data/new_data.csv'
data = pd.read_csv(file_path)

backtranslated_stories = []

for index, row in data.iterrows():
    story = row['story']  
    if pd.notnull(story):  
        backtranslated_story = back_translation(story)
        backtranslated_stories.append(backtranslated_story)
    else:
        backtranslated_stories.append(story)

data['backtranslated_story'] = backtranslated_stories

output_path = '/Users/dhewl/Desktop/han-star/data/data_with_backtranslation.csv'
data.to_csv(output_path, index=False)

print(f"Backtranslated data saved to {output_path}")