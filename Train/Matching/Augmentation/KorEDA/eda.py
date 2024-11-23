import random
import pickle
import re
import pandas as pd

wordnet = {}
with open("Augmentation/KorEDA/wordnet.pickle", "rb") as f:
	wordnet = pickle.load(f)


# 한글만 남기고 나머지는 삭제
def get_only_hangul(line):
	parseText= re.compile('/ ^[ㄱ-ㅎㅏ-ㅣ가-힣]*$/').sub('',line)

	return parseText


########################################################################
# Random deletion
# Randomly delete words from the sentence with probability p
########################################################################
def random_deletion(words, p):
	if len(words) == 1:
		return words

	new_words = []
	for word in words:
		r = random.uniform(0, 1)
		if r > p:
			new_words.append(word)

	if len(new_words) == 0:
		rand_int = random.randint(0, len(words)-1)
		return [words[rand_int]]

	return new_words

########################################################################
# Random swap
# Randomly swap two words in the sentence n times
########################################################################
def random_swap(words, n):
	new_words = words.copy()
	for _ in range(n):
		new_words = swap_word(new_words)

	return new_words

def swap_word(new_words):
	random_idx_1 = random.randint(0, len(new_words)-1)
	random_idx_2 = random_idx_1
	counter = 0

	while random_idx_2 == random_idx_1:
		random_idx_2 = random.randint(0, len(new_words)-1)
		counter += 1
		if counter > 3:
			return new_words

	new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1]
	return new_words



def EDA(sentence, alpha_rs=0.2, p_rd=0.2, num_aug=1):
	sentence = get_only_hangul(sentence)
	words = sentence.split(' ')
	words = [word for word in words if word is not ""]
	num_words = len(words)

	augmented_sentences = []
	num_new_per_technique = int(num_aug/4) + 1

	n_rs = max(1, int(alpha_rs*num_words))

	# rs
	for _ in range(num_new_per_technique):
		a_words = random_swap(words, n_rs)
		augmented_sentences.append(" ".join(a_words))

	# rd
	for _ in range(num_new_per_technique):
		a_words = random_deletion(words, p_rd)
		augmented_sentences.append(" ".join(a_words))

	augmented_sentences = [get_only_hangul(sentence) for sentence in augmented_sentences]
	random.shuffle(augmented_sentences)

	if num_aug >= 1:
		augmented_sentences = augmented_sentences[:num_aug]
	else:
		keep_prob = num_aug / len(augmented_sentences)
		augmented_sentences = [s for s in augmented_sentences if random.uniform(0, 1) < keep_prob]

	return list(set(augmented_sentences))

file_path = 'data/new_detailed_stories.csv'
data = pd.read_csv(file_path)

# Initialize an empty list to store backtranslated stories
backtranslated_stories = []

# Iterate over each row in the data, process the story and append the result
for index, row in data.iterrows():
    story = row['story']  # Get the story text
    if pd.notnull(story):  # Ensure the story is not NaN
        backtranslated_story = EDA(story)[0]
        backtranslated_stories.append(backtranslated_story)
    else:
        backtranslated_stories.append(story)  # Append the original NaN if the story is missing

# Add the backtranslated stories to the DataFrame
data['story'] = backtranslated_stories

# Save the result to a new CSV file
output_path = 'data/random_del.csv'
data.to_csv(output_path, index=False)

print(f"Backtranslated data saved to {output_path}")