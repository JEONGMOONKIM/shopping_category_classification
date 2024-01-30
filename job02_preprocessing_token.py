#job02_preprocessing_token
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from konlpy.tag import Okt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import pickle

#파일로 저장해둔 크롤링 데이터 불러오기
df = pd.read_csv('./crawling_data/shopping_title20240130.csv')

#titles에 4개의 nan값이 있다. 드롭시켜주기.
df.dropna(subset=['titles'], axis=0, inplace=True)


#category에 라벨 달아주기
X = df['titles'] #titles 열의 자료들 따로 X에 담기
Y = df['category'] #category 열의 자료들 따로 Y에 담기

label_encoder = LabelEncoder()
labeled_y = label_encoder.fit_transform(Y) #라벨 붙여주기. category 6개에 0~5까지 숫자 랜덤으로 붙음.
print(labeled_y)
label = label_encoder.classes_
print(label)
with open('./models/label_encoder.pickle', 'wb') as f:
    pickle.dump(label_encoder, f)

#one hot encoding
onehot_y = to_categorical(labeled_y)
print(onehot_y[:3])

#형태소 분리
okt = Okt()
try: #1516번째 데이터가 문제로 나온다.. 왜?
    for i in range(len(X)):
        X[i] = okt.morphs(X[i], stem=True)
        # if i % 1000:
        #     print(i)
except:
    print(i)

print(X[:5])

#불용어 제거후, 단어 다시 합치기
stopwords = pd.read_csv('./stopwords.csv', index_col=0)
try:
    for j in range(len(X)):
        words = []
        for i in range(len(X[j])):
            if len(X[j][i]) > 1:
                if X[j][i] not in list(stopwords['stopword']):
                    words.append(X[j][i])
        X[j] = ' '.join(words)
except:
    print(j) #계속 1516번쨰 놈이 문제로...

print(X[:5])

#형태소들 토큰화 해주기
token = Tokenizer()
token.fit_on_texts(X) #형태소들에 라벨 붙임. 토큰이 그정보 가지고 있다.
tokened_x = token.texts_to_sequences(X) #토큰들을 숫자들의 리스트로 만들어줌. 1~
                                        #단어들이 숫자로 변환 된다.
#print(tokened_x)
#print(token.word_index) #1~8065
wordsize = len(token.word_index) + 1
#print(wordsize) #8066

with open('./models/shopping_token.pickle', 'wb') as f:
    pickle.dump(token, f)


max = 0
for i in range(len(tokened_x)):
    if max < len(tokened_x[i]):
        max = len(tokened_x[i])
print(max)
