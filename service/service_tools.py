# keras preprocess
from tensorflow.keras.preprocessing import sequence
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from sklearn.preprocessing import StandardScaler
import re


class preprocessor:
    
    def __init__(self, driver, batch_size, max_length):
        self.cleaning_re = "@\S+|https?:\S+|http?:\S|[^\u4E00-\u9FD5a-zA-Z]" #text_cleaning_re
        self.driver = driver
        self.batch_size=batch_size
        self.max_length=max_length

    def clean_and_tokenize(self, texts):
        cleaned = []

        for text in texts:
            text = re.sub(self.cleaning_re, '', text)
            cleaned.append(text)

        ws  = self.driver(cleaned, use_delim=False, 
                        batch_size = self.batch_size,
                        max_length = self.max_length)
        res = [" ".join(ls) for ls in ws]
        return res

class kerasPreprocessor:

    def __init__(self, tokenizer=None, max_len=None):
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def preprocess(self, texts):
        text_ints = self.tokenizer.texts_to_sequences(texts)
        text_ints_pad = sequence.pad_sequences(text_ints,
                            maxlen=self.max_len,
                            truncating='pre',
                            padding='pre')
        text_array = np.array(text_ints_pad).astype('int32')

        return text_array

class InfoDataScaler:

    def __init__(self):
        pass

    def scale(self, info_data):
        scaler = StandardScaler()
        scaled_info = scaler.fit_transform(info_data)

        return scaled_info

class Predictor:

    def __init__(self, model=None):
        self.model = model

    def run_prediction(self, text_array, scaled_info):
        proba = self.model.predict({'text':text_array,'info':scaled_info})
        proba = proba.reshape(len(proba),).tolist()

        return proba