from flask import Flask, request, render_template, jsonify
from service.service_tools import preprocessor, kerasPreprocessor, InfoDataScaler, Predictor
from ckip_transformers.nlp import CkipWordSegmenter
import joblib
import tensorflow.keras as keras
import pandas as pd

app = Flask(__name__)


#initialize ckip driver
print("Initializing ckip driver...")
ws_driver = CkipWordSegmenter(device=0, level=1)
print("done")
tokenizer = joblib.load('keras_tokenizer_ad_v3.pkl')
model = keras.models.load_model("lstm_model_ad_v3")
#initialize Preprocessor class
print('Initializing preprocessor...')
preprocessor = preprocessor(ws_driver,1024,200)
print("done")
keras_preprocessor = kerasPreprocessor(tokenizer, 200)
info_scaler = InfoDataScaler()

#initialize Predictor class
predictor = Predictor(model)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # receive ajax payload
    payload = request.get_json()
    corpus  = pd.DataFrame(payload)
    corpus['text_length'] = [len(text) for text in corpus['text']]
    #process data
    #doc_vec = preprocessor.get_doc_vec(text)
    #sentiment = predictor.predict_sentiment(mlmodel, doc_vec)
    segmented = preprocessor.clean_and_tokenize(corpus["text"])
    text_array = keras_preprocessor.preprocess(segmented)
    scaled_info = info_scaler.scale(corpus.iloc[:,1:])
    probas = predictor.run_prediction(text_array, scaled_info)

    spam_rate = round(probas[0],2)*100

    if spam_rate > 50:
        res = "這篇po文廣告成分有%s %%，代表很有可能是廣告"%spam_rate
    else:
        res = "這篇po文廣告成分有%s %%，代表不太有可能是廣告"%spam_rate
    
    #return data to javascript
    return jsonify(res)

# @app.route('/predict', methods=['GET'])
# def predict():
#     if 'text' in request.args:
#        ......

if __name__ == '__main__':
    # run server
    app.run(host = "140.112.147.112", port = 3000, debug=True)