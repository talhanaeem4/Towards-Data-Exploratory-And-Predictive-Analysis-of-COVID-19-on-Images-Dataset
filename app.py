from flask import Flask, render_template,url_for, request
import cv2
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)
covid_model = load_model('C:/Users/oa806/OneDrive/Desktop/Model/covid19_model.h5')
@app.route('/',methods=['GET'])
def Load_page():
    return render_template('index.html')

@app.route('/', methods=["POST"])
def predict():
    imagefile = request.files['imagefile']
    if imagefile:
        image_path = "./images/" + imagefile.filename
        imagefile.save(image_path)
        def prepare(filepath):
            X = []
            IMG_SIZE = 224
            image = cv2.resize(cv2.imread(filepath), (IMG_SIZE, IMG_SIZE))
            image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            X.append(np.array(image))
            return np.array(X)

        pred = covid_model.predict(prepare(image_path))
        if pred > 0.5:
             covid = 'COVID postive'
        else:
             covid = 'Non-COVID'
        return render_template('index.html', prediction = pred, covid = covid)
    else:
        return render_template('index.html')
        
if __name__ == '__main__':
    app.run(port=3000, debug=True)