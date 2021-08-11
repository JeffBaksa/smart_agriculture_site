from flask import Flask, render_template, url_for, request
import numpy as np
import pandas as pd
import pickle
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
# from io import BytesIO
# import base64
# import random
import os
from skimage import io
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

# Dataframes used to calculate similarity of other crops w/ dict
crops = pd.read_csv('../data/Crop_recommendation.csv')
crop_avgs = np.round(crops.groupby('label').mean(), 2)

app = Flask(__name__)

# Load In Model
model = tf.keras.models.load_model("saved_models")

def model_predict(img_path, model):
    img = image.load_img(img_path, grayscale=False, target_size=(224, 224))
    show_img = image.load_img(img_path, grayscale=False, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = np.array(x, 'float32')
    x /= 255
    preds = model.predict(x)
    return preds


@app.route('/')
@app.route('/home')
def home(title=None):
    title="Home"
    return render_template("home.html", title=title)

# MOTIVATION / MISSION 
@app.route('/about')
def about():
    return render_template("about.html")

# RECOMMENDER BASED ON CURRENT PLANT
@app.route('/crop_recommendation')
def crop_recommendation():
    return render_template("crop_recommendation.html")

@app.route('/rec_results', methods=['POST'])
def rec_results():
    input_crop = str(request.form['crop_recc']).lower()
    
    # Load Model 
    filename = 'similarities_d.pkl'
    d = pickle.load(open(filename, 'rb'))
    
    # Get List of Recommendations
    first_rec = None
    second_rec = None
    third_rec = None
    for key, values in d.items():
        if key == input_crop:
            first_rec = values[0]
            second_rec = values[1]
            third_rec = values[2]
    
    # Return template
    if second_rec != None:
        # Load avgs into df to display
        avgs = crop_avgs.loc[[input_crop, first_rec, second_rec, third_rec]]
        return render_template('rec_results.html', crop_avgs=crop_avgs, input_crop=input_crop, avgs=avgs, recommendation1=first_rec, recommendation2=second_rec, recommendation3=third_rec)
    else:
        return render_template('error.html')
            
# IMAGE CLASSIFIER
@app.route('/image_classifier', methods=['GET', 'POST'])
def image_classifier():
    return render_template('image_classifier.html')

@app.route('/image_results', methods=['POST'])
def image_results():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
    
        disease_class = ['Apple - Apple Scab', 'Apple - Black Rot', 'Apple - Cedar Apple Rust', 'Apple - Healthy', 'Corn (Maize) - Cercospora Leaf Spot', 'Corn (Maize) - Common Rust', 'Corn (Maize) - Northern Leaf Blight', 'Corn (Maize) - Healthy', 'Grape - Black Rot', 'Grape - Healthy']
        a = preds[0]
        # Find the largest predicted probability
        ind = np.argmax(a)
        result=disease_class[ind]
        
        # Creating Dict
        disease_text = {}
        for class_ in disease_class:
            if class_ not in disease_class:
                disease_text[class_] = ''
            disease_text[class_] = f"static/text/{class_}.html"
            
        result2 = disease_text[result] 
        # Read in text file
        HtmlFile = open(result2, 'r', encoding='utf-8') 
        source_code = HtmlFile.read()
    return render_template('image_results.html', prediction=result, text=source_code)

# CROP CLASSIFIER / RECOMMENDER
@app.route('/predict')
def predict():
    return render_template('predict.html')

@app.route('/results', methods=['POST'])
def results():
    N = int(request.form['N'])
    P = int(request.form['P'])
    K = int(request.form['K'])
    temperature = float(request.form['temperature'])
    humidity = float(request.form['humidity'])
    ph = float(request.form['ph'])
    rainfall = float(request.form['rainfall'])

    test_data = np.array([N, P, K, temperature, humidity, ph, rainfall]).reshape(1, -1)

    # Load model
    filename = 'naive_bayes_model.pkl'
    model = pickle.load(open(filename, 'rb'))

    # Predict on user input
    my_prediction = model.predict(test_data)
    
    # Read in text file
    HtmlFile = open(f"static/text/{my_prediction[0]}.html", 'r', encoding='utf-8') 
    source_code = HtmlFile.read()
    
    return render_template('results.html', prediction=my_prediction[0], image=f"static/images/{my_prediction[0]}.jpg", text=source_code)

if __name__=="__main__":
    app.run(debug=True)