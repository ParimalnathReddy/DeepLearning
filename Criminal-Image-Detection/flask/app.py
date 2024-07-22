from flask import Flask, render_template, request
from keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
import cv2
app = Flask(__name__)


dic = {0 : 'chain_snatcher', 1 : 'Drug_dealer', 2 : 'Kidnaper', 3 : 'Muderer'}


#Image Size
img_size=256
model = load_model('model.h5')

model.make_predict_function()

def predict_label(img_path):
    img=cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
    resized=cv2.resize(gray,(img_size,img_size)) 
    i = img_to_array(resized)/255.0
    i = i.reshape(1,img_size,img_size,1)
    predict_x=model.predict(i) 
    p=np.argmax(predict_x,axis=1)
    return dic[p[0]]

@app.route("/", methods=['GET', 'POST'])
def main():
    return render_template("index.html")


@app.route("/predict", methods = ['GET', 'POST'])
def upload():
    
    if request.method == 'POST':
       img = request.files['file']
       img_path = "uploads/" + img.filename    
       img.save(img_path)
       p = predict_label(img_path)
       print(p)
       return str(p).lower()

if __name__ =='__main__':
    app.run(debug = True)