import base64
from io import BytesIO
from boto3.session import Session
import numpy as np
from PIL import Image
from flask import Flask, jsonify, request
import tensorflow as tf
from tensorflow.keras.models import load_model
from dotenv import load_dotenv
import os
load_dotenv()


access_key_id = os.environ.get('AWS_ACCESS_KEY_ID')
secret_access_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
app = Flask(__name__)


session = Session(aws_access_key_id=access_key_id,
                  aws_secret_access_key=secret_access_key)

s3 = session.client('s3')

url = "https://skincancermodel.s3.amazonaws.com/model.h5"


params = {'Bucket': 'skincancermodel', 'Key': 'model.h5'}
pre_signed_url = s3.generate_presigned_url('get_object', Params=params)

model_buffer = tf.keras.utils.get_file("model.h5", pre_signed_url)
model = tf.keras.models.load_model(model_buffer)
lesion_type_dict = {
    'nv': 'Melanocytic nevi',
    'mel': 'Melanoma',
    'bkl': 'Benign keratosis-like lesions ',
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratoses',
    'vasc': 'Vascular lesions',
    'df': 'Dermatofibroma'
}

lesion_ID_dict = {
    'nv': 0,
    'mel': 1,
    'bkl': 2,
    'bcc': 3,
    'akiec': 4,
    'vasc': 5,
    'df': 6
}

@app.route("/")
def index():
    return jsonify({"message":"hello"})

@app.route('/api/image', methods=['POST'])
def process_image():
    encoded_image = request.form.get('image')
    image_bytes = base64.b64decode(encoded_image)
    pil_image = Image.open(BytesIO(image_bytes))
    
    # Resize image to 100x100
    pil_image = pil_image.resize((100, 100))
    
    # Convert image to a 3D numpy array (100, 100, 3)
    img_array = np.array(pil_image)
   
    

    predictions = model.predict(np.expand_dims(img_array, axis=0))[0]
    predicted_class = np.argmax(predictions)
    keys = [k for k, v in lesion_ID_dict.items() if v == int(predicted_class)]
    response = lesion_type_dict[keys[0]]  
    
    
    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)


