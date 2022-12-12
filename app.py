import streamlit as st
import tensorflow as tf
import streamlit as st


@st.cache(allow_output_mutation=True)
def load_model():
  model=tf.keras.models.load_model('hand_gesture_recognition.h5')
  return model
with st.spinner('Model is being loaded..'):
  model=load_model()

st.write("""
         #Hand Sign Classification
         """
         )

file = st.file_uploader("Please upload an brain scan file", type=["jpg", "png"])
import cv2
from PIL import Image, ImageOps
import numpy as np
st.set_option('deprecation.showfileUploaderEncoding', False)


def import_and_predict(imagem, model):
    
        # size = (100,120)    
        # image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        # image = np.asarray(image)
        # img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # #img_resize = (cv2.resize(img, dsize=(75, 75),    interpolation=cv2.INTER_CUBIC))/255.
        
        # img_reshape = img[np.newaxis,...]
        
        in_img=[]
        image=imagem
        # image=np.array(image)
        original=image
        image = cv2.resize(image,(100, 120))
        in_img.append(image)
        img = np.asarray(in_img)
        img=img.reshape(img.shape[0], 100, 120, 1)
        pred=np.argmax(model.predict(img))
        return pred
if file is None:
    st.text("Please upload an image file")
else:
    img = np.array(Image.open(file))
    imgcp=img
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (7, 7), 0)
    ret, bw_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    # converting to its binary form
    bw = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    img = cv2.bitwise_not(bw_img)
    
    st.write("Input Image...")
    st.image(imgcp, width=500)
    st.write("After Preprocessing...")
    st.image(img, width=500)
    st.write("Prediction...")
    


    pred = import_and_predict(img, model)
    # score = tf.nn.softmax(predictions[0])
    if pred==0:
      st.write("Input image represents :Blank")
    elif(pred==1):
      st.write("Input image represents: OK")
    elif pred==2:
      st.write("Input image represents :Thumbs up")
    elif(pred==4):
      st.write("Input image represents :Fist")
    elif(pred==5):
      st.write("Input image represents: Five")
    else :
      st.write("Input image represents: Thumbs Down")



