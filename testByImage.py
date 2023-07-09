
import cv2
import numpy as np
from keras.models import model_from_json
from PIL import Image
from matplotlib import pyplot as plt

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# load json and create model
json_file = open('fer.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)

# load weights into new model
emotion_model.load_weights("fer.h5")
print("Loaded model from disk")

def loadData():
    
    image_pil = Image.open("C:\\Users\\Hritik rastogi\\OneDrive\\Pictures\\Screenshots\\Screenshot 2023-05-03 191325.png")
    resized_image = image_pil.resize((48, 48))
    grayscale_image = resized_image.convert("L")
    image_np = np.array(grayscale_image)
    # cv2.imshow(image_np)
    plt.imshow(image_np)

    plt.show()  
    print(image_np.shape)
    X=np.array([image_np])
    
    X = np.expand_dims(X, -1)
    return X

X=loadData()


emotion =   emotion_dict[(int(np.argmax(emotion_model.predict(X))))]
print(emotion)