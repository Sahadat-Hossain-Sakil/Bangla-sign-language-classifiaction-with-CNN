import numpy as np
from keras.models import model_from_json
import operator
import cv2

def most_frequent(nums):
      return max(set(nums), key = nums.count) 

# Loading the model
json_file = open("model-bw2.json", "r")
model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(model_json)

# load weights
loaded_model.load_weights("model-bw2.h5")
print("Loaded model from disk")

cap = cv2.VideoCapture(0)

# Category dictionary
categories = {0: 'o',1: 'a',2: 'i',3: 'u',4: 'e',5: 'OI',6: 'O',7: 'OU',8: 'k',9: 'kh',
              10: 'g',11: 'gh',12: 'Ng',13: 'c',14: 'ch',15: 'j',16: 'jh',17: 'NG',18: 'T',
              19: 'Th',20: 'D',21: 'Dh',22: 't',23: 'th',24: 'd',25: 'dh',26: 'n/N'}

signs = { 0:'অ',1:'আ',2: 'ই',3: 'উ',4: 'এ',5: 'ঐ',6: 'ও',7: 'ঔ',8: 'ক',9: 'খ',
        10: 'গ',11: 'ঘ',12: 'ঙ',13: 'চ',14: 'ছ',15: 'জ',16: 'ঝ',17: 'ঞ',18: 'ট',
        19: 'ঠ',20: 'ড',21: 'ঢ',22: 'ত',23: 'থ',24: 'দ',25: 'ধ',26: 'ন' }
best_predict = []
text = []
m = ''
count = 0
minValue = 70
while True:
    _, frame = cap.read()
    # Simulating mirror image
    frame = cv2.flip(frame, 1)
    
    # Got this from collect-data.py
    # Coordinates of the ROI
    x1 = int(0.5*frame.shape[1])
    y1 = 10
    x2 = frame.shape[1]-10
    y2 = int(0.5*frame.shape[1])
    # Drawing the ROI
    # The increment/decrement by 1 is to compensate for the bounding box
    cv2.rectangle(frame, (x1-1, y1-1), (x2+1, y2+1), (255,0,0) ,1)
    # Extracting the ROI
    roi = frame[y1:y2, x1:x2]
    roi = cv2.resize(roi, (300, 300))
    
    cv2.imshow("Frame", frame)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    blur = cv2.GaussianBlur(gray,(5,5),2)
    
    th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
    ret, test_image = cv2.threshold(th3, minValue, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    #time.sleep(5)
    #cv2.imwrite("/home/rc/Downloads/soe/im1.jpg", roi)
    #test_image = func("/home/rc/Downloads/soe/im1.jpg")


    
    cv2.imshow("test", blur)
    result = loaded_model.predict(test_image.reshape(1, 300, 300, 1))
    result = np.argmax(result)
    word = categories.get(result)
    B_word = signs.get(result)
    best_predict.append(B_word)
    count += 1
    

    if (count==20):
        text_final = most_frequent(best_predict)
        best_predict = []
        flag = 1
        if (flag ==1):
            count = 0
            text.append(text_final)
            m = ''.join(text)
            flag = 0
 
    # Displaying the predictions
    cv2.putText(frame, word, (450, 350), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,255), 2)    
    cv2.imshow("Frame", frame)
    print(B_word)
    print("\t \t \t", m)
    interrupt = cv2.waitKey(10)
    if interrupt & 0xFF == 27: # esc key
        break
    if interrupt & 0xFF == 32: # space key
        text.append(' ')
    if interrupt & 0xFF == 8: # backspace
        del text[-1]
        
cap.release()
cv2.destroyAllWindows()