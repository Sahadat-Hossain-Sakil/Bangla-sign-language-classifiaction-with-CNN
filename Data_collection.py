## Making directories for saving train test data
import os
import cv2
train_val = ['train', 'test']
os.chdir('PATH') # PATH is the direction where we want to create those directory for storing data
sign = ['অ','আ', 'ই', 'উ', 'এ', 'ঐ', 'ও', 'ঔ', 'ক', 'খ', 'গ', 'ঘ', 'ঙ', 'চ', 'ছ', 'জ', 'ঝ', 'ঞ', 'ট', 'ঠ', 'ড', 'ঢ', 'ন', 'ত', 'থ', 'দ', 'ধ']

if not os.path.exists("data"):
    os.makedirs("data")
    for i in train_val:
        fold = "data/" + i
        if not os.path.exists(fold):
            os.makedirs(fold)
for i in sign:
    if not os.path.exists("data/train/" + i):
        os.makedirs("data/train/"+i)
    if not os.path.exists("data/test/" + i):
        os.makedirs("data/test/"+i)

mode = 'train'
directory = 'data/'+mode+'/'
minThreshold = 70

cap = cv2.VideoCapture(0)
interrupt = -1  

while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
### counting fiels of alphabet directories 
    count = { 
             'a': len(os.listdir(directory+"/অ")),'b': len(os.listdir(directory+"/আ")),'c': len(os.listdir(directory+"/ই")),'d': len(os.listdir(directory+"/উ")),'e': len(os.listdir(directory+"/এ")),'f': len(os.listdir(directory+"/ঐ")),
             'g': len(os.listdir(directory+"/ও")),'h': len(os.listdir(directory+"/ঔ")),'i': len(os.listdir(directory+"/ক")),'j': len(os.listdir(directory+"/খ")),'k': len(os.listdir(directory+"/গ")),'l': len(os.listdir(directory+"/ঘ")),
             'm': len(os.listdir(directory+"/ঙ")),'n': len(os.listdir(directory+"/চ")),'o': len(os.listdir(directory+"/ছ")),'p': len(os.listdir(directory+"/জ")),'q': len(os.listdir(directory+"/ঝ")),'r': len(os.listdir(directory+"/ঞ")),
             's': len(os.listdir(directory+"/ট")),'t': len(os.listdir(directory+"/ঠ")),'u': len(os.listdir(directory+"/ড")),'v': len(os.listdir(directory+"/ঢ")),'w': len(os.listdir(directory+"/ন")),'x': len(os.listdir(directory+"/ত")),
             'y': len(os.listdir(directory+"/থ")),'z': len(os.listdir(directory+"/দ")),',': len(os.listdir(directory+"/ধ"))
             }
    cv2.putText(frame, "Data collection", (10, 55), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "a-o : "+str(count['a']), (10, 70), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "b-a : "+str(count['b']), (10, 85), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "c-i : "+str(count['c']), (10, 100), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "d-u : "+str(count['d']), (10, 115), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "e-e : "+str(count['e']), (10, 130), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "f-OI : "+str(count['f']), (10, 145), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "g-O : "+str(count['g']), (10, 160), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "h-OU : "+str(count['h']), (10, 175), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "i-k : "+str(count['i']), (10, 190), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "j-kh : "+str(count['j']), (10, 205), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "k-g : "+str(count['k']), (10, 220), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "l-gh : "+str(count['l']), (10, 235), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "m-Ng : "+str(count['m']), (10, 250), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "n-c : "+str(count['n']), (10, 265), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "o-ch : "+str(count['o']), (10, 280), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "p-j : "+str(count['p']), (10, 295), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "q-jh : "+str(count['q']), (10, 310), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "r-NG : "+str(count['r']), (10, 325), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "s-T : "+str(count['s']), (10, 340), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "t-Th : "+str(count['t']), (10, 355), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "u-D : "+str(count['u']), (10, 370), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "v-Dh : "+str(count['v']), (10, 385), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "w-t : "+str(count['w']), (10, 400), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "x-th : "+str(count['x']), (10, 415), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "y-d : "+str(count['y']), (10, 430), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "z-dh : "+str(count['z']), (10, 445), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, ",-n/N : "+str(count[',']), (10, 460), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)

    x1 = int(0.5*frame.shape[1])
    y1 = 10
    x2 = frame.shape[1]-10
    y2 = int(0.5*frame.shape[1])
    cv2.rectangle(frame, (x1-1, y1-1), (x2+1, y2+1), (255,0,0) ,1)
    roi = frame[y1:y2, x1:x2]
    roi = cv2.resize(roi, (300, 300))
    cv2.imshow("Frame", frame)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    blur = cv2.GaussianBlur(gray,(5,5),2)
    
    th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
    ret, test_image = cv2.threshold(th3, minThreshold, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    test_image = cv2.resize(test_image, (300,300))
    cv2.imshow("test", th3)
    
    interrupt = cv2.waitKey(10)
    if interrupt & 0xFF == 27: # esc
        break
    if interrupt & 0xFF == ord('a'):
        os.chdir(directory + '/অ')
        cv2.imwrite(str(count['a'])+'.jpg', roi)
        os.chdir('D:\\education\\project code\\sign language-2')
    if interrupt & 0xFF == ord('b'):
        os.chdir(directory + '/আ')
        cv2.imwrite(str(count['b'])+'.jpg', roi)
        os.chdir('D:\\education\\project code\\sign language-2')
    if interrupt & 0xFF == ord('c'):
        os.chdir(directory + '/ই')
        cv2.imwrite(str(count['c'])+'.jpg', roi)
        os.chdir('D:\\education\\project code\\sign language-2')
    if interrupt & 0xFF == ord('d'):
        os.chdir(directory + '/উ')
        cv2.imwrite(str(count['d'])+'.jpg', roi)
        os.chdir('D:\\education\\project code\\sign language-2')
    if interrupt & 0xFF == ord('e'):
        os.chdir(directory + '/এ')
        cv2.imwrite(str(count['e'])+'.jpg', roi)
        os.chdir('D:\\education\\project code\\sign language-2')
    if interrupt & 0xFF == ord('f'):
        os.chdir(directory + '/ঐ')
        cv2.imwrite(str(count['f'])+'.jpg', roi)
        os.chdir('D:\\education\\project code\\sign language-2')
    if interrupt & 0xFF == ord('g'):
        os.chdir(directory + '/ও')
        cv2.imwrite(str(count['g'])+'.jpg', roi)
        os.chdir('D:\\education\\project code\\sign language-2')
    if interrupt & 0xFF == ord('h'):
        os.chdir(directory + '/ঔ')
        cv2.imwrite(str(count['h'])+'.jpg', roi)
        os.chdir('D:\\education\\project code\\sign language-2')
    if interrupt & 0xFF == ord('i'):
        os.chdir(directory + '/ক')
        cv2.imwrite(str(count['i'])+'.jpg', roi)
        os.chdir('D:\\education\\project code\\sign language-2')
    if interrupt & 0xFF == ord('j'):
        os.chdir(directory + '/খ')
        cv2.imwrite(str(count['j'])+'.jpg', roi)
        os.chdir('D:\\education\\project code\\sign language-2')
    if interrupt & 0xFF == ord('k'):
        os.chdir(directory + '/গ')
        cv2.imwrite(str(count['k'])+'.jpg', roi)
        os.chdir('D:\\education\\project code\\sign language-2')
    if interrupt & 0xFF == ord('l'):
        os.chdir(directory + '/ঘ')
        cv2.imwrite(str(count['l'])+'.jpg', roi)
        os.chdir('D:\\education\\project code\\sign language-2')
    if interrupt & 0xFF == ord('m'):
        os.chdir(directory + '/ঙ')
        cv2.imwrite(str(count['m'])+'.jpg', roi)
        os.chdir('D:\\education\\project code\\sign language-2')
    if interrupt & 0xFF == ord('n'):
        os.chdir(directory + '/চ')
        cv2.imwrite(str(count['n'])+'.jpg', roi)
        os.chdir('D:\\education\\project code\\sign language-2')
    if interrupt & 0xFF == ord('o'):
        os.chdir(directory + '/ছ')
        cv2.imwrite(str(count['o'])+'.jpg', roi)
        os.chdir('D:\\education\\project code\\sign language-2')
    if interrupt & 0xFF == ord('p'):
        os.chdir(directory + '/জ')
        cv2.imwrite(str(count['p'])+'.jpg', roi)
        os.chdir('D:\\education\\project code\\sign language-2')
    if interrupt & 0xFF == ord('q'):
        os.chdir(directory + '/ঝ')
        cv2.imwrite(str(count['q'])+'.jpg', roi)
        os.chdir('D:\\education\\project code\\sign language-2')
    if interrupt & 0xFF == ord('r'):
        os.chdir(directory + '/ঞ')
        cv2.imwrite(str(count['r'])+'.jpg', roi)
        os.chdir('D:\\education\\project code\\sign language-2')
    if interrupt & 0xFF == ord('s'):
        os.chdir(directory + '/ট')
        cv2.imwrite(str(count['s'])+'.jpg', roi)
        os.chdir('D:\\education\\project code\\sign language-2')
    if interrupt & 0xFF == ord('t'):
        os.chdir(directory + '/ঠ')
        cv2.imwrite(str(count['t'])+'.jpg', roi)
        os.chdir('D:\\education\\project code\\sign language-2')
    if interrupt & 0xFF == ord('u'):
        os.chdir(directory + '/ড')
        cv2.imwrite(str(count['u'])+'.jpg', roi)
        os.chdir('D:\\education\\project code\\sign language-2')
    if interrupt & 0xFF == ord('v'):
        os.chdir(directory + '/ঢ')
        cv2.imwrite(str(count['v'])+'.jpg', roi)
        os.chdir('D:\\education\\project code\\sign language-2')
    if interrupt & 0xFF == ord('w'):
        os.chdir(directory + '/ন')
        cv2.imwrite(str(count['w'])+'.jpg', roi)
        os.chdir('D:\\education\\project code\\sign language-2')
    if interrupt & 0xFF == ord('x'):
        os.chdir(directory + '/ত')
        cv2.imwrite(str(count['x'])+'.jpg', roi)
        os.chdir('D:\\education\\project code\\sign language-2')
    if interrupt & 0xFF == ord('y'):
        os.chdir(directory + '/থ')
        cv2.imwrite(str(count['y'])+'.jpg', roi)
        os.chdir('D:\\education\\project code\\sign language-2')
    if interrupt & 0xFF == ord('z'):
        os.chdir(directory + '/দ')
        cv2.imwrite(str(count['z'])+'.jpg', roi)
        os.chdir('D:\\education\\project code\\sign language-2')  
    if interrupt & 0xFF == ord(','):
        os.chdir(directory + '/ধ')
        cv2.imwrite(str(count[','])+'.jpg', roi)
        os.chdir('D:\\education\\project code\\sign language-2')    
cap.release()
cv2.destroyAllWindows()
