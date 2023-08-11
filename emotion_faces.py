import numpy as np
import cv2 as cv
from keras.models import load_model


class EmotionFaces:

    def __init__(self):
        self.colors = {'neutralny': (255, 255, 255), 'zlosc': (0, 0, 255), 'strach': (0, 0, 0),
                   'szczescie': (0, 255, 255),
                   'smutek': (255, 0, 0), 'zaskoczenie': (255, 245, 0)}

        self.imotions = {0: 'zlosc', 1: 'strach', 2: 'szczescie', 3: 'smutek',
                     4: 'zaskoczenie', 5: 'neutralny'}

        self.model = load_model('my_model_test1.hdf5')
        self.face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
        # self.cam = cv.VideoCapture(0)

    def convert_dtype(self, x):
        x_float = x.astype('float32')
        return x_float

    def normalize(self, x):
        x_n = (x - 0) / 255
        return x_n

    def reshape(self, x):
        x_r = x.reshape((x.shape[0], x.shape[1], x.shape[2], 1))
        return x_r

    def detection(self, img):
        pr = None
        img = cam.read()[1]
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = img[y:y + h, x:x + w]
            roi_gray = cv.resize(roi_gray, (48, 48), interpolation=cv.INTER_AREA)
            roi_gray = self.convert_dtype(np.array([roi_gray]))
            roi_gray = self.normalize(roi_gray)
            roi_gray = self.reshape(roi_gray)
            pr = self.model.predict(roi_gray)[0]

        return pr


app = EmotionFaces()
cam = cv.VideoCapture(0)
while True:
    keypress = cv.waitKey(1)
    pr, img = app.detection(cam)
    cv.imshow('img', img)
    print(pr)
    if keypress == ord('q'):
        cv.destroyAllWindows()
        break
