import tkinter as tk
import tkinter.messagebox
import customtkinter
from PIL import Image, ImageTk
import cv2
import numpy as np
import time

import subprocess
import base64

import numpy as np
import cv2 as cv
from keras.models import load_model

customtkinter.set_appearance_mode("dark")
customtkinter.set_default_color_theme("blue")


class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        self.colors = {'neutralny': (255, 255, 255), 'zlosc': (0, 0, 255), 'strach': (0, 0, 0),
                       'szczescie': (0, 255, 255),
                       'smutek': (255, 0, 0), 'zaskoczenie': (255, 245, 0)}

        self.imotions = {0: 'zlosc', 1: 'strach', 2: 'szczescie', 3: 'smutek',
                         4: 'zaskoczenie', 5: 'neutralny'}

        self.model = load_model('my_model_test1.hdf5')
        self.face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

        # window
        self.cap = cv2.VideoCapture(0)
        self.title("System rozpoznawania emocji twarzy")
        self.geometry(f"{1024}x{680}")
        self.iconphoto(True, tk.PhotoImage(file="Logo_WAT.png"))
        # grid layout
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure((2, 3), weight=0)
        self.grid_rowconfigure((0, 1, 2), weight=1)

        self.sidebar_frame = customtkinter.CTkFrame(self, width=140, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(4, weight=1)

        self.logo_image = customtkinter.CTkImage(Image.open("Logo_WAT.png"), size=(116, 146))

        self.navigation_frame_label = customtkinter.CTkLabel(self.sidebar_frame, text="",
                                                             image=self.logo_image,
                                                             compound="left",
                                                             font=customtkinter.CTkFont(size=15, weight="bold"))
        self.navigation_frame_label.grid(row=0, column=0, padx=20, pady=(20, 10))
        self.save_image_var = tk.IntVar(value=0)
        self.sidebar_button_1 = customtkinter.CTkButton(self.sidebar_frame, text="Zapisz obraz")
        self.sidebar_button_1.grid(row=1, column=0, padx=(20, 20), pady=10)
        self.sidebar_button_2 = customtkinter.CTkButton(self.sidebar_frame, text="Wczytaj obraz")
        self.sidebar_button_2.grid(row=2, column=0, padx=20, pady=10)
        self.sidebar_button_3 = customtkinter.CTkButton(self.sidebar_frame, text="Konfiguracja Wifi")
        self.sidebar_button_3.grid(row=3, column=0, padx=20, pady=10)
        self.appearance_mode_label = customtkinter.CTkLabel(self.sidebar_frame, text="Tryb:", anchor="w")
        self.appearance_mode_label.grid(row=5, column=0, padx=20, pady=(10, 0), sticky='s')
        self.appearance_mode_optionemenu = customtkinter.CTkOptionMenu(self.sidebar_frame, values=["jasny", "ciemny"],
                                                                       command=self.change_appearance_mode_event)
        self.appearance_mode_optionemenu.grid(row=6, column=0, padx=20, pady=(10, 20))
        self.scaling_label = customtkinter.CTkLabel(self.sidebar_frame, text="Skala interfejsu:", anchor="w")
        self.scaling_label.grid(row=7, column=0, padx=20, pady=(10, 0))
        self.scaling_optionemenu = customtkinter.CTkOptionMenu(self.sidebar_frame,
                                                               values=["80%", "90%", "100%", "110%", "120%"],
                                                               command=self.change_scaling_event)
        self.scaling_optionemenu.grid(row=8, column=0, padx=20, pady=(10, 20))

        self.camera_frame = customtkinter.CTkFrame(self, corner_radius=10, fg_color="transparent")
        self.camera_frame.grid(row=0, column=1, rowspan=2, padx=(10, 10), pady=10)
        self.camera = customtkinter.CTkLabel(self.camera_frame, text="")
        self.camera.grid()

        # przechwyc obraz
        self.radiobutton_frame = customtkinter.CTkFrame(self)
        self.radiobutton_frame.grid(row=0, column=2, padx=(20, 20), pady=(10, 0), sticky="nsew")
        self.label_radio_group = customtkinter.CTkLabel(master=self.radiobutton_frame, text="Zamrożenie obrazu:")
        self.label_radio_group.grid(row=0, column=0, pady=(20, 0), padx=10, sticky="nw")
        # self.checkbox_capture = customtkinter.CTkCheckBox(master=self.radiobutton_frame, text="Przechwyc obraz")
        self.button_capture = customtkinter.CTkButton(self.radiobutton_frame, text="Przechwyc obraz")
        self.button_capture.grid(row=1, column=0, pady=(20, 0), padx=10, sticky="nw")
        # real_time & false_color
        self.checkbox_slider_frame = customtkinter.CTkFrame(self)
        self.checkbox_slider_frame.grid(row=1, column=2, padx=(20, 20), pady=(20, 0), sticky="nsew")
        self.label_radio_group = customtkinter.CTkLabel(master=self.checkbox_slider_frame, text="Modyfikacje obrazu:")
        self.label_radio_group.grid(row=0, column=0, pady=(20, 10), padx=10, sticky="nw")
        self.checkbox_real_time = customtkinter.CTkCheckBox(master=self.checkbox_slider_frame,
                                                            text="Obraz w czasie rzeczywistym",
                                                            onvalue=1, offvalue=0)
        self.checkbox_real_time.grid(row=1, column=0, pady=(20, 0), padx=10, sticky="nw")
        self.false_color_var = tk.IntVar(value=0)
        self.checkbox_false_color = customtkinter.CTkCheckBox(master=self.checkbox_slider_frame,
                                                              text="FalseColor")
        self.checkbox_false_color.grid(row=2, column=0, pady=(10, 20), padx=10, sticky="nw")

        # defaults
        self.appearance_mode_optionemenu.set("ciemny")
        self.scaling_optionemenu.set("100%")

    def change_appearance_mode_event(self, nm: str):
        customtkinter.set_appearance_mode("dark" if nm == "ciemny" else "light")

    def change_scaling_event(self, new_scaling: str):
        new_scaling_float = int(new_scaling.replace("%", "")) / 100
        customtkinter.set_widget_scaling(new_scaling_float)

    def convert_dtype(self, x):
        x_float = x.astype('float32')
        return x_float

    def normalize(self, x):
        x_n = (x - 0) / 255
        return x_n

    def reshape(self, x):
        x_r = x.reshape((x.shape[0], x.shape[1], x.shape[2], 1))
        return x_r

    def calc_fps(self):
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        fps = int(fps)
        fps = str(fps)
        return fps

    def detection(self, img):
        pr = None
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            roi_gray = cv.resize(roi_gray, (48, 48), interpolation=cv.INTER_AREA)
            roi_gray = self.convert_dtype(np.array([roi_gray]))
            roi_gray = self.normalize(roi_gray)
            roi_gray = self.reshape(roi_gray)
            pr = self.model.predict(roi_gray)[0]
            max_emo = np.argmax(pr)
            cv.rectangle(img, (x, y), (x + w, y + h), self.colors[self.imotions[max_emo]], 1)
            cv.rectangle(img, (x, y), (x + w, y + h + 125), self.colors[self.imotions[max_emo]], 1)
            counter = 0
            for i in range(len(pr)):
                cv.rectangle(img, (x, y + h + counter + 5), (x + int(w * pr[i]), y + h + counter + 20),
                             self.colors[self.imotions[i]],
                             -2)
                counter += 20
                cv.putText(img, str(int(pr[i] * 100)), (x + int(w * pr[i]), (y + h + counter + 5)),
                           cv.FONT_HERSHEY_SIMPLEX,
                           0.50, (0, 0, 0), 1)
                if i != 5:
                    cv.putText(img, self.imotions[i], (x, (y + h + counter)), cv.FONT_HERSHEY_SIMPLEX, 0.75,
                               (255, 255, 255),
                               1)
                else:
                    cv.putText(img, self.imotions[i], (x, (y + h + counter)), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0),
                               1)
        return pr, img

    def streaming(self):
        ret, img = self.cap.read()
        pr, img = self.detection(img)
        cv2image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(cv2image)
        ImgTks = ImageTk.PhotoImage(image=img)
        self.camera.imgtk = ImgTks
        self.camera.configure(image=ImgTks)
        self.after(1, self.streaming)

if __name__ == "__main__":
    app = App()
    app.streaming()
    app.mainloop()