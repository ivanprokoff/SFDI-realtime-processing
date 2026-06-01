import customtkinter
from open_close_button import Button
from PIL import Image as I
from external_functions import create_patient_directory, predict_hb
import tkinter
import time
from datetime import datetime

class Side_Frame(customtkinter.CTkFrame):
    """Frame with patient id, buttons to open and close the cameras"""
    def __init__(self, parent, container):
        super().__init__(container)

        self.patient_entry = customtkinter.CTkEntry(self, justify='center')
        self.patient_entry.grid(row=0, column=0, padx=20, pady=10)
        self.patient_entry.insert(0, '0')
        self.folder_button = customtkinter.CTkButton(self,
                                                     command=lambda *args: [create_patient_directory(
                                                         self.patient_entry.get()),
                                                         parent.log_frame.insert_log('Directory'),
                                                         parent.tabview.fill_entry()],
                                                     text='Create directory')

        self.folder_button.grid(row=0, column=1, padx=5, pady=10)
        self.black_button = Button(self, text_color='white',fg_color='black')
        self.black_button.set_commands(open_text='Black', close_text='close RGB camera',
                                       open_commands=lambda *args: [
                                           parent.projection_window.set_background('black'),
                                       ],
                                       close_commands=None)
        self.black_button.grid(row=3, column=1, padx=5, pady=10)

        self.white_button = Button(self,fg_color='white', text_color='black')
        self.white_button.set_commands(open_text='White', close_text=None,
                                       open_commands=lambda *args: [
                                           parent.projection_window.set_background('white'),
                                       ],
                                       close_commands=None)
        self.white_button.grid(row=4, column=1, padx=5, pady=10)

        self.rgb_button = Button(self)
        self.rgb_button.set_commands(open_text='RGB view', close_text='close RGB camera',
                                     open_commands=lambda *args: [
                                         parent.stop(),
                                         parent.camera.open_camera(),
                                         parent.translate_rgb_cam()],
                                     close_commands=lambda *args: [  # self.rgb_button.change_function(),
                                         parent.camera.release_camera()])
        self.rgb_button.grid(row=3, column=0, padx=20, pady=10)

        self.thor_button = Button(self)
        self.thor_button.set_commands(open_text='Thor camera', close_text='close Thor camera',
                                      open_commands=lambda *args: [parent.stop(),
                                                                   parent.thor_camera.cam.start_acquisition(
                                                                       auto_start=False, nframes=1,
                                                                       frames_per_trigger=1),
                                                                   parent.translate_thor_cam()],
                                      # parent.translate_thor_cam()],
                                      close_commands=lambda *args: [self.thor_button.change_function(),
                                                                    parent.thor_camera.release_camera()])

        self.thor_button.grid(row=4, column=0, padx=20, pady=10)

        self.open_thor_button = Button(self, fg_color="transparent", border_width=1)
        self.open_thor_button.set_commands(open_text='Open Thor camera', close_text='close Thor camera',

                                           open_commands=lambda *args: [parent.stop(),
                                                                        self.open_thor_button.change_function(),
                                                                        parent.thor_camera.open_camera(),
                                                                        ],

                                           close_commands=lambda *args: [self.open_thor_button.change_function(),
                                                                         parent.thor_camera.release_camera()])

        self.open_thor_button.grid(row=5, column=0, padx=[20, 20], pady=[50, 0])

        # # Создаем переменную, которая берет начальное значение из main_app
        # self.zhang_var = customtkinter.BooleanVar(value=parent.use_zhang)
        #
        # self.zhang_switch = customtkinter.CTkSwitch(
        #     self,
        #     text="Zhang patterns",
        #     variable=self.zhang_var,  # Привязываем переменную
        #     command=parent.toggle_zhang
        # )
        # self.zhang_switch.grid(row=5, column=0, padx=20, pady=10)


class Log_Window(customtkinter.CTkFrame):
    """Window for logging events during measurement"""
    def __init__(self, parent, container):
        super().__init__(container)

        self.textbox = customtkinter.CTkTextbox(master=self, width=400, height=200, corner_radius=0, spacing1=10)
        self.textbox.grid(row=0, column=0, sticky="nsew")
        self.parent = parent
        date = str(datetime.date(datetime.now()))

        self.log_path = f'C:/Users/madpl/clinic_data/Logs/{date}/{date}_log.txt'

    def insert_log(self, command='Infrared', *args):
        """Inserts log and saves to txt depending on the event"""
        t = time.localtime()
        current_time = time.strftime("%H:%M:%S", t)

        if command == 'SFDI':
            line = f'{current_time}     {self.parent.patient_entry.get()} SFDI measured'
        elif command == 'SFDIStatus':
            message = args[0] if args else ''
            line = f'{current_time}      {self.parent.patient_entry.get()} {message}'
        elif command == 'Infrared':
            line = f'{current_time}      {args}'
        elif command == 'RGB':

            line = f'{current_time}      {args}'
        elif command == 'Directory':
            line = f'{current_time}      {self.parent.patient_entry.get()} directory created'
        elif command == 'Predict':
            line = f'{current_time}      {self.parent.patient_entry.get()} {args}'
        elif command == 'Exception':
            line = f'{current_time}      ThorCam is closed'
        else:
            line = f'{current_time}      {args}'

        self.parent.text_box.insert('0.0', line + '\n')

        with open(self.log_path, 'a') as f:
            f.write(line)
            f.write('\n')

class Translation(customtkinter.CTkFrame):
    """Class for the frame that translates the view from the cameras"""
    def __init__(self, parent, container):
        super().__init__(container)

        self.pattern_copy = customtkinter.CTkLabel(self, image=None, text='')
        self.pattern_copy.grid(row=0, column=0, rowspan=10, padx=2, pady=2)
        black_image = I.new('RGB', (500, 500))
        img = customtkinter.CTkImage(black_image, size=(500, 500))
        self.pattern_copy.configure(image=img)


class TabWindow(customtkinter.CTkTabview):
    """Class for the window with all methods in a tab frame"""
    def __init__(self, parent, container):
        super().__init__(container)
        self.parent = parent

        self.add("Infrared")
        self.exposure_entry = customtkinter.CTkEntry(self.tab("Infrared"), justify='center')
        self.exposure_entry.insert(0, '67')
        self.exposure_entry.grid(row=0, column=0, columnspan=1, padx=(20, 20), pady=10, sticky="nsew")

        self.exposure_button = customtkinter.CTkButton(master=self.tab("Infrared"), fg_color="transparent",
                                                       text_color=("gray10", "#DCE4EE"), text='Set exposure',
                                                       border_width=1,
                                                       command=lambda: [parent.thor_camera.change_exposition(
                                                           int(self.exposure_entry.get())),
                                                           self.fill_entry()])

        self.exposure_button.grid(row=0, column=1, padx=(20, 20), pady=(10, 10), sticky="nsew")

        self.infrared_name_entry = customtkinter.CTkEntry(self.tab("Infrared"), justify='center')

        self.infrared_name_entry.grid(row=1, column=0, columnspan=1, padx=(20, 20), pady=(20, 20), sticky="nsew")

        self.thor_photo_button = customtkinter.CTkButton(master=self.tab("Infrared"), fg_color="transparent",
                                                         text_color=("gray10", "#DCE4EE"), text='Take photo',
                                                         border_width=1,
                                                         command=lambda *args: [
                                                             create_patient_directory(parent.patient_entry.get()),

                                                             parent.save_thor_image(),

                                                         ])

        self.thor_photo_button.grid(row=1, column=1, padx=(20, 20), pady=(20, 20), sticky="nsew")

        self.radio_var = tkinter.IntVar(value=730)

        self.radio_button_730 = customtkinter.CTkRadioButton(self.tab('Infrared'), text="730", variable=self.radio_var,
                                                             value=730,
                                                             command=lambda *args: self.fill_entry())
        self.radio_button_730.grid(row=2, column=0)

        self.radio_button_850 = customtkinter.CTkRadioButton(self.tab('Infrared'), text='850', variable=self.radio_var,
                                                             value=850,
                                                             command=lambda *args: self.fill_entry())
        self.radio_button_850.grid(row=2, column=1)

        self.fill_entry()

        self.add("Photo")

        self.rgb_photo_button = customtkinter.CTkButton(master=self.tab("Photo"), fg_color="transparent",
                                                        text_color=("gray10", "#DCE4EE"), text='Take photo',
                                                        border_width=1,
                                                        command=lambda *args: [
                                                            create_patient_directory(parent.patient_entry.get()),
                                                            parent.save_rgb_image(),
                                                        ])
        self.rgb_photo_button.grid(row=0, column=0, columnspan=2, padx=(70, 20), pady=(40, 10), sticky="e")

        self.predict_hb_button = customtkinter.CTkButton(master=self.tab("Photo"), fg_color="transparent",
                                                         text_color=("gray10", "#DCE4EE"), text='Predict Hb',
                                                         border_width=1,
                                                         command=lambda *args: [
                                                             predict_hb(parent.log_frame),
                                                         ])
        self.predict_hb_button.grid(row=1, column=0, columnspan=2, padx=(70, 20), pady=(10, 10), sticky="w")

        self.add("SFDI")

        self.sfdi_button = customtkinter.CTkButton(master=self.tab("SFDI"), fg_color="transparent",
                                                   text_color=("gray10", "#DCE4EE"), text='SFDI',
                                                   command=lambda *arg: [
                                                       create_patient_directory(parent.patient_entry.get()),
                                                       parent.renew_current_directory(),

                                                       parent.begin_sfdi()],
                                                   border_width=1)

        self.stop_button = customtkinter.CTkButton(master=self.tab("SFDI"), fg_color="transparent",
                                                   text_color=("gray10", "#DCE4EE"), text='Stop',
                                                   command=lambda *arg: [

                                                       parent.stop()],
                                                   border_width=1)
        # self.tabview.tab("CTkTabview").grid_columnconfigure(0, weight=1)  # configure grid of individual tabs
        self.tab("SFDI").grid_columnconfigure(0, weight=1)
        self.sfdi_button.grid(row=1, column=0, padx=(20, 20), pady=(10, 10), sticky="nsew")
        self.stop_button.grid(row=2, column=0, padx=(20, 20), pady=(10, 10), sticky="nsew")

        self.sfdi_reset_button = customtkinter.CTkButton(master=self.tab("SFDI"), fg_color="transparent",
                                                         text_color=("gray10", "#DCE4EE"), text='Reset',
                                                         command=lambda *arg: [
                                                             parent.reset_sfdi_plots()],
                                                         border_width=1)
        self.sfdi_reset_button.grid(row=3, column=0, padx=(20, 20), pady=(10, 10), sticky="nsew")

        self.brightness_button = customtkinter.CTkButton(
            master=self.tab("SFDI"),
            text="Яркость паттернов",
            text_color=("gray10", "#DCE4EE"),
            command=self.parent.open_brightness_window,
            fg_color="transparent",
            border_width=1
        )
        self.brightness_button.grid(row=4, column=0, padx=(20, 20), pady=(10, 10), sticky="nsew")

        self.zhang_var = customtkinter.BooleanVar(value=parent.use_zhang)
        self.zhang_switch = customtkinter.CTkSwitch(
            master=self.tab("SFDI"),
            text="Zhang patterns",
            variable=self.zhang_var,
            command=self.parent.toggle_zhang
        )
        # zhang_switch скрыт из UI, функционал сохранён через parent.use_zhang / toggle_zhang

        self.auto_sfdi_button = customtkinter.CTkButton(
            master=self.tab("SFDI"),
            fg_color="transparent",
            text_color=("gray10", "#DCE4EE"),
            text="Auto: OFF",
            border_width=1,
            command=parent.begin_auto_sfdi,
        )
        self.auto_sfdi_button.grid(row=5, column=0, padx=(20, 20), pady=(10, 10), sticky="nsew")

    def fill_entry(self):
        """Fills the exposition entry """
        patient = self.parent.patient_entry.get()
        exposure = self.exposure_entry.get()
        wv = self.radio_var.get()
        self.infrared_name_entry.delete(0, customtkinter.END)
        line = f'{patient}_{exposure}_{wv}'
        self.infrared_name_entry.insert(0, line)
