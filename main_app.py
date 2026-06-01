import cv2
import customtkinter

import numpy as np
import os.path
import sys
import external_functions
from PIL import ImageEnhance, ImageDraw
from PIL import Image as I
from projection import Projection, read_patterns_paths
from rgb_cam import Camera
from thorcam import Thorcam
from frames import Side_Frame, Translation, TabWindow, Log_Window
import csv
from datetime import datetime
import os
import time
from realtime_sfdi import prewarm_sfdi_model, roi_rect_on_preview, run_realtime_sfdi_cycle

try:
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
except ImportError:
    FigureCanvasTkAgg = None
    Figure = None

customtkinter.set_appearance_mode("Dark")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("dark-blue")  # Themes: "blue" (standard), "green", "dark-blue"

PATTERN_COORDS_TO_SAVE = (100,900,0,800) #bot,top,left,right or "upper","lower", left, right in PIL notation

def save_sfdi_image(thor_img, file_name, save_only_pattern_part=False,
                    pattern_coords=None):
    """
    Save image from camera. If save_only_pattern_part is True,
    save only part that has "projection" on it using coordinates passed through pattern_coords tuple

    :param thor_img: (PIL.Image) Image from Thorlabs Camer
    :param file_name: (str) Path to file
    :param save_only_pattern_part: (bool) - default False
    :param pattern_coords: tuple(int) - bot,top,left,right
    :return: None
    """

    if save_only_pattern_part:
        assert not pattern_coords is None, "Pattern coordinates should be specified! is save_only_pattern is True"
        # row-column notation. Upper-lower - first coord,
        # upper should be less than lower
        # left-right - second coord. left should be less than right

        upper,lower,left,right = pattern_coords
        sub_image = thor_img.crop((left, upper, right, lower))
        sub_image.save(file_name)
    else:
        thor_img.save(file_name)

class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        # configure window
        self.animation = True
        self.bind('<Escape>', lambda *args: [sys.exit(1)])
        self.flag = True
        self.first_frame = True
        self.current_directory = None
        self.title("Clinical app")

        # Added: use_zhang
        #Changed: patterns read
        self.use_zhang = False
        self.patterns = read_patterns_paths(use_zhang=self.use_zhang)

        self.iter_patterns = iter(self.patterns)
        self.preloaded_patterns = {}
        for key, (path, factor) in self.patterns.items():
            with I.open(path) as im:
                enhancer = ImageEnhance.Brightness(im)
                enhanced_img = enhancer.enhance(factor)
                ctk_img = customtkinter.CTkImage(enhanced_img, size=(enhanced_img.width, enhanced_img.height))
                self.preloaded_patterns[key] = ctk_img

        self.exposure = 66.68 / 1000  # МЕНЯЛИ ДЛЯ УСТРАННЕНИЯ РАССИНХРОНА
        self.pattern_factors = [0.42, 0.54, 0.62]
        self.geometry('%dx%d+%d+%d' % (1520, 900, 0, 0))

        self.auto_capture = False
        self.camera = Camera(self)
        self.thor_camera = Thorcam(self)
        self.after(1000, lambda: prewarm_sfdi_model(self.patterns))

        """
        SIDEBAR FRAME
        """
        container = customtkinter.CTkFrame(self, width=80, height=1, corner_radius=0, border_width=1,
                                           border_color='RED')
        container.grid(row=0, column=0, rowspan=1, columnspan=1, pady=[50, 10], padx=20, sticky='NW')

        self.sidebar_frame = Side_Frame(self, container)
        self.patient_entry = self.sidebar_frame.patient_entry
        self.sidebar_frame.grid(row=0, column=0)

        """
          Log_frame
        """
        container = customtkinter.CTkFrame(self, width=40, height=1, corner_radius=0)
        container.grid(row=1, column=2, rowspan=1, columnspan=1, pady=[10, 10], padx=20, sticky='nswe')

        self.log_frame = Log_Window(self, container)
        self.text_box = self.log_frame.textbox
        self.log_frame.grid(row=0, column=0)

        """
          Hemoglobin display
        """
        hemo_container = customtkinter.CTkFrame(self, width=40, corner_radius=0)
        hemo_container.grid(row=2, column=2, columnspan=1, pady=[0, 10], padx=20, sticky='nswe')

        customtkinter.CTkLabel(
            hemo_container,
            text="HEMOGLOBIN",
            text_color="red",
            font=customtkinter.CTkFont(size=18, weight="bold"),
        ).grid(row=0, column=0, padx=10, pady=(10, 2))

        self.hemo_value_label = customtkinter.CTkLabel(
            hemo_container,
            text="—",
            text_color="red",
            font=customtkinter.CTkFont(size=28, weight="bold"),
        )
        self.hemo_value_label.grid(row=1, column=0, padx=10, pady=(2, 10))

        self.sfdi_plot_history = {
            'green': {'mua': [], 'mus': []},
            'red': {'mua': [], 'mus': []},
        }
        self._setup_sfdi_plots()

        """
        Frame window for camera translation
        """
        container_center = customtkinter.CTkFrame(self, width=90, height=100, border_color='black', border_width=1)
        container_center.grid(row=0, column=4, rowspan=5, columnspan=1, pady=50, padx=20, sticky="NW")

        self.translation_frame = Translation(self, container_center)
        self.translation_frame.grid(row=0, column=0, rowspan=10)
        self.pattern_copy = self.translation_frame.pattern_copy

        """
        TAB frame with all methods
        """
        self.tab_frame = customtkinter.CTkFrame(self, width=30, height=1, corner_radius=0, border_width=1,
                                                border_color='black')
        self.tab_frame.grid(row=0, column=2, padx=10, pady=(40, 0), columnspan=1, sticky="NW")

        container = customtkinter.CTkTabview(self.tab_frame, width=40, height=10)
        container.grid(row=0, column=0, padx=20, pady=(0, 0), sticky="NseW")

        self.tabview = TabWindow(self, container)
        self.tabview.grid(row=0, column=0, columnspan=2)

        self.projection_window = Projection(self)

        # self.projection_window.set_first_picture()
        self.infrared_name_entry = self.tabview.infrared_name_entry

    def toggle_zhang(self):
        # Берем актуальное значение прямо со свитча
        self.use_zhang = self.tabview.zhang_var.get()

        # Правильный вызов логгера (передаем 'Infrared' как категорию, или оставь как есть,
        # так как мы добавили блок else в frames.py)
        self.log_frame.insert_log('Infrared', f"use_zhang changed to {self.use_zhang}")

        # перечитать паттерны
        self.patterns = read_patterns_paths(use_zhang=self.use_zhang)

        self.iter_patterns = iter(self.patterns)
        self.preloaded_patterns = {}

        for key, (path, factor) in self.patterns.items():
            with I.open(path) as im:
                enhancer = ImageEnhance.Brightness(im)
                enhanced_img = enhancer.enhance(factor)

                ctk_img = customtkinter.CTkImage(
                    enhanced_img,
                    size=(enhanced_img.width, enhanced_img.height)
                )

                self.preloaded_patterns[key] = ctk_img

    def translate_rgb_cam(self):
        """Translates view from rgb cam"""
        self.flag = True
        while self.animation and self.flag and self.camera.cap:

            frame = self.camera.get_frame()
            if frame is not None:
                frame = cv2.flip(frame, 1)
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                frame = cv2.flip(frame, 0)
                frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

                frame = frame[:, :, ::-1]

                pil_img = I.fromarray(frame)
                img = customtkinter.CTkImage(pil_img, size=(np.shape(pil_img)[1], np.shape(pil_img)[0]))

                draw = ImageDraw.Draw(pil_img)

                draw.rectangle(((133, 320), (177, 350)), fill=None, outline=255)

                draw.rectangle(((233, 238), (267, 272)), fill=None, outline=255)
                draw.rectangle(((313, 285), (347, 310)), fill=None, outline=255)

                # draw.rectangle(((570 // 2, 570 // 2), (510 // 2, 490 // 2)), fill=None, outline=255)


                self.pattern_copy['image'] = img
                self.pattern_copy.configure(image=img)
                self.pattern_copy.update()

            else:
                black_image = I.new('RGB', (500, 500))
                img = customtkinter.CTkImage(black_image, size=(500, 500))
                self.pattern_copy.configure(image=img)
            self.after(5)

    def renew_current_directory(self, mode='SFDI'):
        """Updates current directory variable"""
        self.current_directory = external_functions.return_current_directory(self.patient_entry.get(), mode)

    def save_thor_image(self, filename=''):
        """Saves an image from thorcam during infrared measurements """
        if self.thor_camera.open:
            self.renew_current_directory('Infrared')
            img = self.thor_camera.get_frame()
            filename = f'{self.current_directory}/{self.tabview.infrared_name_entry.get()}_{self.tabview.exposure_entry.get()}_1.TIF'

            for i in range(1, 10):

                if os.path.isfile(filename):
                    filename = filename[:-5]
                    filename += f'{i}.TIF'
                else:
                    I.fromarray(img).save(filename)
                    self.log_frame.insert_log('Infrared', filename[38:])

                    break

    def save_rgb_image(self, filename=''):
        """Saves rgb cam image during Photo mode"""
        if self.camera.cap:
            self.renew_current_directory(mode='Photo')
            img = self.camera.get_frame()
            filename = f'{self.current_directory}/1.png'

            for i in range(1, 10):

                if os.path.isfile(filename):
                    filename = filename[:-5]
                    filename += f'{i}.png'
                else:
                    cv2.imwrite(filename, img)
                    self.log_frame.insert_log('RGB', filename[38:])

                    break

    def translate_thor_cam(self):
        """Translates thorcam view"""
        self.flag = True
        while self.flag and self.thor_camera.cam is not None and self.thor_camera.open:
            raw_frame = self.thor_camera.get_frame()
            if raw_frame is not None:
                raw_img = (raw_frame.T.astype('float')[::2, ::2] * 255 // 1023).astype('uint8')
                thor_img = I.fromarray(raw_img).transpose(I.FLIP_LEFT_RIGHT)

                draw = ImageDraw.Draw(thor_img)
                draw.rectangle(roi_rect_on_preview(raw_frame), fill=None, outline=255)

                tk_thor_img = customtkinter.CTkImage(thor_img, size=(np.shape(raw_img)[1],
                                                                     np.shape(raw_img)[0]
                                                                     )

                                                     )

                self.pattern_copy.configure(image=tk_thor_img)

                self.pattern_copy.update()

            else:

                black_image = I.new('RGB', (500, 500))
                img = customtkinter.CTkImage(black_image, size=(500, 500))
                self.pattern_copy.configure(image=img)
            self.after(30)

    def stop(self):
        """Stops cameras and pattern translation"""
        self.thor_camera.stop_acquisition()
        self.camera.release_camera()
        self.flag = False
        external_functions.change_button_state(self, block=False)

    def _setup_sfdi_plots(self):
        """Creates the 2x2 realtime SFDI graph panel."""
        container = customtkinter.CTkFrame(self, width=560, height=390, corner_radius=0)
        container.grid(row=1, column=0, rowspan=2, columnspan=1, pady=(0, 10), padx=20, sticky='nsew')
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)
        self.sfdi_plot_container = container

        if Figure is None or FigureCanvasTkAgg is None:
            customtkinter.CTkLabel(
                container,
                text='Matplotlib is unavailable',
                width=560,
                height=390,
            ).grid(row=0, column=0, sticky='nsew')
            self.sfdi_canvas = None
            self.sfdi_axes = {}
            return

        self.sfdi_figure = Figure(figsize=(5.6, 3.9), dpi=100, facecolor='#242424')
        axes = self.sfdi_figure.subplots(2, 2)
        self.sfdi_axes = {
            ('green', 'mua'): axes[0][0],
            ('green', 'mus'): axes[0][1],
            ('red', 'mua'): axes[1][0],
            ('red', 'mus'): axes[1][1],
        }
        self.sfdi_figure.tight_layout(pad=1.3)

        self.sfdi_canvas = FigureCanvasTkAgg(self.sfdi_figure, master=container)
        self.sfdi_canvas.get_tk_widget().grid(row=0, column=0, sticky='nsew')
        self._redraw_sfdi_plots()

    def reset_sfdi_plots(self):
        """Clears realtime SFDI graph history."""
        for color in self.sfdi_plot_history:
            for parameter in self.sfdi_plot_history[color]:
                self.sfdi_plot_history[color][parameter].clear()
        self._redraw_sfdi_plots()

    def update_sfdi_plots(self, metrics):
        """Appends one realtime SFDI result to the graph history."""
        for color in ['green', 'red']:
            for parameter in ['mua', 'mus']:
                value = metrics.get(color, {}).get(parameter)
                if value is not None:
                    self.sfdi_plot_history[color][parameter].append(value)
        self._redraw_sfdi_plots()

        mua_green = metrics.get('green', {}).get('mua')
        mua_red = metrics.get('red', {}).get('mua')
        if mua_green is not None and mua_red is not None:
            self.update_hemoglobin(mua_green, mua_red)

    def compute_hemoglobin(self, mua_green: float, mua_red: float) -> float:
        return (mua_green + mua_red) / 2.0

    def update_hemoglobin(self, mua_green: float, mua_red: float) -> None:
        value = self.compute_hemoglobin(mua_green, mua_red)
        self.hemo_value_label.configure(text=f"{value:.4f}")

    def _redraw_sfdi_plots(self):
        if not getattr(self, 'sfdi_canvas', None):
            return

        titles = {
            ('green', 'mua'): 'green mu_a',
            ('green', 'mus'): "green mu_s'",
            ('red', 'mua'): 'red mu_a',
            ('red', 'mus'): "red mu_s'",
        }
        colors = {'green': '#45c46a', 'red': '#e55757'}

        for key, axis in self.sfdi_axes.items():
            color, parameter = key
            values = self.sfdi_plot_history[color][parameter]
            axis.clear()
            axis.set_facecolor('#1f1f1f')
            axis.set_title(titles[key], color='#f0f0f0', fontsize=9)
            axis.set_xlabel('measurement', color='#d0d0d0', fontsize=7)
            axis.set_ylabel('mm^-1', color='#d0d0d0', fontsize=7)
            axis.tick_params(colors='#d0d0d0', labelsize=7)
            axis.grid(True, color='#3a3a3a', linewidth=0.5)
            for spine in axis.spines.values():
                spine.set_color('#5a5a5a')

            if values:
                x = list(range(1, len(values) + 1))
                axis.plot(x, values, color=colors[color], linewidth=1.4)
                axis.scatter(x, values, color=colors[color], s=14)
                axis.set_xlim(0.8, len(values) + 0.2)
            else:
                axis.set_xlim(0.8, 1.2)

        self.sfdi_figure.tight_layout(pad=1.3)
        self.sfdi_canvas.draw_idle()

    def save_factors_snapshot(self):
        """
        Сохраняет текущие factors в отдельный CSV-файл
        внутри текущей папки измерения (создаётся при каждом запуске SFDI)
        """
        if not hasattr(self, 'current_directory') or not self.current_directory:
            return

        # Время запуска — чтобы имя файла было уникальным
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # Имя файла
        filename = f"factors_{timestamp}.csv"
        filepath = os.path.join(self.current_directory, filename)

        # Данные
        g, b, r = self.pattern_factors

        fieldnames = ['parameter', 'value', 'description']
        rows = [
            {'parameter': 'timestamp', 'value': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
             'description': 'Time'},
            {'parameter': 'green_factor', 'value': f"{g:.4f}", 'description': 'Green'},
            {'parameter': 'blue_factor', 'value': f"{b:.4f}", 'description': 'Blue'},
            {'parameter': 'red_factor', 'value': f"{r:.4f}", 'description': 'Red'},
        ]

        try:
            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

            print(f"Factors сохранены - {filepath}")
        except Exception as e:
            print(f"Ошибка сохранения factors: {e}")

    def begin_sfdi(self):
        """Runs one short realtime SFDI cycle."""
        return run_realtime_sfdi_cycle(self)

    def begin_auto_sfdi(self):
        """Toggle auto-capture mode (one cycle every 10 seconds)."""
        self.auto_capture = not self.auto_capture
        label = "Auto: ON" if self.auto_capture else "Auto: OFF"
        self.tabview.auto_sfdi_button.configure(text=label)
        if self.auto_capture:
            self._auto_sfdi_loop()

    def _auto_sfdi_loop(self):
        if not self.auto_capture:
            return
        external_functions.create_patient_directory(self.patient_entry.get())
        self.renew_current_directory()
        run_realtime_sfdi_cycle(self)
        self.after(10000, self._auto_sfdi_loop)

    def begin_full_sfdi(self):
        """A loop for pattern translation to projector, taking thorcam photos and
        saving them in a relevant directory. Rather large function for now."""
        blue_flag = True
        green_flag = True
        red_flag = True
        time_multiplication_factor = 1.0
        save_only_pattern_part = True
        red_exposure_time = 66.68 * 3 # МЕНЯЛИ ДЛЯ УСТРАННЕНИЯ РАССИНХРОНА

        if self.thor_camera.cam:
            self.thor_camera.cam.set_exposure(self.exposure)

            self.thor_camera.cam.start_acquisition(auto_start=False, nframes=1, frames_per_trigger=1)
            pattern_list = list(self.patterns.items())
            pattern_list = pattern_list + [pattern_list[-1]]
            self.flag = True
            external_functions.change_button_state(self, block=True)

        else:
            self.log_frame.insert_log('Exception')
            return

        for i, (key, img_name) in enumerate(pattern_list[:]):
            if not self.flag:
                self.projection_window.set_background()
                break
            with I.open(img_name[0]) as im:

                enhancer = ImageEnhance.Brightness(im)
                color_name = key[0]  # 'green', 'blue' или 'red'
                color_idx = {'green': 0, 'blue': 1, 'red': 2}[color_name]

                # Берём factor из настроек
                current_factor = self.pattern_factors[color_idx]

                img = enhancer.enhance(current_factor)
            img = customtkinter.CTkImage(img, size=(np.shape(img)[1], np.shape(img)[0]))

            self.projection_window.pattern_window['image'] = img
            self.projection_window.pattern_window.configure(image=img)
            self.projection_window.update()
            self.after(int(time_multiplication_factor*30))

            raw_img = self.thor_camera.get_frame()

            if raw_img is not None:
                translation_img = (raw_img.T.astype('float')[::2, ::2] * 255 // 1023).astype('uint8')
                translation_img = I.fromarray(translation_img).transpose(I.FLIP_LEFT_RIGHT)

                draw = ImageDraw.Draw(translation_img)
                draw.rectangle(((570//2, 570//2), (510//2, 490//2)), fill=None, outline=255)

                thor_img = I.fromarray(raw_img)
                tk_thor_img = customtkinter.CTkImage(translation_img, size=(np.shape(translation_img)[1],
                                                                            np.shape(translation_img)[0]))

                file_name = f'{self.current_directory}/{pattern_list[i - 1][0][0]}/{pattern_list[i - 1][0][1]}.TIF'

                if os.path.isfile(file_name):
                    self.patient_entry.configure(state='normal')
                    while os.path.isfile(file_name):
                        self.patient_entry.insert('end', '_1')
                        external_functions.create_patient_directory(self.patient_entry.get(), modes=['SFDI'])
                        self.renew_current_directory('SFDI')
                        file_name = f'{self.current_directory}/{pattern_list[i - 1][0][0]}/{pattern_list[i - 1][0][1]}.TIF'
                    self.patient_entry.configure(state='disabled')

                if i != 0:
                    save_sfdi_image(thor_img, file_name,
                                    save_only_pattern_part=save_only_pattern_part,
                                    pattern_coords=PATTERN_COORDS_TO_SAVE)


                self.pattern_copy.configure(image=tk_thor_img)
                self.pattern_copy.update()
                self.after(int(time_multiplication_factor*15))


            if red_flag and 'red' in img_name[0]:
                self.thor_camera.change_exposition(red_exposure_time)
                red_flag = False
            self.after(int(time_multiplication_factor*20))


            # if red_flag and ' red' in img_name[0]:
            #     self.thor_camera.change_exposition(100)
            #     red_flag = False



        self.log_frame.insert_log('SFDI')
       # self.after(2000)
        #self.thor_camera.cam.stop_acquisition()
        #self.after(2000)
        external_functions.change_button_state(self, block=False)
        self.after(2000)
        self.save_factors_snapshot()

    def open_brightness_window(self):
        """Окошко для настройки factors"""
        if hasattr(self, 'brightness_win') and self.brightness_win.winfo_exists():
            self.brightness_win.lift()
            return

        win = customtkinter.CTkToplevel(self)
        win.title("Настройка factors")
        win.geometry("380x420")
        win.resizable(False, False)
        self.brightness_win = win

        customtkinter.CTkLabel(
            win,
            text="Яркость паттернов (factors)",
            font=("Arial", 16, "bold")
        ).pack(pady=15)

        colors = ["green", "blue", "red"]
        hex_colors = ["#00ff88", "#4488ff", "#ff4444"]
        self.factor_labels = {}

        for i, (col, hex_col) in enumerate(zip(colors, hex_colors)):
            frame = customtkinter.CTkFrame(win)
            frame.pack(pady=8, padx=30, fill="x")

            customtkinter.CTkLabel(
                frame,
                text=col.capitalize(),
                text_color=hex_col,
                font=("Arial", 14, "bold")
            ).pack()

            # Текущее значение
            val_label = customtkinter.CTkLabel(frame, text=f"{self.pattern_factors[i]:.3f}", width=80)
            val_label.pack(pady=2)
            self.factor_labels[col] = val_label

            # Ползунок
            slider = customtkinter.CTkSlider(
                frame,
                from_=0,
                to=1,
                number_of_steps=100,
                command=lambda v, idx=i, c=col: self._update_factor_debug(idx, v, c)
            )
            slider.set(self.pattern_factors[i])
            slider.pack(fill="x", padx=20, pady=5)

        customtkinter.CTkButton(
            win,
            text="Закрыть",
            command=win.destroy
        ).pack(pady=15)

        self._print_factors("Окошко открыто")

    def _update_factor_debug(self, idx, value, color_name):
        """Обновляет factor и выводит в консоль отладку"""
        value = round(value, 3)
        self.pattern_factors[idx] = value
        self.factor_labels[color_name].configure(text=f"{value:.3f}")

        self._print_factors(f"Изменено: {color_name}")

    def _print_factors(self, event=""):
        """Печатает текущие factors"""
        g, b, r = self.pattern_factors
        print(f"FACTORS {event:20} → Green: {g:.3f} | Blue: {b:.3f} | Red: {r:.3f}")


if __name__ == "__main__":
    external_functions.create_today_directory()
    app = App()
    app.mainloop()
