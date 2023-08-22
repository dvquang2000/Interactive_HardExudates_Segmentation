import tkinter as tk
from tkinter import *
from tkinter import messagebox, filedialog, ttk

import cv2
import numpy as np
from PIL import Image
from automatic_module.predict import VGGUnet_predict
from interactive_demo.canvas import CanvasImage
from interactive_demo.controller import InteractiveController
from interactive_demo.wrappers import BoundedNumericalEntry, FocusHorizontalScale, FocusCheckButton, \
    FocusButton, FocusLabelFrame

import os



class InteractiveDemoApp(ttk.Frame):
    def __init__(self, master, args, model):
        super().__init__(master)
        self.master = master
        master.title("Hard Exudates Segmentation")
        master.withdraw()
        master.update_idletasks()
        x = (master.winfo_screenwidth() - master.winfo_reqwidth()) / 2
        y = (master.winfo_screenheight() - master.winfo_reqheight()) / 2
        master.geometry("+%d+%d" % (x, y))
        self.pack(fill="both", expand=True)

        self.brs_modes = ['NoBRS', 'RGB-BRS', 'DistMap-BRS', 'f-BRS-A', 'f-BRS-B', 'f-BRS-C']
        self.limit_longest_size = args.limit_longest_size

        self.controller = InteractiveController(model, args.device,
                                                predictor_params={'brs_mode': 'NoBRS'},
                                                update_image_callback=self._update_image)
        
        self.automatic_segmentation_model = args.automatic_weight
        self.filename = []
        self.image_size = ()
        self._init_state()
        self._add_menu()
        self._add_canvas()
        self._add_buttons()
        self._analys_information()

        master.bind('<space>', lambda event: self.controller.finish_object())
        master.bind('a', lambda event: self.controller.partially_finish_object())

        self.state['zoomin_params']['skip_clicks'].trace(mode='w', callback=self._reset_predictor)
        self.state['zoomin_params']['target_size'].trace(mode='w', callback=self._reset_predictor)
        self.state['zoomin_params']['expansion_ratio'].trace(mode='w', callback=self._reset_predictor)
        self.state['predictor_params']['net_clicks_limit'].trace(mode='w', callback=self._change_brs_mode)
        self.state['lbfgs_max_iters'].trace(mode='w', callback=self._change_brs_mode)
        self._change_brs_mode()



    def _init_state(self):
        self.state = {
            'zoomin_params': {
                'use_zoom_in': tk.BooleanVar(value=True),
                'fixed_crop': tk.BooleanVar(value=True),
                'skip_clicks': tk.IntVar(value=-1),
                'target_size': tk.IntVar(value=min(400, self.limit_longest_size)),
                'expansion_ratio': tk.DoubleVar(value=1.4)
            },

            'predictor_params': {
                'net_clicks_limit': tk.IntVar(value=8)
            },
            'brs_mode': tk.StringVar(value='NoBRS'),
            'prob_thresh': tk.DoubleVar(value=0.5),
            'lbfgs_max_iters': tk.IntVar(value=20),

            'alpha_blend': tk.DoubleVar(value=0.5),
            'click_radius': tk.IntVar(value=3),

            'roi_size': tk.IntVar(value=10),
            'manual_button_size': tk.IntVar(value=5)
        }

    def _add_menu(self):
        self.menubar = FocusLabelFrame(self, bd=1)
        self.menubar.pack(side=tk.TOP, fill='x')

        button = FocusButton(self.menubar, text='Load image', command=self._load_image_callback)
        button.pack(side=tk.LEFT)
        self.save_mask_btn = FocusButton(self.menubar, text='Save mask', command=self._save_mask_callback)
        self.save_mask_btn.pack(side=tk.LEFT)
        self.save_mask_btn.configure(state=tk.DISABLED)


        self.load_mask_btn = FocusButton(self.menubar, text='Load mask', command=self._load_mask_callback)
        self.load_mask_btn.pack(side=tk.LEFT)
        self.load_mask_btn.configure(state=tk.DISABLED)

        button = FocusButton(self.menubar, text='About', command=self._about_callback)
        button.pack(side=tk.LEFT)
        button = FocusButton(self.menubar, text='Exit', command=self.master.quit)
        button.pack(side=tk.LEFT)

    def _add_canvas(self):
        self.canvas_frame = FocusLabelFrame(self, text="Image")
        self.canvas_frame.rowconfigure(0, weight=1)
        self.canvas_frame.columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(self.canvas_frame, highlightthickness=0, cursor="hand1", width=400, height=400)
        self.canvas.grid(row=0, column=0, sticky='nswe', padx=5, pady=5)

        self.image_on_canvas = None
        self.canvas_frame.pack(side=tk.LEFT, fill="both", expand=True, padx=5, pady=5)

    def _add_buttons(self):
        self.control_frame = FocusLabelFrame(self, text="Controls")
        self.control_frame.pack(side=tk.TOP, fill='x', padx=5, pady=5)
        master = self.control_frame

        self.seg_frame = FocusLabelFrame(master)
        self.seg_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=3)
        self.segmentation_button= \
            FocusButton(self.seg_frame, text='Auto segmentation', bg='#b6d7a8', fg='black', width=45, height=2,
                        state=tk.DISABLED, command=self._load_mask_automatic)
        self.segmentation_button.pack(side=tk.LEFT, fill=tk.X, padx=10, pady=3)
        

        self.clicks_options_frame = FocusLabelFrame(master, text="Clicks management")
        self.clicks_options_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=3)
        self.finish_object_button = \
            FocusButton(self.clicks_options_frame, text='Finish\nobject', bg='green', fg='black', width=10, height=2,
                        state=tk.DISABLED, command=self.controller.finish_object)
        self.finish_object_button.pack(side=tk.LEFT, fill=tk.X, padx=10, pady=3)
        self.undo_click_button = \
            FocusButton(self.clicks_options_frame, text='Undo click', bg='#ffe599', fg='black', width=10, height=2,
                        state=tk.DISABLED, command=self.controller.undo_click)
        self.undo_click_button.pack(side=tk.LEFT, fill=tk.X, padx=10, pady=3)
        self.reset_clicks_button = \
            FocusButton(self.clicks_options_frame, text='Reset clicks', bg='#ea9999', fg='black', width=10, height=2,
                        state=tk.DISABLED, command=self._reset_last_object)
        self.reset_clicks_button.pack(side=tk.LEFT, fill=tk.X, padx=10, pady=3)


        self.prob_thresh_frame = FocusLabelFrame(master, text="Predictions threshold")
        self.prob_thresh_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=3)
        FocusHorizontalScale(self.prob_thresh_frame, from_=0.0, to=1.0, command=self._update_prob_thresh,
                             variable=self.state['prob_thresh']).pack(padx=10)

        self.alpha_blend_frame = FocusLabelFrame(master, text="Alpha blending coefficient")
        self.alpha_blend_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=3)
        FocusHorizontalScale(self.alpha_blend_frame, from_=0.0, to=1.0, command=self._update_blend_alpha,
                             variable=self.state['alpha_blend']).pack(padx=10, anchor=tk.CENTER)

        self.click_radius_frame = FocusLabelFrame(master, text="Visualisation click radius")
        self.click_radius_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=3)
        FocusHorizontalScale(self.click_radius_frame, from_=0, to=7, resolution=1, command=self._update_click_radius,
                             variable=self.state['click_radius']).pack(padx=10, anchor=tk.CENTER)

        self.roi_thresh = FocusLabelFrame(master, text="ROI radius")
        self.roi_thresh.pack(side=tk.TOP, fill=tk.X, padx=10, pady=3)
        FocusHorizontalScale(self.roi_thresh, from_=0, to=200, resolution=1, command=self._update_roi_size,
                             variable=self.state['roi_size']).pack(padx=10, anchor=tk.CENTER)
        

    def _analys_information(self):
        self.segmentation_information_frame = FocusLabelFrame(self, text="Information")
        self.segmentation_information_frame.pack(side=tk.TOP, fill='x', padx=5, pady=5)
        master = self. segmentation_information_frame


        self.information_base_frame = FocusLabelFrame(master, text="Base_information")
        self.information_base_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=3)

        tk.Label(self.information_base_frame, text="File name").grid(row=0, column=0, pady=1, sticky='e')
        tk.Label(self.information_base_frame, text="Image size").grid(row=1, column=0, pady=1, sticky='e')
        tk.Label(self.information_base_frame, text="Inner circle area").grid(row=2, column=0, pady=1, sticky='e')


        self.textboxFilename = tk.Text(self.information_base_frame,width=25, height=1.2)
        self.textboxFilename.grid(row=0, column=1, padx=10, pady=1, sticky='w')
        self.textboxFilename.insert('1.0', 'None')

        self.textboxSizeImage = tk.Text(self.information_base_frame,width=12, height=1)
        self.textboxSizeImage.grid(row=1, column=1, padx=10, pady=1, sticky='w')
        self.textboxSizeImage.insert('1.0', str(0))


        self.textboxInnerCircleArea = tk.Text(self.information_base_frame,width=8, height=1)
        self.textboxInnerCircleArea.grid(row=2, column=1, padx=10, pady=1, sticky='w')
        self.textboxInnerCircleArea.insert('1.0', str(0))

        self.information_base_frame.columnconfigure((0, 1, 2,), weight=1)

        self.slices_options_frame = FocusLabelFrame(master, text="Segmentation_information")
        self.slices_options_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=3)


        tk.Label(self.slices_options_frame, text="Total segmented pixels").grid(row=0, column=0, pady=1, sticky='e')
        tk.Label(self.slices_options_frame, text="Number of components").grid(row=1, column=0, pady=1, sticky='e')
        tk.Label(self.slices_options_frame, text="EX/Inner circle area").grid(row=2, column=0, pady=1, sticky='e')
        tk.Label(self.slices_options_frame, text="Max area").grid(row=3, column=0, pady=1, sticky='e')
        tk.Label(self.slices_options_frame, text="Min area").grid(row=4, column=0, pady=1, sticky='e')
        
        self.textboxSliceIndex = tk.Text(self.slices_options_frame, width=10, height=1)
        self.textboxSliceIndex.grid(row=0, column=1, padx=10, pady=1, sticky='w')
        self.textboxSliceIndex.insert('1.0', str(0))

        self.textboxSliceIndex1 = tk.Text(self.slices_options_frame, width=10, height=1)
        self.textboxSliceIndex1.grid(row=1, column=1, padx=10, pady=1, sticky='w')
        self.textboxSliceIndex1.insert('1.0', str(0))

        self.textboxSliceIndex2 = tk.Text(self.slices_options_frame, width=10, height=1)
        self.textboxSliceIndex2.grid(row=2, column=1, padx=10, pady=1, sticky='w')
        self.textboxSliceIndex2.insert('1.0', str(0))



        self.textboxMaxarea = tk.Text(self.slices_options_frame, width=10, height=1)
        self.textboxMaxarea.grid(row=3, column=1, padx=10, pady=1, sticky='w')
        self.textboxMaxarea.insert('1.0', str(0))

        self.textboxMinarea = tk.Text(self.slices_options_frame, width=10, height=1)
        self.textboxMinarea.grid(row=4, column=1, padx=10, pady=1, sticky='w')
        self.textboxMinarea.insert('1.0', str(0))

        self.slices_options_frame.columnconfigure((0, 1, 2,), weight=1)
        

       

    def _load_image_callback(self):
        self.menubar.focus_set()
        if self._check_entry(self):
            filename = filedialog.askopenfilename(parent=self.master, filetypes=[
                ("Images", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("All files", "*.*"),
            ], title="Chose an image")

            if len(filename) > 0:
                self.filename = filename
                image = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (1024,1024))
                self.image_size = image.shape[:-1]
                self.textboxSizeImage.delete('1.0', tk.END)
                self.textboxSizeImage.insert('1.0', str(self.image_size))
                self.controller.set_image(image)

                self.save_mask_btn.configure(state=tk.NORMAL)
                self.load_mask_btn.configure(state=tk.NORMAL)

                self.segmentation_button.configure(state=tk.NORMAL)

                self.textboxInnerCircleArea.delete('1.0', tk.END)
                self.textboxInnerCircleArea.insert('1.0', self.cal_area_inner_circle())
                self.textboxFilename.delete('1.0', tk.END)
                self.textboxFilename.insert('1.0', str(os.path.split(self.filename)[1]))
                
                image_name = str(os.path.split(self.filename)[1][:-4]) + '.png'
                self.controller.update_filename(image_name)

    def _save_mask_callback(self):
        self.menubar.focus_set()
        if self._check_entry(self):
            mask = self.controller.result_mask
            if mask is None:
                return

            filename = filedialog.asksaveasfilename(parent=self.master, initialfile=str(os.path.split(self.filename)[1][:-4]) + '.tiff', filetypes=[
                ("PNG image", "*.png"),
                ("BMP image", "*.bmp"),
                ("TIFF image", "*.tiff"),
                ("All files", "*.*"),
            ], title="Save the current mask as...")

            if len(filename) > 0:
                if mask.max() < 256:
                    mask = mask.astype(np.uint8)
                    mask *= 255 // mask.max()
                cv2.imwrite(filename, mask)

    def _load_mask_callback(self):
        if not self.controller.net.with_prev_mask:
            messagebox.showwarning("Warning", "The current model doesn't support loading external masks. "
                                              "Please use ITER-M models for that purpose.")
            return

        self.menubar.focus_set()
        if self._check_entry(self):
            filename = filedialog.askopenfilename(parent=self.master, filetypes=[
                ("Binary mask (png, bmp, tiff)", "*.png *.bmp *.tif"),
                ("All files", "*.*"),
            ], title="Chose an image")

            if len(filename) > 0:
                mask = cv2.imread(filename)[:, :, 0] > 127
                self.controller.set_mask(mask)
                self._update_image()
    
    def _load_mask_automatic(self):
        if not self.controller.net.with_prev_mask:
            messagebox.showwarning("Warning", "The current model doesn't support loading external masks. "
                                              "Please use ITER-M models for that purpose.")
            return
        self.menubar.focus_set()
        if self._check_entry(self):
            if len(self.filename) > 0:
                mask = VGGUnet_predict(self.filename, self.automatic_segmentation_model)
                mask = mask[:, :, 0] / 255.0
                self.controller.set_mask(mask)
                self._update_image()

    def _about_callback(self):
        self.menubar.focus_set()

        text = [
            "Developed by:",
            "K.Sofiiuk and I. Petrov",
            "The MIT License, 2021"
        ]

        messagebox.showinfo("About Demo", '\n'.join(text))

    def _reset_last_object(self):
        self.state['alpha_blend'].set(0.5)
        self.state['prob_thresh'].set(0.5)
        self.controller.reset_last_object()

    def _update_prob_thresh(self, value):
        if self.controller.is_incomplete_mask:
            self.controller.prob_thresh = self.state['prob_thresh'].get()
            self._update_image()

    def _update_blend_alpha(self, value):
        self._update_image()

    def _update_click_radius(self, *args):
        if self.image_on_canvas is None:
            return

        self._update_image()

    def _update_roi_size(self, value):
        self.controller.update_roi_size(self.state['roi_size'].get())
        self._update_image()


    def _update_manual_button_size(self, value):
        self.controller.update_manual_button_size(self.state['manual_button_size'].get())
        self._update_image()

    def _change_brs_mode(self, *args):
        self._reset_predictor()

    def _reset_predictor(self, *args, **kwargs):
        brs_mode = self.state['brs_mode'].get()
        prob_thresh = self.state['prob_thresh'].get()
        net_clicks_limit = None if brs_mode == 'NoBRS' else self.state['predictor_params']['net_clicks_limit'].get()

        if self.state['zoomin_params']['use_zoom_in'].get():
            zoomin_params = {
                'skip_clicks': self.state['zoomin_params']['skip_clicks'].get(),
                'target_size': self.state['zoomin_params']['target_size'].get(),
                'expansion_ratio': self.state['zoomin_params']['expansion_ratio'].get()
            }
            if self.state['zoomin_params']['fixed_crop'].get():
                zoomin_params['target_size'] = (zoomin_params['target_size'], zoomin_params['target_size'])
        else:
            zoomin_params = None

        predictor_params = {
            'brs_mode': brs_mode,
            'prob_thresh': prob_thresh,
            'zoom_in_params': zoomin_params,
            'predictor_params': {
                'net_clicks_limit': net_clicks_limit,
                'max_size': self.limit_longest_size
            },
            'brs_opt_func_params': {'min_iou_diff': 1e-3},
            'lbfgs_params': {'maxfun': self.state['lbfgs_max_iters'].get()}
        }
        self.controller.reset_predictor(predictor_params)

    def _click_callback(self, is_positive, x, y):
        self.canvas.focus_set()

        if self.image_on_canvas is None:
            messagebox.showwarning("Warning", "Please load an image first")
            return

        if self._check_entry(self):
            self.controller.add_click(x, y, is_positive)

    def _update_image(self, reset_canvas=False):
        image = self.controller.get_visualization(alpha_blend=self.state['alpha_blend'].get(),
                                                  click_radius=self.state['click_radius'].get())
        if self.image_on_canvas is None:
            self.image_on_canvas = CanvasImage(self.canvas_frame, self.canvas)
            self.image_on_canvas.register_click_callback(self._click_callback)

        self._set_click_dependent_widgets_state()
        if image is not None:
            self.image_on_canvas.reload_image(Image.fromarray(image), reset_canvas)
            self.textboxSliceIndex.delete('1.0', tk.END)
            self.textboxSliceIndex.insert('1.0', str(self.controller.lesion_pixels_number()))

            self.textboxSliceIndex1.delete('1.0', tk.END)
            self.textboxSliceIndex1.insert('1.0', str(self.controller.lesion_number()[0]))


            self.textboxMaxarea.delete('1.0', tk.END)
            self.textboxMaxarea.insert('1.0', str(self.controller.lesion_number()[2]))

            self.textboxMinarea.delete('1.0', tk.END)
            self.textboxMinarea.insert('1.0', str(self.controller.lesion_number()[1]))


            self.textboxSliceIndex2.delete('1.0', tk.END)
            self.textboxSliceIndex2.insert('1.0', str(round(100 * self.controller.lesion_pixels_number() / self.cal_area_inner_circle(), 3)) +'%')


    def _set_click_dependent_widgets_state(self):
        after_1st_click_state = tk.NORMAL if self.controller.is_incomplete_mask else tk.DISABLED
        before_1st_click_state = tk.DISABLED if self.controller.is_incomplete_mask else tk.NORMAL

        self.finish_object_button.configure(state=after_1st_click_state)
        self.undo_click_button.configure(state=after_1st_click_state)
        self.reset_clicks_button.configure(state=after_1st_click_state)

    def cal_area_inner_circle(self):
        if self._check_entry(self):
            if len(self.filename) == 0:
                return 0
            image = cv2.imread(self.filename)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            total_pixel = gray.size
            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            output = cv2.connectedComponentsWithStats(thresh, 8, cv2.CV_32S)
            (numLabels, labels, stats, centroids) = output
            result = total_pixel - stats[1][-1]
            return result

    def _check_entry(self, widget):
        all_checked = True
        if widget.winfo_children is not None:
            for w in widget.winfo_children():
                all_checked = all_checked and self._check_entry(w)

        if getattr(widget, "_check_bounds", None) is not None:
            all_checked = all_checked and widget._check_bounds(widget.get(), '-1')

        return all_checked