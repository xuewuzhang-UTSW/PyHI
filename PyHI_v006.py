#!/usr/bin/python3
"""
Run with Python3
Dependence: numpy, matplotlib, PIL, PyQt5, mrcfile, scipy, mplcursors
"""

import sys
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import functools 
import mplcursors
from PIL import Image, ImageOps
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QGuiApplication
import PyQt5.QtCore as QtCore
import matplotlib.backends.backend_qt5agg as plt_qtbackend
import mrcfile
from numpy.core.fromnumeric import ndim
from scipy.optimize import minimize
from scipy import ndimage, spatial, special, signal

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self._CentralWidget = QTabWidget(self)
        self.setWindowTitle('PyHI')
        self.setCentralWidget(self._CentralWidget)
        self.tab1 = QWidget()
        self.tab2 = QWidget()
        self.tab3 = QWidget()
        self.tab1.layout = QHBoxLayout()
        self.tab2.layout = QHBoxLayout()
        self.tab3.layout = QHBoxLayout() 
        self.tab1.setLayout(self.tab1.layout)
        self.tab2.setLayout(self.tab2.layout)
        self.tab3.setLayout(self.tab3.layout)
        self._CentralWidget.addTab(self.tab1, "Power spectrum analyzer")
        self._CentralWidget.addTab(self.tab2, "Lattice generator")
        self._CentralWidget.addTab(self.tab3, "3D model")

        self.mainMenu = self.menuBar()
        self.mainMenu.setNativeMenuBar(False)
        self.fileMenu = self.mainMenu.addMenu(' &File')

        self.imgFileOpenButton = QAction(' &Open 2D Images', self)
        self.imgFileOpenButton.triggered.connect(self.open_2D_classes)
        self.fileMenu.addAction(self.imgFileOpenButton)

        self.PowerspecFileOpenButton = QAction(' &Open power spectrum', self)
        self.PowerspecFileOpenButton.triggered.connect(self.load_power_spec)
        self.fileMenu.addAction(self.PowerspecFileOpenButton)

        self.fftSaveButton = QAction(' &Save Curent Power Spectrum', self)
        self.fftSaveButton.triggered.connect(self.save_power_spec)
        self.fftSaveButton.setEnabled(False)
        self.fileMenu.addAction(self.fftSaveButton)

        self.paramSaveButton = QAction(' &Save Parameters', self)
        self.paramSaveButton.triggered.connect(self.save_para)
        self.paramSaveButton.setEnabled(False)
        self.fileMenu.addAction(self.paramSaveButton)

        self.paramLoadButton = QAction(' &Load Parameters', self)
        self.paramLoadButton.triggered.connect(self.load_para)
        self.paramLoadButton.setEnabled(False)
        self.fileMenu.addAction(self.paramLoadButton)

        self.exitButton = QAction(' &Exit', self)
        self.exitButton.triggered.connect(self.close)
        self.fileMenu.addAction(self.exitButton)
        
        self.default_parameters()
        self.overall_layout()
        self.button_push_connect()
    
    def calc_vector_length_angle(self, x, y):
        l = math.sqrt(x**2+y**2)
        a = math.acos(x/l)*180/math.pi
        return [l, a]
    
    def default_parameters(self):
        #global parameters
        self.power_spec_only = 0
        self.LL_distance = 1
        self.angpix = 1

        # Tab1 default Parameters
        self.origin = [0,0]
        self.tab1_LL = []
        self.tab1_LL_label = []

        self.mrc_data_array = None
        self.twoD_img_shown = None
        self.midpoint_of_2dimage = None
        self.oneD_profile_of_2dimage = None
        self.tab1_fft_shown = None
        self.measure_on = False
        self.tab1_LL_draw_on = False
        self.dist_line_in_2D = ''

        self.radius_H = 0
        self.radius_error = 0
        self.ruler_width = 0
        self.ruler_height = 0

        self.LL_amp_plot = ''
        self.LL_bessel_plot = ''
        self.LL_bessel_plot_fill_low = ''
        self.LL_bessel_plot_fill_hihg = ''
        self.LL_legend = ''
        self.LL_phase_plot = ''
        self.LL_phase_diff_ind = ''
        self.n_Bessel = 0
        self.current_img_fft_amp = None
        
        #Tab2 default parameters
        self.tab2_fft_shown = None
        self.tab2_lattice_lines = []
        self.x_v1 = 15
        self.y_v1 = 4
        self.x_v2 = -5
        self.y_v2 = 12
        self.n_lines1 = 2
        self.n_lines2 = 2
        self.n_lower_lines1 = 0
        self.n_lower_lines2 = 0
        self.v1_n = 0
        self.v2_n = 0
        self.tab2_LL = []
        self.tab2_line_labels = []
        self.n_tab2_LL = 30
        self.tab2_fft_ax_lw = 1.0
        self.v1_lattice_lw = 0.5
        self.v2_lattice_lw = 0.5
        self.tab2_LL_lw = 0.5
        self.tab2_fft_peaks_plot = []
        self.tab2_fft_peaks_annot = []

        self.ori_unitV_lines_rs = []
        self.circum_rs = 0
        self.rise_rs_main = 0
        self.twist_rs_main = 360
        self.n_start_main = 1

        self.dots_rs = []
        self.dots_rs_label = {} 
        self.dots_rs_seq = []
        self.dots_rs_plot = None
        self.strand_line_rs = []
        self.strand_line_info = []

    def overall_layout(self):
        #Tab1
        self.tab1_left_side = QVBoxLayout()
        self.tab1_right_side = QVBoxLayout()
        self.tab1.layout.addLayout(self.tab1_left_side)
        self.tab1.layout.addLayout(self.tab1_right_side)
        self.tab1_fft_window()

        self.tab1_ctrl_layout = QHBoxLayout()
        self.tab1_ctrl_layout.setContentsMargins(0, 0, 0, 0)
        self.tab1_ctrl_widget = QWidget()
        self.tab1_ctrl_widget.setFixedHeight(250)
        self.tab1_ctrl_widget.setLayout(self.tab1_ctrl_layout)
        self.tab1_left_side.addWidget(self.tab1_ctrl_widget)

        self.tab1_ctrl_left_column()
        self.tab1_ctrl_right_column()
        self.class2D_window()
        self.bessel_plot_window()

        #Tab2
        self.tab2_left_side = QVBoxLayout()
        self.tab2_right_side = QVBoxLayout()
        self.tab2.layout.addLayout(self.tab2_left_side)
        self.tab2.layout.addLayout(self.tab2_right_side)
        self.tab2_fft_window()
        self.realspace_plot_window()

        self.tab2_ctrl_left_layout = QGridLayout()
        self.tab2_ctrl_left_layout.setContentsMargins(0, 0, 0, 0)
        self.tab2_ctrl_left_widget = QWidget()
        self.tab2_ctrl_left_widget.setFixedSize(500,150)
        self.tab2_ctrl_left_widget.setLayout(self.tab2_ctrl_left_layout)
        self.tab2_left_side.addWidget(self.tab2_ctrl_left_widget)

        self.tab2_ctrl_right_layout = QGridLayout()
        self.tab2_ctrl_right_layout.setContentsMargins(0, 0, 0, 0)
        self.tab2_ctrl_right_widget = QWidget()
        self.tab2_ctrl_right_widget.setFixedSize(500,150)
        self.tab2_ctrl_right_widget.setLayout(self.tab2_ctrl_right_layout)
        self.tab2_right_side.addWidget(self.tab2_ctrl_right_widget)

        self.tab2_right_side_ctrl()
        self.tab2_left_side_ctrl()
        
        #Tab3 
        self.tab3_left_side = QGridLayout()
        self.tab3_left_side.setContentsMargins(0, 0, 0, 0)
        self.tab3_left_side_widget = QWidget()
        self.tab3_left_side_widget.setFixedWidth(300)
        self.tab3_left_side_widget.setLayout(self.tab3_left_side)
        self.tab3.layout.addWidget(self.tab3_left_side_widget)
        self.tab3_right_side = QVBoxLayout()
        self.tab3.layout.addLayout(self.tab3_right_side)
        
        self.tab3_left_side_ctrl()
        self.tab3_3D_plot_window()
    
    def tab1_fft_window(self):
        self.tab1_figfft, self.tab1_axfft = plt.subplots()
        self.tab1_figfft.suptitle('Power spectrum', fontsize=10)
        self.tab1_figfft.tight_layout()
        self.tab1_axfft.axis('off')
        self.tab1_axfft.format_coord = lambda x, y: f'x={round(x)-self.origin[0]:.0f}, y={round(y)-self.origin[1]:.0f}'
        self.tab1_figfft_canvas = plt_qtbackend.FigureCanvasQTAgg(self.tab1_figfft)
        self.tab1_figfft_canvas.setMinimumSize(600,400)
        self.tab1_fft_toolbar = plt_qtbackend.NavigationToolbar2QT(self.tab1_figfft_canvas, self._CentralWidget)
        self.tab1_fft_toolbar.setFixedHeight(35)
        self.tab1_left_side.addWidget(self.tab1_figfft_canvas)
        self.tab1_left_side.addWidget(self.tab1_fft_toolbar)
        
        self.tab1_figfft_canvas.mpl_connect('button_press_event', self.set_origin_by_click)

    def tab1_ctrl_left_column(self):
        layout = QGridLayout()

        self.tab1_labels_col1 = {}
        self.tab1_buttons_col1 = {}
        self.tab1_text_col1 = {}

        labels = {
            'Threshold low:': (0, 0, 1, 1),
            'Threshold high:': (1, 0, 1, 1),
            'Class number:': (2, 0, 1, 1),
            'Rotate Img:': (4, 0, 1, 1),
            '(degree)': (4, 2, 1, 1),
            'Shift Img:': (5, 0, 1, 1),
            '(pixel [x])': (5, 2, 1, 1),
            '(pixel [y])': (6, 2, 1, 1),
            'Repeat distance:': (7, 1, 1, 2),
        }
        buttons = {
            'Set LL dist': (6, 0, 1, 1),
            'Draw LL': (7, 0, 1, 1),
        }

        txt_fields = {
            'Y_dist': (6, 1, 1, 1, self.LL_distance),
        }

        self.minSigma_chooser = QSlider(QtCore.Qt.Horizontal)
        self.minSigma_chooser.setMinimum(-31)
        self.minSigma_chooser.setMaximum(30)
        self.minSigma_chooser.setValue(-1)
        self.minSigma_chooser.setFixedWidth(200)
        layout.addWidget(self.minSigma_chooser, 0, 1, 1, 2)
        self.minSigma_chooser.setEnabled(False)

        self.maxSigma_chooser = QSlider(QtCore.Qt.Horizontal)
        self.maxSigma_chooser.setMinimum(-30)
        self.maxSigma_chooser.setMaximum(31)
        self.maxSigma_chooser.setValue(8)
        self.maxSigma_chooser.setFixedWidth(200)
        layout.addWidget(self.maxSigma_chooser, 1, 1, 1, 2)
        self.maxSigma_chooser.setEnabled(False)

        self.slice_chooser = QSpinBox()
        self.slice_chooser.setValue(1)
        self.slice_chooser.setRange(1,10)
        self.slice_chooser.setEnabled(False)
        self.slice_chooser.setFixedWidth(70)
        layout.addWidget(self.slice_chooser, 2, 1, 1, 1)
        self.slice_chooser.setToolTip('Choose 2D class to display.')

        self.auto_align_toggle = QCheckBox('Align Img')
        self.auto_align_toggle.setCheckable(False)
        layout.addWidget(self.auto_align_toggle, 3, 0, 1, 1)
        self.auto_align_toggle.setToolTip('''Automatic center and align 2D image
        2D image should have positive contrast (protein density is white while background is dark)
        If not, check "Invert Img" before using this function ''')

        self.twoD_inv_toggle = QCheckBox(f'Invert Img')
        self.twoD_inv_toggle.setCheckable(False)
        self.twoD_inv_toggle.setFixedWidth(100)
        layout.addWidget(self.twoD_inv_toggle, 3, 1, 1, 1)
        self.twoD_inv_toggle.setToolTip('Invert black and white of the 2D image')

        self.ps_cmap_toggle = QCheckBox('Invert PS')
        self.ps_cmap_toggle.setCheckable(False)
        layout.addWidget(self.ps_cmap_toggle, 3, 2, 1, 1)
        self.ps_cmap_toggle.setToolTip('Invert black and white of the power spectrum')

        self.img_rotation_chooser = QDoubleSpinBox()
        self.img_rotation_chooser.setValue(0)
        self.img_rotation_chooser.setRange(-180.0,180.0)
        self.img_rotation_chooser.setSingleStep(1)
        self.img_rotation_chooser.setEnabled(False)
        self.img_rotation_chooser.setFixedWidth(70)
        layout.addWidget(self.img_rotation_chooser, 4, 1, 1, 1)
        self.img_rotation_chooser.setToolTip('''Rotate the 2D image and power spectrum to the vertical orientation.
        Use the vertical lines in the images as the references''')

        self.img_shift_chooser = QSpinBox()
        self.img_shift_chooser.setValue(0)
        self.img_shift_chooser.setRange(-500,500)
        self.img_shift_chooser.setEnabled(False)
        self.img_shift_chooser.setFixedWidth(70)
        layout.addWidget(self.img_shift_chooser, 5, 1, 1, 1)
        self.img_shift_chooser.setToolTip('''Shift the 2D image horizontally to the center
        Use the blue vertical line in the 2D image window as the reference''')

        for txt, pos in labels.items():
            self.tab1_labels_col1[txt] = QLabel(txt)
            layout.addWidget(self.tab1_labels_col1[txt], pos[0], pos[1], pos[2], pos[3])
        
        for txt, pos in buttons.items():
            self.tab1_buttons_col1[txt] = QPushButton(txt)
            self.tab1_buttons_col1[txt].setFixedWidth(100)
            self.tab1_buttons_col1[txt].setEnabled(False)
            layout.addWidget(self.tab1_buttons_col1[txt], pos[0], pos[1], pos[2], pos[3])
        
        for txt, pos in txt_fields.items():
            self.tab1_text_col1[txt] = QLineEdit(f'{pos[4]:2.1f}')
            self.tab1_text_col1[txt].setFixedWidth(70)
            layout.addWidget(self.tab1_text_col1[txt], pos[0], pos[1], pos[2], pos[3])

        self.tab1_text_col1['Y_dist'].setToolTip('Layerline distance. (Example: 2.1)')

        self.tab1_buttons_col1['Draw LL'].setToolTip('''Click to toggle layerlines on/off
        Test different LL-distance values above to match layerlines in the power spectrum.\n
        The default origin is at the center of the power spectrum.  If this is incorrect,
        Option/Alt-click on the actual origin to reset it.''')
        self.tab1_buttons_col1['Set LL dist'].setToolTip('''Click to set distance between layerlines and draw layerlines
        Use 1D profile to help find the distance.
        Or hover mouse in locations of the power spectrum, 
        coordinates are shown at the right side of the toolbar\n
        The default origin is at the center of the power spectrum.  If this is incorrect,
        Option/Alt-click on the actual origin to reset it.''')

        divider = QFrame()
        divider.setFrameShape(QFrame.VLine)
        divider.setFrameShadow(QFrame.Sunken)
        layout.addWidget(divider, 0, 3, 8,1)

        self.tab1_ctrl_layout.addLayout(layout)

    def tab1_ctrl_right_column(self):
        layout = QGridLayout()

        self.tab1_labels_col2 = {}
        self.tab1_buttons_col2 = {}
        self.tab1_text_col2 = {}

        buttons = {
            'Set Angpix': (0, 0, 1, 1),
            'Measure': (1, 0, 1, 1),
            'Calc LL plot': (8, 0, 1, 1),
        }

        labels = {
            '(\u212B/pixel)': (0, 3, 1, 1),
            'Measure': (1, 1, 1, 3),
            'Radius (\u212B):': (2, 0, 1, 1),
            '+/-': (2, 2, 1, 1),
            'Layerline plot parameters:': (4, 0, 1, 4),
            'Y-coord range (pixel):': (5, 0, 1, 3),
            'Plot width (pixel):': (6, 0, 1, 3),
            'Bessel order (integer):': (7, 0, 1, 3),
            'CC=': (8, 3, 1, 1),
        }

        txt_fields = {
            'Angpix': (0, 1, 1, 1),
            'helix_radius': (2, 1, 1, 1),
            'radius_error': (2, 3, 1, 1),
            'LL_Y_range': (5, 3, 1, 1),
            'LL_width': (6, 3, 1, 1),
            'Bessel_order': (7, 3, 1, 1),
        }

        for txt, pos in labels.items():
            if txt == 'Measure':
                self.tab1_labels_col2[txt] = QLabel('')
                self.tab1_labels_col2[txt].setFixedWidth(200)
            else:
                self.tab1_labels_col2[txt] = QLabel(txt)
            layout.addWidget(self.tab1_labels_col2[txt], pos[0], pos[1], pos[2], pos[3])
        
        for txt, pos in buttons.items():
            self.tab1_buttons_col2[txt] = QPushButton(txt)
            self.tab1_buttons_col2[txt].setFixedWidth(100)
            self.tab1_buttons_col2[txt].setEnabled(False)
            layout.addWidget(self.tab1_buttons_col2[txt], pos[0], pos[1], pos[2], pos[3])

        for txt, pos in txt_fields.items():
            self.tab1_text_col2[txt] = QLineEdit()
            self.tab1_text_col2[txt].setMaximumWidth(60)
            layout.addWidget(self.tab1_text_col2[txt], pos[0], pos[1], pos[2], pos[3])

        self.tab1_buttons_col2['Set Angpix'].setToolTip('Click to set angstrom/pixel of the image\nNo need to set manually if read from MRC image correctly')
        self.tab1_text_col2['Angpix'].setToolTip('Angpix of image. (Example: 1.08)')
        self.tab1_buttons_col2['Measure'].setToolTip('''click me and then two points in 2D image to measure distance
        To get the radius of the helix, click the two side edges,
        radius will be automatically calculated and set.
        Radius can also be set manually by typing number into the field below.''')
        self.tab1_buttons_col2['Calc LL plot'].setToolTip('''Click to plot the layerline plots
        Needs helix radius and the layerline plot parameters ''')
        self.tab1_text_col2['helix_radius'].setToolTip('Set radius of the helix\nBy using the measure button or typing the number here')
        self.tab1_text_col2['radius_error'].setToolTip('Set +/- range of the radius\nLeave it emtpy if unsure')
        self.tab1_text_col2['LL_Y_range'].setToolTip('''Set low and high bounds of the power spectrum for plotting
        Input two numbers separated by "," (Example: 2,3)
        Put two identical numbers (For example: 2,2) if only want to plot one pixel slice\n
        To find the desired pixel numbers:
        Hover mouse on Power Spectrum, coordinates will be shown at the lower right corner of the Toolbar''')
        self.tab1_text_col2['LL_width'].setToolTip('''Set width of the power spectrum around the miradian for plotting
        Input one number (Example: 30). To find the appropriate width:
        Hover mouse on Power Spectrum, coordinates will be shown at the lower right corner of the Toolbar''')
        self.tab1_text_col2['Bessel_order'].setToolTip('''Expected Bessel order of the layerline peak. (Example: 1)
        Try different numbers, find one such that:
        The first peak of the amplitude plot (blue) matches closely the first Bessel peak (red)
        The phase plot (red) matches closely the predicted red dots\n
        After set correctly, Bessell peaks can be annotated on power spectrum by middle-button click
        Hold and drag to move annotation
        Righ-button click to remove annotation''')

        divider = QFrame()
        divider.setMaximumWidth(260)
        divider.setFrameShape(QFrame.HLine)
        divider.setFrameShadow(QFrame.Sunken)
        layout.addWidget(divider, 3, 0, 1, 4) 

        spacer_item = QSpacerItem(1, 30, QSizePolicy.Expanding, QSizePolicy.Minimum)
        layout.addItem(spacer_item, 0, 3, 7, 1)

        self.tab1_ctrl_layout.addLayout(layout)

    def class2D_window(self):
        self.fig2d, self.ax2d = plt.subplots()
        self.fig2d.suptitle('2D image', fontsize=10)
        self.ax2d.axis('off')
        self.ax2d.format_coord = lambda x, y: f'x={round(x)-self.origin[0]:.0f}, y={round(y)-self.origin[1]:.0f}'
        self.fig2d.tight_layout()
        self.fig2d_canvas = plt_qtbackend.FigureCanvasQTAgg(self.fig2d)
        self.fig2d_canvas.setFixedWidth(500)
        self.toolbar2d = plt_qtbackend.NavigationToolbar2QT(self.fig2d_canvas, self._CentralWidget)
        self.tab1_right_side.addWidget(self.fig2d_canvas)
        self.tab1_right_side.addWidget(self.toolbar2d)
        self.toolbar2d.setFixedSize(500,35)

    def bessel_plot_window(self):
        self.fig_bessel, (self.ax_amp, self.ax_phase) = plt.subplots(2, 1, sharex=True)
        self.fig_bessel.suptitle('Layerline plots', fontsize=10, y=.98)
        self.fig_bessel.subplots_adjust(top=0.90, bottom=0.15, left=0.15, right=0.95)
        self.ax_amp.set_ylabel('Amplitude')
        self.ax_amp.grid(which='both', axis='x')
        self.ax_phase.set_xlabel('R (pixel)')
        self.ax_phase.set_ylabel('\u0394Phase')
        self.ax_phase.set_xlim(-10,10)
        self.ax_phase.xaxis.set_major_locator(MaxNLocator(integer=True))
        self.ax_phase.set_ylim(-10, 190)
        self.ax_phase.set_yticks(list(range(0, 181, 60)))
        self.ax_phase.grid(which='both', axis='x')
        self.fig_bessel_canvas = plt_qtbackend.FigureCanvasQTAgg(self.fig_bessel)
        self.fig_bessel_canvas.setFixedSize(500,300)
        self.toolbar_bessel = plt_qtbackend.NavigationToolbar2QT(self.fig_bessel_canvas, self._CentralWidget) 
        self.toolbar_bessel.setFixedSize(500,25)
        
        self.tab1_right_side.addWidget(self.fig_bessel_canvas)
        self.tab1_right_side.addWidget(self.toolbar_bessel)

    def tab2_fft_window(self):
        self.tab2_figfft, self.tab2_axfft = plt.subplots()
        self.tab2_figfft.suptitle('Fourier space lattice', fontsize=10)
        self.tab2_axfft.axis('off')
        self.tab2_axfft.axis('scaled')
        self.tab2_axfft.format_coord = lambda x, y: f'x={round(x)-self.origin[0]:.0f}, y={round(y)-self.origin[1]:.0f}'
        self.tab2_figfft.tight_layout()
        self.tab2_figfft_canvas = plt_qtbackend.FigureCanvasQTAgg(self.tab2_figfft)
        self.tab2_fft_toolbar = plt_qtbackend.NavigationToolbar2QT(self.tab2_figfft_canvas, self._CentralWidget)
        self.tab2_fft_toolbar.setFixedHeight(35)
        self.tab2_left_side.addWidget(self.tab2_figfft_canvas)                                    
        self.tab2_left_side.addWidget(self.tab2_fft_toolbar)

        self.tab2_figfft_canvas.mpl_connect('button_press_event', self.set_vectors_by_click)

    def realspace_plot_window(self):
        self.fig_rs, self.ax_rs = plt.subplots()
        self.fig_rs.suptitle('Real space lattice', fontsize=10)
        self.ax_rs.axis('scaled')
        self.ax_rs.set_xlabel('azimuthal angle (\u00B0)')
        self.ax_rs.set_ylabel('rise (\u212B)')
        self.fig_rs.tight_layout()
        self.fig_rs_canvas = plt_qtbackend.FigureCanvasQTAgg(self.fig_rs)
        self.toolbar_rs = plt_qtbackend.NavigationToolbar2QT(self.fig_rs_canvas, self)
        self.toolbar_rs.setFixedHeight(35)

        self.tab2_right_side.addWidget(self.fig_rs_canvas)
        self.tab2_right_side.addWidget(self.toolbar_rs)

    def tab2_right_side_ctrl(self):
        self.buttons_rs = {}
        self.labels_rs = {}
        self.text_fields_rs = []
        self.spin_boxes_rs = {}

        buttons = {
            'Sequence label': (0, 0, 1, 1),
            '[h,k] label': (0, 1, 1, 1),
            'Delete last': (4, 0, 1, 1),
            'Delete all': (4, 1, 1, 1),
        }

        labels_rs = {
            'Rise_Twist': (1, 0, 1, 2),
            'Adjust plot boundaries:': (0, 3, 1, 2),
            'X lower bound \u2B0D:': (1, 3, 1, 1),
            'X upper bound \u2B0D:': (2, 3, 1, 1),
            'Y lower bound \u2B0D:': (3, 3, 1, 1),
            'Y upper bound \u2B0D:': (4, 3, 1, 1),
        }

        spinboxes = {
            'x_low': (1, 4, 1, 1),
            'x_high': (2, 4, 1, 1),
            'y_low': (3, 4, 1, 1),
            'y_high': (4, 4, 1, 1),
        }

        for txt, pos in labels_rs.items():
            if 'Rise_Twist' in txt:
                self.labels_rs[txt] = QLabel('')
                self.labels_rs[txt].setStyleSheet("color:red")
            else:
                self.labels_rs[txt] = QLabel(txt)
            self.tab2_ctrl_right_layout.addWidget(self.labels_rs[txt], pos[0], pos[1], pos[2], pos[3])

        for txt, pos in buttons.items():
            self.buttons_rs[txt] = QPushButton(txt)
            self.buttons_rs[txt].setFixedWidth(130)
            self.buttons_rs[txt].setFocusPolicy(QtCore.Qt.NoFocus)
            self.buttons_rs[txt].setEnabled(False)
            self.tab2_ctrl_right_layout.addWidget(self.buttons_rs[txt], pos[0], pos[1], pos[2], pos[3])

        for txt, pos in spinboxes.items():
            self.spin_boxes_rs[txt] = QSpinBox()
            self.spin_boxes_rs[txt].setFixedWidth(60)
            self.spin_boxes_rs[txt].setRange(-10,10)
            self.spin_boxes_rs[txt].setEnabled(False)
            self.tab2_ctrl_right_layout.addWidget(self.spin_boxes_rs[txt], pos[0], pos[1],pos[2], pos[3])

        self.buttons_rs['[h,k] label'].setToolTip('Toggle point label ([h, k] Miller index)')
        self.buttons_rs['Sequence label'].setToolTip('''Toggle point label (sequential number)\n
        This also triggers drawing of the strand(s) representing the primary helical family,
        and the calcualtion of its symmetry parameters (twist, rise and n-stat number),
        which are displayed in the text line below.  
        They can also be displayed on the plot by middle-mouse button click on a strand
        Hold and drag to move the yellow text box; Right-mouse button click to remove\n
        These results may be incorrect. Check with "Draw strand" below to be sure.''')

        self.draw_strand_switch = QCheckBox('Draw strand manually')
        self.draw_strand_switch.setCheckable(False)
        self.tab2_ctrl_right_layout.addWidget(self.draw_strand_switch, 3, 0, 1, 2)
        self.draw_strand_switch.setToolTip('''Toggle switch for drawing strand\n
        After successful refine, check this check box and
        Cmd/Ctrl-click two neighboring points to define a strand\n
        Symmetry parameters of individual strand families are printed in the terminal window
        They can also be displayed on the plot by middle-mouse button click on a strand
        Hold and drag to move the yellow text box; Right-mouse button click to remove''')

        divider1_rs = QFrame()
        divider1_rs.setFrameShape(QFrame.VLine)
        divider1_rs.setFrameShadow(QFrame.Sunken)
        self.tab2_ctrl_right_layout.addWidget(divider1_rs, 0, 2, 5, 1)
        divider2_rs = QFrame()
        divider2_rs.setFrameShape(QFrame.HLine)
        divider2_rs.setFrameShadow(QFrame.Sunken)
        self.tab2_ctrl_right_layout.addWidget(divider2_rs, 2, 0, 1, 2)

    def tab2_left_side_ctrl(self):
        self.tab2_labels = {}
        self.tab2_buttons = {}
        self.tab2_spinboxes = {}

        labels = {
           'no. upper': (0, 1, 1, 1),
           'no. lower': (0, 2, 1, 1),
           'linewidth': (0, 3, 1, 1),
           'Bessel order': (0, 4, 1, 1),
           'Vector 1': (1, 0, 1, 1),
           'Vector 2': (2, 0, 1, 1),
           'Layerline': (3, 0, 1, 1),
        } 

        buttons = {
            'Draw lattice': (5, 0, 1, 1),
            'Refine': (5, 1, 1, 1),
            'Show peaks': (5, 2, 1, 1), 
        }
        boxes = {
            'n_upper_v1': (1, 1, 1, 1, self.n_lines1, 1, 50),
            'n_lower_v1': (1, 2, 1, 1, self.n_lower_lines1, 0, 50),
            'lw_v1': (1, 3, 1, 1, self.v1_lattice_lw, 0, 3),
            'Bessel_v1': (1, 4, 1, 1, 3, 1, 200),
            'n_upper_v2': (2, 1, 1, 1, self.n_lines2, 1, 50),
            'n_lower_v2': (2, 2, 1, 1, self.n_lower_lines2, 0, 50),
            'lw_v2': (2, 3, 1, 1, self.v2_lattice_lw, 0, 3),
            'Bessel_v2': (2, 4, 1, 1, 1, 1, 200),
            'n_LL': (3, 1, 1, 1, self.n_tab2_LL, 0, 50),
            'lw_LL': (3, 3, 1, 1, self.tab2_LL_lw, 0, 3), 
        }

        for txt, pos in labels.items():
            self.tab2_labels[txt] = QLabel(txt)
            self.tab2_ctrl_left_layout.addWidget(self.tab2_labels[txt], pos[0], pos[1], pos[2], pos[3])

        for txt, pos in buttons.items():
            self.tab2_buttons[txt] = QPushButton(txt)    
            self.tab2_buttons[txt].setFixedWidth(95)
            self.tab2_buttons[txt].setEnabled(False)
            self.tab2_ctrl_left_layout.addWidget(self.tab2_buttons[txt], pos[0], pos[1], pos[2], pos[3])

        self.tab2_buttons['Draw lattice'].setToolTip('''Draw fourier space lattice. 
        Need to define layerlines in the other tab first.\n
        To adjust the two base vectors:
        V1: Command/Control + mouse click on a diffraction peak
        V2: Shift + mouse click on a diffraction peak''')

        self.tab2_buttons['Refine'].setToolTip('''Refine parameters.  Make sure:
        The two base vectors and layer lines are set and 
        The Bessel orders of the two base vecotrs are set.\n
        Adjust base vectors and repeat if:
        Lattice points do not match peaks in the power spectrum
        Residual after refinement (printed in terminal) is not zero.\n
        If successful, real space lattice will be shown in the right panel.
        If "pix~1/\u212B" set correctly, "rise" in real space will have correct scale in angstrom.\n''') 
        
        self.tab2_buttons['Show peaks'].setToolTip('''Show/hide peaks and their Bessel orders based on the refined lattice
        Useful for cross-check with the assignments in Tab1.''')        

        for txt, pos in boxes.items():
            if 'lw' in txt:
                self.tab2_spinboxes[txt] = QDoubleSpinBox()
                self.tab2_spinboxes[txt].setSingleStep(0.5)
            else:
                self.tab2_spinboxes[txt] = QSpinBox()
            self.tab2_spinboxes[txt].setFixedWidth(60)
            self.tab2_spinboxes[txt].setValue(pos[4])
            self.tab2_spinboxes[txt].setRange(pos[5], pos[6])
            self.tab2_ctrl_left_layout.addWidget(self.tab2_spinboxes[txt], pos[0], pos[1], pos[2], pos[3])

            
        self.tab2_spinboxes['Bessel_v1'].setToolTip('Absolute value of Bessel order of vector 1\n(Positive integer)')
        self.tab2_spinboxes['Bessel_v2'].setToolTip('Absolute value of Bessel order of vector 2\n(Positive integer)')

        self.tab2_symmetrize_fft_switch = QCheckBox('sym')
        self.tab2_symmetrize_fft_switch.setCheckable(False)
        self.tab2_ctrl_left_layout.addWidget(self.tab2_symmetrize_fft_switch, 0, 0, 1 ,1)
        self.tab2_symmetrize_fft_switch.setToolTip('Symmetrize the power spectrum.')

        divider1 = QFrame()
        divider1.setFrameShape(QFrame.HLine)
        divider1.setFrameShadow(QFrame.Sunken)
        self.tab2_ctrl_left_layout.addWidget(divider1, 4, 0, 1, 5)

    def tab3_left_side_ctrl(self):
        self.tab3_labels = {}
        self.tab3_buttons = {}
        self.tab3_text = {}

        labels ={
            'Rise (\u212B)': (0, 0, 1, 1),
            'Twist (\u00B0)': (1, 0, 1, 1),
            'Point group': (2, 0, 1, 1),
            'Tube diameter (\u212B)': (3, 0, 1, 1),
            'Subunit diameter (\u212B)': (4, 0, 1, 1),
            'Box dim (pixel)': (5, 0, 1, 1),
            'Pixel size (\u212B)': (6, 0, 1, 1),
        } 

        buttons = {
            'Autofill': (7, 0, 1, 1),
            'Draw 3D': (8, 0, 1, 1),
            'Relion command':(9, 0, 1, 1)
        }

        txt_fields = {
            'rise': (0, 1, 1, 1),
            'twist': (1, 1, 1, 1),
            'pg': (2, 1, 1, 1),
            'td': (3, 1, 1, 1),
            'sd': (4, 1, 1, 1),
            'bd': (5, 1, 1, 1),
            'ps': (6, 1, 1, 1),
            'rc': (10, 0, 1, 2),
        }

        for txt, pos in labels.items():
            self.tab3_labels[txt] = QLabel(txt)
            self.tab3_left_side.addWidget(self.tab3_labels[txt], pos[0], pos[1], pos[2], pos[3])

        for txt, pos in buttons.items():
            self.tab3_buttons[txt] = QPushButton(txt)
            self.tab3_buttons[txt].setFixedWidth(150)
            self.tab3_left_side.addWidget(self.tab3_buttons[txt], pos[0], pos[1], pos[2], pos[3])

        for txt, pos in txt_fields.items():
            if not txt == 'rc':
                self.tab3_text[txt] = QLineEdit()
                self.tab3_text[txt].setMaximumWidth(80)
                self.tab3_left_side.addWidget(self.tab3_text[txt], pos[0], pos[1], pos[2], pos[3])
            else:
                self.tab3_text[txt] = QTextEdit()
                self.tab3_left_side.addWidget(self.tab3_text[txt], pos[0], pos[1], pos[2], pos[3])

        spacer_item = QSpacerItem(100, 1, QSizePolicy.Minimum, QSizePolicy.Expanding)
        self.tab3_left_side.addItem(spacer_item, 11, 0, 1, 2)

    def tab3_3D_plot_window(self):
        self.fig_3d = plt.figure()
        self.ax_3d = self.fig_3d.add_subplot(projection='3d')
        self.fig_3d.suptitle('3D lattice', fontsize=10)
        self.fig_3d.tight_layout()
        self.ax_3d.view_init(elev=10, azim=-90)
        self.ax_3d.grid(False)
        self.ax_3d.set_xlabel('x (\u212B)')
        self.ax_3d.set_ylabel('y (\u212B)')
        self.ax_3d.set_zlabel('z (\u212B)')
        self.fig_3d_canvas = plt_qtbackend.FigureCanvasQTAgg(self.fig_3d)
        self.toolbar_3d = plt_qtbackend.NavigationToolbar2QT(self.fig_3d_canvas, self)
        self.toolbar_3d.setFixedHeight(35)

        self.tab3_right_side.addWidget(self.fig_3d_canvas)
        self.tab3_right_side.addWidget(self.toolbar_3d)

    def open_2D_classes(self):
        self.power_spec_only = 0
        twoD_img_file_name = QFileDialog.getOpenFileName(self, 'choose 2D image(s)', '.', 
                                                          filter="Image file (*.tiff *tif *.jpg *.png *jpeg *.mrc *.mrcs)")
        if twoD_img_file_name[0] == '':
            return
        try:
            if '.mrc' in twoD_img_file_name[0]:
                with mrcfile.open(twoD_img_file_name[0]) as f:
                    self.mrc_data_array = f.data
                    if self.mrc_data_array.ndim == 2:
                        self.mrc_data_array = self.mrc_data_array.reshape(1, self.mrc_data_array.shape[0], self.mrc_data_array.shape[1])
     
                    try:
                        y_pix = f.header.ny
                        y_dim = f.header.cella.y
                        self.angpix = y_dim/y_pix
                        if self.angpix == 0:
                            self.angpix = 1
                            QMessageBox.information(self, 'Alert', 'Could not read angpix\n Set to 1!')
                    except:
                        self.angpix = 1
                        QMessageBox.information(self, 'Alert', 'Could not read angpix\n Set to 1!')
            else:
                self.mrc_data_array = Image.open(twoD_img_file_name[0])
                self.mrc_data_array = ImageOps.flip(self.mrc_data_array)
                self.mrc_data_array = np.asarray(self.mrc_data_array.convert('L'))
                self.mrc_data_array = self.mrc_data_array.reshape(1, self.mrc_data_array.shape[0], self.mrc_data_array.shape[1])
                self.angpix = 1
                QMessageBox.information(self, 'Information', 'Reading non-MRC image\nAngpix set to 1\nManually set it if you know it')
        except:
            QMessageBox.information(self, 'Error', 'Not a valid mrc or mrcs file?')
            return

        self.img_xdim = self.mrc_data_array.shape[2]
        self.img_ydim = self.mrc_data_array.shape[1]
        self.origin[0] = self.img_xdim/2
        self.origin[1] = self.img_ydim/2
        self.reset_tab1_display()
        self.reset_tab2_display()
        self.setWindowTitle(f'PyHI: {twoD_img_file_name[0]}')

        self.calc_current_img_array()
        self.draw_tab2_fft()
        
        try:
            self.slice_chooser.valueChanged.disconnect()
        except:
            pass
        self.slice_chooser.valueChanged.connect(self.calc_current_img_array)     

    def load_power_spec(self):
        self.power_spec_only = 1
        power_spec_filename = QFileDialog.getOpenFileName(self, 'choose power spec image', '.', 
                                                          filter="Image file (*.tiff *.tif *.jpg *.png *.jpeg *.mrc *.mrcs)")
        if power_spec_filename[0] == '':
            return
        
        try:
            if '.mrc' in power_spec_filename[0]:
                with mrcfile.open(power_spec_filename[0], permissive=True) as f:
                    self.current_img_fft_amp = f.data
                    if self.current_img_fft_amp.ndim == 2:
                        self.current_img_fft_amp = self.current_img_fft_amp.reshape(1, self.current_img_fft_amp.shape[0], self.current_img_fft_amp.shape[1])
                    try:
                        y_pix = f.header.ny
                        y_dim = f.header.cella.y
                        self.angpix = y_dim/y_pix
                        if self.angpix == 0:
                            self.angpix = 1
                            QMessageBox.information(self, 'Alert', 'Could not read angpix\n Set to 1!')
                    except:
                        self.angpix = 1
                        QMessageBox.information(self, 'Alert', 'Could not read angpix\n Set to 1!')
     
            else:
                self.current_img_fft_amp = Image.open(power_spec_filename[0])
                self.current_img_fft_amp = ImageOps.flip(self.current_img_fft_amp)
                self.current_img_fft_amp = np.asarray(self.current_img_fft_amp.convert('L'))
                self.current_img_fft_amp = self.current_img_fft_amp.reshape(1, self.current_img_fft_amp.shape[0], self.current_img_fft_amp.shape[1])
                self.angpix = 1
                QMessageBox.information(self, 'Information', 'Reading non-MRC image\nAngpix set to 1\nManually set it if you know it')
        except:
            QMessageBox.information(self, 'Error', 'Not a valid image file?')
            return

        self.current_img_fft_phase = np.zeros_like(self.current_img_fft_amp)
        self.img_xdim = self.current_img_fft_amp.shape[2]
        self.img_ydim = self.current_img_fft_amp.shape[1]
        self.origin[0] = self.img_xdim/2
        self.origin[1] = self.img_ydim/2

        self.setWindowTitle(f'PyHI: {power_spec_filename[0]}')
        QMessageBox.information(self, 'Alert', 'Loaded power spectrum only\nNo phase information\nIgnore phase plot.')
        self.reset_tab1_display()
        self.reset_tab2_display()
        self.draw_tab1_fft()
        self.draw_tab2_fft()
        
        try:
            self.slice_chooser.valueChanged.disconnect()
        except:
            pass
        self.slice_chooser.valueChanged.connect(self.draw_tab1_fft)     

    def save_power_spec(self):
        name, ext = QFileDialog.getSaveFileName(self, 'Save MRC', '.', filter=".mrc")
        if name != '':
            if name[-4:] == '.mrc':
                name = name[:-4]
            name1 = name + ext
            x_ang = self.angpix*self.current_img_fft_amp_rotated.shape[1]
            y_ang = self.angpix*self.current_img_fft_amp_rotated.shape[0]
            with mrcfile.new(name1, overwrite=True) as f:
                f.set_data(self.current_img_fft_amp_rotated.astype('float32'))
                f.header.cella=(x_ang, y_ang, 0)

            name2 = name + '_symmetrized' + ext
            amp_average = self.symmetrize_fft(self.current_img_fft_amp_rotated) 
            with mrcfile.new(name2, overwrite=True) as f:
                f.set_data(amp_average.astype('float32'))
                f.header.cella=(x_ang, y_ang, 0)

    def symmetrize_fft(self, input_fft):
        amp_average =  (input_fft[1::1, 1::1] + input_fft[-1:0:-1, 1::1] + input_fft[1::1, -1:0:-1] + input_fft[-1:0:-1, -1:0:-1])/4
        amp_average = np.vstack((input_fft[0, 1::], amp_average))
        amp_average = np.hstack((input_fft[:, 0].reshape(-1, 1), amp_average))
        return amp_average

    def save_para(self):
        name, ext = QFileDialog.getSaveFileName(self, 'Save File', '.', filter=".txt")
        if name != '':
            if name[-4:] != '.txt':
                name = name + ext
            
            with open(name, 'w') as f:
                f.write(f'{"Angpix:":20s}{self.angpix:10.2f}\n\n')
                f.write(f'{"Origin X:":20s}{self.origin[0]:10.2f}\n')
                f.write(f'{"Origin Y:":20s}{self.origin[1]:10.2f}\n\n')
                f.write(f'{"Layerline distance:":20s}{self.LL_distance:10.2f}\n\n')

                f.write(f'Base vector 1:\n')
                f.write(f'{"X_coord 1:":20s}{self.x_v1:10.2f}\n')
                f.write(f'{"Y_coord 1:":20s}{self.y_v1:10.2f}\n')
                f.write(f'{"Bessel order 1:":20s}{self.v1_n:10d}\n')
                f.write(f'{"RS length:":20s}{self.length_v1_rs:10.2f}\n')
                f.write(f'{"RS angle:":20s}{self.angle1_rs:10.2f}\n\n')

                f.write(f'Base vector 2:\n')
                f.write(f'{"X_coord 2:":20s}{self.x_v2:10.2f}\n')
                f.write(f'{"Y_coord 2:":20s}{self.y_v2:10.2f}\n')
                f.write(f'{"Bessel order 2:":20s}{abs(self.v2_n):10d}\n')
                f.write(f'{"RS length:":20s}{self.length_v2_rs:10.2f}\n')
                f.write(f'{"RS angle:":20s}{self.angle2_rs:10.2f}\n\n')

                f.write(f'{"Helix radius:":20s}{self.radius_H:10.2f}\n')
                f.write(f'{"Rise/subunit:":20s}{self.rise_rs_main:10.2f}\n')
                f.write(f'{"Twist/subunit:":20s}{self.twist_rs_main:10.2f}\n')
                f.write(f'{"n-start:":20s}{self.n_start_main:10d}')

    def load_para(self):
        para_file_name = QFileDialog.getOpenFileName(self, 'choose Para file', '.', filter="Para file (*.txt)")
        if para_file_name[0] == '':
            return
        para_dict ={}
        with open(para_file_name[0],'r') as f:
            for line in f.readlines():
                if ':' in line:
                    name, para = line.split(':')
                    if para != '\n':
                        if 'Bessel' in name or 'n_start' in name:
                            para_dict[name] = int(para)
                        else:
                            para_dict[name] = float(para)
        try:
            self.origin[0] = para_dict['Origin X']         
            self.origin[1] = para_dict['Origin Y']         
            angpix = para_dict['Angpix']
            self.x_v1 = para_dict['X_coord 1']
            self.y_v1 = para_dict['Y_coord 1']
            self.x_v2 = para_dict['X_coord 2']
            self.y_v2 = para_dict['Y_coord 2']
            v1_n = para_dict['Bessel order 1']
            v2_n = para_dict['Bessel order 2']
            LL_distance = para_dict['Layerline distance']
            radius = para_dict['Helix radius']

            self.tab1_text_col1['Y_dist'].setText(f'{LL_distance:.2f}')
            self.tab1_text_col2['Angpix'].setText(f'{angpix:.2f}')
            self.set_angpix()
            self.tab1_text_col2['helix_radius'].setText(f'{radius:.2f}')
            self.check_LL_plots_inputs()
            self.draw_tab1_LL()

            self.tab2_spinboxes['Bessel_v1'].setValue(v1_n)
            self.tab2_spinboxes['Bessel_v2'].setValue(v2_n)
            self.draw_tab2_lattice_lines()
            self.opt_para()

        except:
            QMessageBox.information(self, 'Error', 'Check parameter file format!')                

    def reset_tab1_display(self):
        self.tab1_text_col2['Angpix'].setText(f'{self.angpix:3.2f}')
        for buttons in self.tab1_buttons_col1.values():
            buttons.setEnabled(True)
        for buttons in self.tab1_buttons_col2.values():
            buttons.setEnabled(True)
        if self.measure_on == True:
            self.measure_distance()
        self.tab1_text_col2['LL_Y_range'].setText('')
        self.tab1_text_col2['LL_width'].setText('')
        self.tab1_text_col2['Bessel_order'].setText('')
        self.tab1_text_col2['helix_radius'].setText('')
        self.tab1_labels_col2['Measure'].setText('')
        self.tab1_text_col2['radius_error'].setText('')
        self.clear_tab1_LL()
        if self.twoD_img_shown is not None:
            self.twoD_img_shown.remove()
            self.twoD_img_shown = None
            self.midpoint_of_2dimage.remove()
            self.midpoint_of_2dimage = None
            self.oneD_profile_of_2dimage.remove()
            self.oneD_profile_of_2dimage = None
            self.fig2d_canvas.draw()
        if self.tab1_fft_shown is not None:
            self.tab1_fft_shown.remove()
            self.tab1_fft_shown = None
            self.tab1_figfft_canvas.draw()
        if self.LL_amp_plot != '' and self.LL_phase_plot != '':
            self.LL_amp_plot.remove()
            self.LL_amp_plot = ''
            self.LL_bessel_plot.remove()
            self.LL_bessel_plot = ''
            self.LL_bessel_plot_fill_low.remove()
            self.LL_bessel_plot_fill_low = ''
            self.LL_bessel_plot_fill_high.remove()
            self.LL_bessel_plot_fill_high = ''
            self.LL_legend.remove()
            self.LL_legend = ''
            self.LL_phase_plot.remove()
            self.LL_phase_plot = ''
            self.LL_phase_diff_ind.remove()
            self.LL_phase_diff_ind = ''
            self.fig_bessel_canvas.draw()

        self.minSigma_chooser.blockSignals(True)
        self.maxSigma_chooser.blockSignals(True)
        self.minSigma_chooser.setEnabled(True)
        self.maxSigma_chooser.setEnabled(True)
        self.minSigma_chooser.setValue(-1)
        self.maxSigma_chooser.setValue(8)
        self.minSigma_chooser.blockSignals(False)
        self.maxSigma_chooser.blockSignals(False)

        self.tab2_symmetrize_fft_switch.blockSignals(True)
        self.tab2_symmetrize_fft_switch.setCheckable(True)
        self.tab2_symmetrize_fft_switch.setChecked(False)
        self.tab2_symmetrize_fft_switch.blockSignals(False)

        self.slice_chooser.blockSignals(True)
        self.img_shift_chooser.blockSignals(True)
        self.img_rotation_chooser.blockSignals(True)
        self.auto_align_toggle.blockSignals(True)
        self.twoD_inv_toggle.blockSignals(True)
        self.ps_cmap_toggle.blockSignals(True)

        self.slice_chooser.setValue(1)
        self.img_rotation_chooser.setValue(0)
        self.img_rotation_chooser.setEnabled(True)
        self.img_shift_chooser.setValue(0)
        self.auto_align_toggle.setChecked(False)
        self.twoD_inv_toggle.setChecked(False)
        self.ps_cmap_toggle.setCheckable(True)
        self.ps_cmap_toggle.setChecked(False)

        self.slice_chooser.blockSignals(False)
        self.img_shift_chooser.blockSignals(False)
        self.img_rotation_chooser.blockSignals(False)
        self.auto_align_toggle.blockSignals(False)
        self.twoD_inv_toggle.blockSignals(False)
        self.ps_cmap_toggle.blockSignals(False)

        self.slice_chooser.setEnabled(True)
        if self.power_spec_only == 0:
            self.auto_align_toggle.setCheckable(True)
            self.twoD_inv_toggle.setCheckable(True)
            self.img_shift_chooser.setEnabled(True)
            self.slice_chooser.setMaximum(self.mrc_data_array.shape[0])
            self.img_shift_chooser.setRange(-self.img_xdim, self.img_xdim)
        elif self.power_spec_only == 1:
            self.auto_align_toggle.setCheckable(False)
            self.twoD_inv_toggle.setCheckable(False)
            self.img_shift_chooser.setEnabled(False)
            self.slice_chooser.setMaximum(self.current_img_fft_amp.shape[0])
            self.tab1_buttons_col2['Measure'].setEnabled(False)
    
    def reset_tab2_display(self):
        self.tab2_spinboxes['n_upper_v1'].setValue(2)
        self.tab2_spinboxes['n_upper_v2'].setValue(2)
        self.tab2_spinboxes['n_lower_v1'].setValue(0)
        self.tab2_spinboxes['n_lower_v2'].setValue(0)
        self.tab2_spinboxes['n_LL'].setValue(30)
        self.tab2_spinboxes['lw_v1'].setValue(0.5)
        self.tab2_spinboxes['lw_v2'].setValue(0.5)
        self.tab2_spinboxes['lw_LL'].setValue(0.5)
        self.tab2_spinboxes['Bessel_v1'].setValue(3)
        self.tab2_spinboxes['Bessel_v2'].setValue(1)
        if self.tab2_fft_shown != None:
            self.tab2_fft_shown.remove()   
            self.tab2_fft_shown = None
        self.undo_refine()
        if "Clear lattice" in self.tab2_buttons['Draw lattice'].text():
            self.tab2_lattice_lines_toggle()

        self.twist_rs_main = 360
        self.rise_rs_main = 0
        self.circum_rs = 0
        self.n_start_main = 1
        
    def calc_current_img_array(self):
        slice_chosen = self.slice_chooser.value() - 1
        self.current_img_array = self.mrc_data_array[slice_chosen]

        if self.twoD_inv_toggle.isChecked():
            self.current_img_array = -self.current_img_array

        if self.auto_align_toggle.isChecked():
            self.reset_rotation_shift()
            try:
                self.current_img_array = self.auto_align(self.current_img_array)
            except:
                print('Alignment failed. Empty image?')
            self.draw_2d_image(self.current_img_array)
        else:
            if self.img_rotation_chooser.value() != 0 or self.img_shift_chooser.value() != 0:
                self.set_img_rotation_shift()
            else:
                self.draw_2d_image(self.current_img_array)
            
    def draw_2d_image(self, img_array): 
        min = img_array.min()
        max = img_array.max()

        if self.twoD_img_shown == None:
            self.twoD_img_shown = self.ax2d.imshow(img_array, origin='lower', cmap='Greys_r')
            self.ax2d.set_xlim(0, self.img_xdim-1)
            self.ax2d.set_ylim(0, self.img_ydim-1)
            self.toolbar2d.update()
        else:
            self.twoD_img_shown.set_data(img_array)

        if self.midpoint_of_2dimage == None:
            self.midpoint_of_2dimage=(self.ax2d.plot(
                [self.img_xdim/2-0.5, self.img_xdim/2-0.5], [0, self.img_ydim-1], '-', color='b', lw=0.5))[0]
        else:
            self.midpoint_of_2dimage.set_data([self.img_xdim/2-0.5, self.img_xdim/2-0.5], [0, self.img_ydim-1])

        oneD_profile_array = np.sum(img_array, axis=0) 
        oneD_profile_array = oneD_profile_array - oneD_profile_array.min()   
        oneD_profile_array = oneD_profile_array*(0.1*self.img_ydim/oneD_profile_array.max())
        if self.oneD_profile_of_2dimage == None:
            self.oneD_profile_of_2dimage =(self.ax2d.plot(oneD_profile_array, '-', color='cyan', lw=0.5))[0]
        else:
            self.oneD_profile_of_2dimage.set_data(np.arange(self.img_xdim), oneD_profile_array)

        self.twoD_img_shown.set_clim(min, max)
        self.fig2d_canvas.draw()

        self.calculate_fft(img_array)

    def raw_moments(self, intensity, i_order, j_order):
        nrows, ncols = intensity.shape
        y_indices, x_indices =  np.mgrid[:nrows, :ncols]
        return (intensity*x_indices**i_order*y_indices**j_order).sum()
    
    def cov_xc_yc(self, intensity):
        intensity = intensity - intensity.min()
        intensity_sum = intensity.sum()
        m10 = self.raw_moments(intensity, 1, 0)
        m01 = self.raw_moments(intensity, 0, 1)
        x_c = m10/intensity_sum
        y_c = m01/intensity_sum 
        u11 = (self.raw_moments(intensity, 1, 1) - x_c*m01)/intensity_sum
        u20 = (self.raw_moments(intensity, 2, 0) - x_c*m10)/intensity_sum
        u02 = (self.raw_moments(intensity, 0, 2) - y_c*m01)/intensity_sum
        cov = np.array([[u20, u11], [u11, u02]])
        return (cov, x_c, y_c)

    def auto_align(self, intensity):
        cov, x_centroid, y_centroid = self.cov_xc_yc(intensity)
        evals, evecs = np.linalg.eig(cov)
        sort_indices = np.argsort(evals)[::-1]
        x_v1, y_v1 = evecs[:, sort_indices[0]]
        angle = np.degrees(np.arctan(y_v1/x_v1))
        rotate = angle - 90
        shift_y = self.img_ydim/2 - y_centroid
        shift_x = self.img_xdim/2 - x_centroid
        intensity = ndimage.shift(intensity, [shift_y, shift_x])
        intensity = ndimage.rotate(intensity, rotate, reshape=False, mode='constant')
        return intensity

    def auto_align_toggle_signal(self):
        self.reset_rotation_shift()
        self.calc_current_img_array()
    
    def reset_rotation_shift(self):
        self.img_rotation_chooser.blockSignals(True)
        self.img_rotation_chooser.setValue(0)
        self.img_shift_chooser.blockSignals(True)
        self.img_shift_chooser.setValue(0)
        self.img_rotation_chooser.blockSignals(False)
        self.img_shift_chooser.blockSignals(False)
    
    def set_img_rotation_shift(self):
        img_rotation = self.img_rotation_chooser.value() 
        img_shift = self.img_shift_chooser.value()
        if self.power_spec_only == 0:
            if self.mrc_data_array is not None:
                img_array = self.current_img_array
                if img_rotation != 0:
                    img_array = ndimage.rotate(img_array, img_rotation, reshape=False, mode='constant')
                if img_shift != 0:
                    img_array = ndimage.shift(img_array, [0, img_shift])
            self.draw_2d_image(img_array)
        else:
            self.draw_tab1_fft()

    def calculate_fft(self, img_array):
        current_img_fft = np.fft.fftshift(np.fft.fft2(img_array))
        self.current_img_fft_amp = np.abs(current_img_fft)
        self.current_img_fft_phase = np.angle(current_img_fft)
        self.draw_tab1_fft()    
    
    def draw_tab1_fft(self):
        if self.power_spec_only == 1:
            img_rotation = self.img_rotation_chooser.value()
            slice_chosen = self.slice_chooser.value() - 1
            current_img_fft_amp = self.current_img_fft_amp[slice_chosen]
            self.current_img_fft_amp_rotated = ndimage.rotate(current_img_fft_amp, img_rotation, reshape=False)
        else:
            self.current_img_fft_amp_rotated = self.current_img_fft_amp
        if self.tab1_fft_shown is None:
            self.tab1_fft_shown = self.tab1_axfft.imshow(self.current_img_fft_amp_rotated, origin='lower')
            self.tab1_axfft.set_xlim(0, self.img_xdim-1)
            self.tab1_axfft.set_ylim(0, self.img_ydim-1)
            self.tab1_fft_toolbar.update()
        else:
            self.tab1_fft_shown.set_data(self.current_img_fft_amp_rotated)

        if self.ps_cmap_toggle.isChecked():
            self.tab1_fft_shown.set_cmap('Greys')
        else:
            self.tab1_fft_shown.set_cmap('Greys_r')
        
        contrast_high = self.maxSigma_chooser.value()
        contrast_low = self.minSigma_chooser.value()
        high = self.current_img_fft_amp_rotated.mean() + contrast_high*self.current_img_fft_amp_rotated.std()
        low = self.current_img_fft_amp_rotated.mean() + contrast_low*self.current_img_fft_amp_rotated.std()
        self.tab1_fft_shown.set_clim(vmin=low, vmax=high)
        self.tab1_figfft_canvas.draw()
        self.new_mrcs_status = 0

        self.fftSaveButton.setEnabled(True)
        self.paramLoadButton.setEnabled(True)

        self.check_LL_plots_inputs()

    def draw_tab2_fft(self):
        try:
            if self.tab2_symmetrize_fft_switch.isChecked():
                fft_for_tab2 = self.symmetrize_fft(self.current_img_fft_amp_rotated)
            else:
                fft_for_tab2 = self.current_img_fft_amp_rotated
            
            if self.tab2_fft_shown is None: 
                self.tab2_fft_shown = self.tab2_axfft.imshow(fft_for_tab2, origin='lower')
                self.tab2_axfft.set_xlim(0, self.img_xdim-1)
                self.tab2_axfft.set_ylim(0, self.img_ydim-1)
                self.tab2_fft_toolbar.update()
            else:
                self.tab2_fft_shown.set_data(fft_for_tab2)

            if self.ps_cmap_toggle.isChecked():
                self.tab2_fft_shown.set_cmap('Greys')
            else:
                self.tab2_fft_shown.set_cmap('Greys_r')

            contrast_high = self.maxSigma_chooser.value()
            contrast_low = self.minSigma_chooser.value()
            high = fft_for_tab2.mean() + contrast_high*fft_for_tab2.std()
            low = fft_for_tab2.mean() + contrast_low*fft_for_tab2.std()
            self.tab2_fft_shown.set_clim(vmin=low, vmax=high)
            self.tab2_figfft_canvas.draw()
        except:
            return

    def set_origin_by_click(self, event):
        mod = QGuiApplication.keyboardModifiers()
        if mod == QtCore.Qt.AltModifier:
            self.origin[0] = event.xdata
            self.origin[1] = event.ydata
            self.draw_tab1_LL()        

    def set_vectors_by_click(self, event):
        mod = QGuiApplication.keyboardModifiers()
        if mod == QtCore.Qt.ControlModifier: 
            y = round((event.ydata - self.origin[1])/self.LL_distance)*self.LL_distance
            if y <= 0:
                QMessageBox.information(self, 'Error', 'Vector must end at a layerline!')
                return
            x = event.xdata - self.origin[0]
            _, ang = self.calc_vector_length_angle(x, y)
            _, ang2 = self.calc_vector_length_angle(self.x_v2, self.y_v2)
            if (not 0 < ang < ang2) or (ang >= 90):
                QMessageBox.information(self, 'Error', 'Angel 1 must be in range of 0-90 and smaller than Angle 2')
                return
            self.x_v1 = x 
            self.y_v1 = y
            self.draw_tab2_lattice_lines()
        elif mod == QtCore.Qt.ShiftModifier: 
            y = round((event.ydata - self.origin[1])/self.LL_distance)*self.LL_distance
            if y <= 0:
                QMessageBox.information(self, 'Error', 'Vector must end at a layerline!')
                return
            x = event.xdata - self.origin[0]
            _, ang = self.calc_vector_length_angle(x, y)
            _, ang1 = self.calc_vector_length_angle(self.x_v1, self.y_v1)
            if (not ang1 < ang < 180):
                QMessageBox.information(self, 'Error', 'Angel 2 must be larger than Angle 1 and smaller than 180')
                return
            self.x_v2 = x
            self.y_v2 = y
            self.draw_tab2_lattice_lines()

    def measure_distance(self):
        if self.angpix == 0:
            QMessageBox.information(self,'Error', 'Angpix not known!')
            return
        if self.measure_on == False:
            self.tab1_buttons_col2['Measure'].setText('Stop Meas')
            self.points_x = []
            self.points_y = []
            self.cid_measure_2D = self.fig2d_canvas.mpl_connect('button_press_event', self.click_draw_measure)
            self.measure_on = True
        else:
            self.tab1_buttons_col2['Measure'].setText('Measure')
            self.fig2d_canvas.mpl_disconnect(self.cid_measure_2D)
            self.measure_on = False
            if self.dist_line_in_2D != '':
                self.dist_line_in_2D.remove()
                self.dist_line_in_2D = ''
            self.fig2d_canvas.draw()
    
    def click_draw_measure(self, event):
        self.points_x.append(event.xdata)
        self.points_y.append(event.ydata)
        if len(self.points_x) == 2:
            self.ruler_width = abs(self.points_x[0] - self.points_x[1])*self.angpix
            self.ruler_height = abs(self.points_y[0] - self.points_y[1])*self.angpix
            length = math.sqrt(self.ruler_width**2 + self.ruler_height**2)
            self.set_measure_txt(self.ruler_width, self.ruler_height, length)

            if self.dist_line_in_2D == '':
                self.dist_line_in_2D, = self.ax2d.plot(self.points_x, self.points_y, 'o-', color='r', linewidth=1, markersize=3)
            else:
                self.dist_line_in_2D.set_data(self.points_x, self.points_y)
            self.fig2d_canvas.draw()
            self.points_x = []
            self.points_y = []

    def set_measure_txt(self, w, h, l):
            txt = f'w={w:.1f}; h={h:.1f}; l={l:.1f} (\u212B)'
            self.tab1_labels_col2['Measure'].setText(txt)
            radius=w/2
            self.tab1_text_col2['helix_radius'].setText(f'{radius:.2f}')
    
    def toggle_draw_tab1_LL(self):
        if self.tab1_LL_draw_on == False:
            self.draw_tab1_LL()
        else:
            self.clear_tab1_LL()
            self.tab1_buttons_col1['Draw LL'].setText('Draw LL')
            self.tab1_LL_draw_on = False

    def draw_tab1_LL(self):
        self.clear_tab1_LL()
        try: 
            self.LL_distance = float(self.tab1_text_col1['Y_dist'].text())
            x_label = 2*self.img_xdim/5

            for i in range(1, 1+int(self.img_ydim/(self.LL_distance*2))):
                y = self.origin[1] + i*self.LL_distance
                self.tab1_LL.append(self.tab1_axfft.axline((self.origin[0], y), slope=0, ls=':', color='orange', linewidth=0.5))
                if i%5 == 0:
                    self.tab1_LL_label.append(self.tab1_axfft.annotate(str(i), (x_label, y), color='orange'))
            
            self.tab1_LL.append(self.tab1_axfft.axline((self.origin[0], self.origin[1]), slope=0, ls='-', color='orange', linewidth=0.5))
            self.tab1_LL.append(self.tab1_axfft.axline((self.origin[0], self.origin[1]), slope=math.inf, ls='-', color='orange', linewidth=0.5))

            repeat_dist = (self.current_img_fft_amp.shape[0]/self.LL_distance)*self.angpix
            self.tab1_labels_col1['Repeat distance:'].setText(f'Repeat distance: {repeat_dist:4.1f} \u212B')
            self.tab1_figfft_canvas.draw()
            self.tab1_LL_draw_on = True
            self.tab1_buttons_col1['Draw LL'].setText('Clear LL')

            self.tab1_cursor_annotate()

            self.tab2_buttons['Draw lattice'].setEnabled(True)
            self.tab2_buttons['Refine'].setEnabled(True)

        except:
           QMessageBox.information(self, 'Error', 'Need to load image and\nset sensible LL distance first!')

    def clear_tab1_LL(self):
        if len(self.tab1_LL) != 0:
            for lines in self.tab1_LL:
                lines.remove()
        if len(self.tab1_LL_label) !=0:
            for LL_label in self.tab1_LL_label:
                LL_label.remove()
        self.tab1_LL = []
        self.tab1_LL_label = []
        self.tab1_figfft_canvas.draw()
        self.tab1_labels_col1['Repeat distance:'].setText('Repeat distance:')

    def tab1_cursor_annotate(self):
       self.tab1_fft_cursor = mplcursors.cursor(self.tab1_LL[:-2], multiple=True, bindings={'select':2})
       @self.tab1_fft_cursor.connect('add')
       def _(sel):
           if sel.target[0] < self.current_img_fft_amp.shape[1]/2:
               n_Bessel = ''.join(("-", self.tab1_text_col2['Bessel_order'].text()))
               pos_x = sel.target[0] - 15
               pos_y = sel.target[1] + 1
           else:
               n_Bessel = self.tab1_text_col2['Bessel_order'].text()
               pos_x = sel.target[0] + 10 
               pos_y = sel.target[1] + 1
           sel.annotation.set(text=f'l={self.tab1_LL.index(sel.artist)+1}; n={n_Bessel}', position=[pos_x,pos_y])
           sel.annotation.arrow_patch.set(arrowstyle='simple',fc='yellow', alpha=0.5)

    def get_tab2_line_parameter_box_values(self):
        self.n_lines1 = self.tab2_spinboxes['n_upper_v1'].value()
        self.n_lines2 = self.tab2_spinboxes['n_upper_v2'].value()
        self.n_lower_lines1 =  self.tab2_spinboxes['n_lower_v1'].value()
        self.n_lower_lines2 = self.tab2_spinboxes['n_lower_v2'].value()
        self.n_tab2_LL = self.tab2_spinboxes['n_LL'].value()
        self.v1_lattice_lw = self.tab2_spinboxes['lw_v1'].value()
        self.v2_lattice_lw = self.tab2_spinboxes['lw_v2'].value()
        self.tab2_LL_lw = self.tab2_spinboxes['lw_LL'].value()

    def draw_tab2_lattice_lines(self):
        self.clear_tab2_lattice_lines()
        self.tab2_buttons['Refine'].setText('Refine')
        self.get_tab2_line_parameter_box_values()

        self.y_v1 = round(self.y_v1/self.LL_distance)*self.LL_distance
        if self.y_v1 == 0:
            self.y_v1 = self.LL_distance
        self.y_v2 = round(self.y_v2/self.LL_distance)*self.LL_distance
        if self.y_v2 == 0:
            self.y_v2 = self.LL_distance

        self.tab2_lattice_lines.append(self.tab2_axfft.axline((self.origin[0], self.origin[1]), slope=0, color='orange', linewidth=self.tab2_fft_ax_lw))     
        self.tab2_lattice_lines.append(self.tab2_axfft.axline((self.origin[0], self.origin[1]), slope=math.inf, color='orange', linewidth=self.tab2_fft_ax_lw))
    
        for i in range(-self.n_lower_lines1, self.n_lines1):
            self.tab2_lattice_lines.append(self.tab2_axfft.axline((-i*self.x_v2+self.origin[0], i*self.y_v2+self.origin[1]), slope=-self.y_v1/self.x_v1, color='c', linewidth=self.v1_lattice_lw))
        for i in range(-self.n_lower_lines2, self.n_lines2):
            self.tab2_lattice_lines.append(self.tab2_axfft.axline((-i*self.x_v1+self.origin[0], i*self.y_v1+self.origin[1]), slope=-self.y_v2/self.x_v2, color='c', linewidth=self.v2_lattice_lw))
        for i in range(-self.n_lower_lines1, self.n_lines1):
            self.tab2_lattice_lines.append(self.tab2_axfft.axline((i*self.x_v2+self.origin[0], i*self.y_v2+self.origin[1]), slope=self.y_v1/self.x_v1, color='r', linewidth=self.v1_lattice_lw))
        for i in range(-self.n_lower_lines2, self.n_lines2):
            self.tab2_lattice_lines.append(self.tab2_axfft.axline((i*self.x_v1+self.origin[0], i*self.y_v1+self.origin[1]), slope=self.y_v2/self.x_v2, color='r', linewidth=self.v2_lattice_lw))

        base_vector_arrow1 = self.tab2_axfft.arrow(self.origin[0], self.origin[1], self.x_v1,self.y_v1, length_includes_head=True, 
                                           head_width= self.v1_lattice_lw*1.5, lw=self.v1_lattice_lw*2, color='m', overhang=0.3)
        base_vector_arrow2 = self.tab2_axfft.arrow(self.origin[0], self.origin[1], self.x_v2,self.y_v2, length_includes_head=True, 
                                           head_width= self.v2_lattice_lw*1.5, lw=self.v2_lattice_lw*2, color='m', overhang=0.3)
        self.tab2_lattice_lines.append(base_vector_arrow1)
        self.tab2_lattice_lines.append(base_vector_arrow2)
        self.tab2_line_labels.append(self.tab2_axfft.annotate('v1', (self.origin[0]+self.x_v1/2, self.origin[1]+self.y_v1/2), color='m', size=10))
        self.tab2_line_labels.append(self.tab2_axfft.annotate('v2', (self.origin[0]+self.x_v2/2, self.origin[1]+self.y_v2/2), color='m', size=10))
        
        x_label = 2*self.current_img_fft_amp_rotated.shape[1]/5
        for i in range(1, self.n_tab2_LL+1):
            y = i*self.LL_distance+self.origin[1]
            if y <= self.current_img_fft_amp_rotated.shape[0]:
                self.tab2_LL.append(self.tab2_axfft.axline((self.origin[0], y), slope=0, linestyle=':', color='orange', linewidth=self.tab2_LL_lw))
                if i%5 == 0:
                    self.tab2_line_labels.append(self.tab2_axfft.annotate(f'l={i}', (x_label, y), color='orange', size=10))

        self.tab2_figfft_canvas.draw()
        self.tab2_buttons['Draw lattice'].setText("Clear lattice")

    def clear_tab2_lattice_lines(self):
        if len(self.tab2_lattice_lines) != 0:
            for lines in self.tab2_lattice_lines:
                lines.remove()
        self.tab2_lattice_lines = []
        if len(self.tab2_LL) != 0:
            for lines in self.tab2_LL:
                lines.remove()
        self.tab2_LL = []
        if len(self.tab2_line_labels) != 0:
            for label in self.tab2_line_labels:
                label.remove()
        self.tab2_line_labels = []
        self.tab2_figfft_canvas.draw()            
        self.tab2_buttons['Draw lattice'].setText("Draw lattice")
        
    def tab2_lattice_lines_toggle(self):
        if self.tab2_buttons['Draw lattice'].text() == 'Draw lattice':
            self.draw_tab2_lattice_lines()
        else:
            self.clear_tab2_lattice_lines()

        if "Hide peaks" in self.tab2_buttons['Show peaks'].text():
            self.tab2_fft_peaks_toggle()
        self.tab2_buttons['Show peaks'].setEnabled(False)

    def set_angpix(self):
        try:
            angpix_old = self.angpix
            self.angpix = float(self.tab1_text_col2['Angpix'].text())
            if angpix_old !=self.angpix:
                if self.ruler_height != 0 or self.ruler_height != 0:
                    self.ruler_width = self.ruler_width*self.angpix/angpix_old
                    self.ruler_height = self.ruler_height*self.angpix/angpix_old
                    length = math.sqrt(self.ruler_width**2 + self.ruler_height**2)
                    self.set_measure_txt(self.ruler_width, self.ruler_height, length)
                    self.check_LL_plots_inputs()
        except:
            QMessageBox.information(self,'Error', 'Check your input!')

    def change_contrast(self, id, n):
        if id == 'low':
            contrast_high = self.maxSigma_chooser.value() 
            contrast_low = n
            if contrast_high <= contrast_low:
                contrast_high = int(contrast_low + 1)
                self.maxSigma_chooser.setValue(contrast_high)
        elif id == 'high':
            contrast_low = self.minSigma_chooser.value()
            contrast_high = n
            if contrast_high <= contrast_low:
                contrast_low = int(contrast_high - 1)
                self.minSigma_chooser.setValue(contrast_low)
        min = round(self.current_img_fft_amp_rotated.mean() + contrast_low*self.current_img_fft_amp_rotated.std(), 1)
        max = round(self.current_img_fft_amp_rotated.mean() + contrast_high*self.current_img_fft_amp_rotated.std(), 1)
        self.tab1_fft_shown.set_clim(min, max)
        self.tab1_figfft_canvas.draw()
    
    def calc_LL_plot(self):
        try:
            self.n_Bessel = int(self.tab1_text_col2['Bessel_order'].text())
            Y = self.tab1_text_col2['LL_Y_range'].text().split(',')
            self.radius_H = float(self.tab1_text_col2['helix_radius'].text())
            try:
                self.radius_error = float(self.tab1_text_col2['radius_error'].text())
            except:
                self.radius_error = 0
            if self.radius_H <= 0:
                QMessageBox.information(self, 'Alert', 'To calculate layerline plot, \nHelix radius must be larger than 0!')
                return
            if self.radius_H <= self.radius_error:
                QMessageBox.information(self, 'Alert', 'Radius error must be smaller than radius\nTry ~10% of radius')
                return
                
            Y1 = int(Y[0]) + int(self.origin[1])
            Y2 = int(Y[1]) + int(self.origin[1])
            if Y1>Y2:
                Y1, Y2 = Y2, Y1
            Y_phase = round((Y1 + Y2)/2)
            self.X_off_center = round(int(self.tab1_text_col2['LL_width'].text())/2)
            X1 = int(self.origin[0] - self.X_off_center) 
            X2 = int(self.origin[0] + self.X_off_center) 
            self.X_index = np.arange(-self.X_off_center, self.X_off_center+1)
    
            amp_data = self.current_img_fft_amp_rotated[Y1:Y2+1, X1:X2+1]
            amp_data = np.average(amp_data, axis=0)
            if self.ps_cmap_toggle.isChecked():
                amp_data = -amp_data
            amp_data = amp_data - amp_data.min()

            if self.angpix == 0:
                QMessageBox.information(self,'Alert', 'Angpix = 0')
                Bessel_data = np.zeros_like(amp_data)
            else:
                X_index_oversample = np.linspace(self.X_index[0],self.X_index[-1],5*len(self.X_index))
                k = 2*math.pi*self.radius_H*(self.X_index/(self.current_img_fft_amp_rotated.shape[1]*self.angpix))
                ko = 2*math.pi*self.radius_H*(X_index_oversample/(self.current_img_fft_amp_rotated.shape[1]*self.angpix))
                ks = 2*math.pi*(self.radius_H-self.radius_error)*(X_index_oversample/(self.current_img_fft_amp_rotated.shape[1]*self.angpix))
                kb = 2*math.pi*(self.radius_H+self.radius_error)*(X_index_oversample/(self.current_img_fft_amp_rotated.shape[1]*self.angpix))
                Bessel_data = np.abs(special.jv(self.n_Bessel, k))
                Bessel_data_o = np.abs(special.jv(self.n_Bessel, ko))
                Bessel_data_l = np.abs(special.jv(self.n_Bessel, ks))
                Bessel_data_h = np.abs(special.jv(self.n_Bessel, kb))
                amp_data_max = amp_data.max()
                scale_o = amp_data_max/Bessel_data_o.max()
                scale_l = amp_data_max/Bessel_data_l.max()
                scale_h = amp_data_max/Bessel_data_h.max()
                Bessel_data_o = Bessel_data_o*scale_o
                Bessel_data_l = Bessel_data_l*scale_l
                Bessel_data_h = Bessel_data_h*scale_h
            cc_Bessel_vs_data = (np.corrcoef(Bessel_data, amp_data))[0,1]
            self.tab1_labels_col2['CC='].setText(f'CC={cc_Bessel_vs_data:.2f}')
    
            if self.LL_amp_plot == '':
                self.LL_amp_plot, = self.ax_amp.plot(self.X_index, amp_data, color='b')
                self.LL_bessel_plot, = self.ax_amp.plot(X_index_oversample, Bessel_data_o, color='r')
                self.LL_legend = self.ax_amp.legend((self.LL_amp_plot, self.LL_bessel_plot), ('Data', 'Predict'), loc='upper right', fontsize=7, handlelength=1)
            else:
                self.LL_amp_plot.set_data(self.X_index, amp_data)
                self.LL_bessel_plot.set_data(X_index_oversample, Bessel_data_o)
            if self.LL_bessel_plot_fill_low != '':
                self.LL_bessel_plot_fill_low.remove()
                self.LL_bessel_plot_fill_low = ''
                self.LL_bessel_plot_fill_high.remove()
                self.LL_bessel_plot_fill_high = ''
            self.LL_bessel_plot_fill_low = self.ax_amp.fill_between(
                X_index_oversample, Bessel_data_l, Bessel_data_o, alpha=0.1, color='r', interpolate=True)
            self.LL_bessel_plot_fill_high = self.ax_amp.fill_between(
                X_index_oversample, Bessel_data_o, Bessel_data_h, alpha=0.1, color='r', interpolate=True)
            self.ax_amp.set_xlim(self.X_index[0], self.X_index[-1])
            self.ax_amp.set_ylim(0, amp_data.max()*1.15)
            
            X_original_index = self.X_index + int(self.current_img_fft_amp_rotated.shape[1]/2)
            phase_data = self.current_img_fft_phase[Y_phase, X_original_index[0]:X_original_index[-1]+1]
            phase_data = 180*phase_data/np.pi
            print('********************************')
            print('Phases of the plotted pixels:')
            print(f'{phase_data}\n')
            phase_diff = np.abs(phase_data - phase_data[::-1])
            phase_diff = [i if i <= 180 else 360 - i for i in phase_diff]

            if self.LL_phase_plot == '':
                self.LL_phase_plot, = self.ax_phase.plot(self.X_index, phase_diff, '.-', color='b')
            else:
                self.LL_phase_plot.set_data(self.X_index, phase_diff)

            Bessel_max_index = len(Bessel_data)//2 - Bessel_data.argmax()
            Bessel_max_pair = [Bessel_max_index, -Bessel_max_index]
            if self.LL_phase_diff_ind == '':
                if self.n_Bessel%2 == 0:
                    self.LL_phase_diff_ind, = self.ax_phase.plot(Bessel_max_pair, [0,0], '.', color='r', markersize=10)
                else:
                    self.LL_phase_diff_ind, = self.ax_phase.plot(Bessel_max_pair, [180,180], '.', color='r', markersize=10)
            else:
                if self.n_Bessel%2 == 0:
                    self.LL_phase_diff_ind.set_data(Bessel_max_pair, [0,0])
                else:
                    self.LL_phase_diff_ind.set_data(Bessel_max_pair, [180,180])

            self.ax_phase.set_xlim(self.X_index[0], self.X_index[-1])

            self.toolbar_bessel.update()
            self.fig_bessel_canvas.draw()

            if len(self.tab1_LL) >= 1:
                for LL in self.tab1_LL:
                    LL.set_color('orange')
                Y_mean = (Y1 + Y2)/2
                for i in range(len(self.tab1_LL)):
                    LL_y = self.tab1_LL[i].get_data()[1][0]
                    if abs(LL_y - Y_mean) <= self.LL_distance/2:
                        self.tab1_LL[i].set_color('w')
                        self.tab1_figfft_canvas.draw()
                        break

        except:
            QMessageBox.information(self,'Error', 'For layerline plots, \nHelix radius and layerline parameters need to be set correctly.')

    def check_LL_plots_inputs(self):
        if (self.tab1_text_col2['LL_Y_range'].text() != '' and
            self.tab1_text_col2['LL_width'].text() != '' and 
            self.tab1_text_col2['Bessel_order'].text() != '' and
            self.tab1_text_col2['helix_radius'].text() != ''):
            self.calc_LL_plot()
    
    def calculate_rs_para(self):
        length_v1, angle_v1 = self.calc_vector_length_angle(self.x_v1, self.y_v1)
        length_v2, angle_v2 = self.calc_vector_length_angle(self.x_v2, self.y_v2)
        
        self.angle1_rs = -(angle_v2 - 90)
        self.angle2_rs = -(angle_v1 - 90)

        self.length_v1_rs = self.angpix*self.current_img_fft_amp_rotated.shape[0]/abs((length_v1*math.sin(math.pi*(angle_v2 - angle_v1)/180)))
        self.length_v2_rs = self.angpix*self.current_img_fft_amp_rotated.shape[0]/abs((length_v2*math.sin(math.pi*(angle_v2 - angle_v1)/180)))
        self.x_v1_rs = self.length_v1_rs*math.cos(self.angle1_rs*math.pi/180)
        self.y_v1_rs = self.length_v1_rs*math.sin(self.angle1_rs*math.pi/180)
        self.x_v2_rs = self.length_v2_rs*math.cos(self.angle2_rs*math.pi/180)
        self.y_v2_rs = self.length_v2_rs*math.sin(self.angle2_rs*math.pi/180)

        self.circum_rs = self.v1_n*self.x_v1_rs + self.v2_n*self.x_v2_rs

    def opt_para(self):
        self.v1_n = self.tab2_spinboxes['Bessel_v1'].value()
        self.v2_n = self.tab2_spinboxes['Bessel_v2'].value()
        _, ang2 = self.calc_vector_length_angle(self.x_v2, self.y_v2)
        if ang2 < 90:
            # This is the opposite of the actual Bessel order, but treated this way for convenience
            # Internally it is all consistent
            self.v2_n = - self.v2_n

        def f(paras):
            x1 = paras[0]
            x2 = -self.v2_n*x1/self.v1_n
            l1, a1 = self.calc_vector_length_angle(x1, self.y_v1)
            l2, a2 = self.calc_vector_length_angle(x2, self.y_v2)
            a_rs1 = -(a2 - 90)
            a_rs2 = -(a1 - 90)
            l_v1_rs = self.angpix*self.current_img_fft_amp_rotated.shape[0]/abs((l1*math.sin(math.pi*(a1 - a2)/180)))
            l_v2_rs = self.angpix*self.current_img_fft_amp_rotated.shape[0]/abs((l2*math.sin(math.pi*(a1 - a2)/180)))
            y_v1_rs = l_v1_rs*math.sin(a_rs1*math.pi/180)
            y_v2_rs = l_v2_rs*math.sin(a_rs2*math.pi/180)
            y_rs_sum = abs(y_v1_rs*self.v1_n + y_v2_rs*self.v2_n)
            return y_rs_sum

        self.x_v1_old = self.x_v1
        self.x_v2_old = self.x_v2
        v1_v2_sum_len = self.x_v1 + abs(self.x_v2)
        x_v1 = v1_v2_sum_len*self.v1_n/(self.v1_n + abs(self.v2_n))
        x_v1_range = [x_v1*0.5, x_v1*1.5]

        optimum = minimize(f, [x_v1], bounds=[x_v1_range]) 
        self.x_v1 = optimum.x[0]
        self.x_v2 = -self.v2_n*self.x_v1/self.v1_n

        self.draw_tab2_lattice_lines()
        self.tab2_buttons['Show peaks'].setEnabled(True)
    
        self.calculate_rs_para()
        x_low = -0.002*self.circum_rs
        x_high = 1.002*self.circum_rs
        y_low = -0.2
        y_high = 5*(abs(self.y_v1_rs) + abs(self.y_v2_rs))
        self.reset_real_space_plot()
        self.add_real_space_point_group(x_low, x_high, y_low, y_high)
        if "Hide peaks" in self.tab2_buttons['Show peaks'].text():
            self.tab2_fft_peaks_toggle()

        offset_new = f([self.x_v1])
        print("\n******************************************************")
        print(f'Residual after refinement:{offset_new}')
        if offset_new < 10**-8:
            print("Looks good!\n\nIf lattice points do not match diffraction peaks,")
            print("adjust the two base vectors and refine again.")
            self.paramSaveButton.setEnabled(True)
            for i in self.buttons_rs:
                self.buttons_rs[i].setEnabled(True)
            for i in self.spin_boxes_rs:
                self.spin_boxes_rs[i].blockSignals(True)
                self.spin_boxes_rs[i].setValue(0)
                self.spin_boxes_rs[i].blockSignals(False)
                self.spin_boxes_rs[i].setEnabled(True)
            self.redraw_real_space_point_seq()
        else:
            print("Not so good. Adjust the two base vectors and refine again.")
        print("******************************************************\n")

    def undo_refine(self):
        try:
            self.x_v1 = self.x_v1_old
            self.x_v2 = self.x_v2_old
            for i in self.text_fields_rs:
                i.setText('')
            self.draw_tab2_lattice_lines()
            self.tab2_buttons['Refine'].setText('Refine')
            self.reset_real_space_plot()

            for i in self.buttons_rs:
                self.buttons_rs[i].setEnabled(False)
            for i in self.spin_boxes_rs:
                self.spin_boxes_rs[i].blockSignals(True)
                self.spin_boxes_rs[i].setValue(0)
                self.spin_boxes_rs[i].blockSignals(False)
                self.spin_boxes_rs[i].setEnabled(False)

            self.tab2_buttons['Show peaks'].setEnabled(False)
            if "Hide peaks" in self.tab2_buttons['Show peaks'].text():
                self.tab2_fft_peaks_toggle()
            
        except:
            return

    def tab2_refine_toggle(self):
        if "Refine" in self.tab2_buttons['Refine'].text():
            self.opt_para()
            self.tab2_buttons['Refine'].setText("Undo refine")
        else:
            self.undo_refine()
            self.tab2_buttons['Refine'].setText("Refine")

    def draw_tab2_fft_peaks(self):
        v1_fac = np.arange(-self.n_lower_lines2, self.n_lines2)
        v2_fac = np.arange(-self.n_lower_lines1, self.n_lines1)
        peaks = np.array([[0,self.origin[0],self.origin[1]]])
        peaks_m = np.empty_like(peaks)
        for i in v1_fac: 
            for j in v2_fac:
                #Again, v2_n positive actually means negative
                n = i*self.v1_n - j*self.v2_n
                x = self.x_v1*i + self.x_v2*j + self.origin[0]
                x_m = -self.x_v1*i - self.x_v2*j + self.origin[0]
                y = self.y_v1*i + self.y_v2*j + self.origin[1]
                if y > self.origin[1]:
                    peaks = np.vstack((peaks, [[n, x, y]]))
                    peaks_m = np.vstack((peaks_m, [[-n, x_m, y]]))
        
        self.tab2_fft_peaks_plot = self.tab2_axfft.plot(peaks_m[:,1], peaks_m[:,2],'co', peaks[:,1], peaks[:,2], 'ro', markersize=4)
        for i in peaks_m:
            self.tab2_fft_peaks_annot.append(self.tab2_axfft.annotate(str(int(i[0])), (i[1], i[2]), 
                fontsize=10, color='c', xytext=[4,0], textcoords='offset points'))
        for i in peaks:
            self.tab2_fft_peaks_annot.append(self.tab2_axfft.annotate(str(int(i[0])), (i[1], i[2]), 
                fontsize=10, color='r', xytext=[4,0], textcoords='offset points'))

        self.tab2_figfft_canvas.draw() 
        self.tab2_buttons['Show peaks'].setText('Hide peaks')
    
    def clear_tab2_fft_peaks(self):
        if len(self.tab2_fft_peaks_plot) >=1:
            for l in self.tab2_fft_peaks_plot:
                l.remove() 
            self.tab2_fft_peaks_plot = []
        if len(self.tab2_fft_peaks_annot) > 0:
            for label in self.tab2_fft_peaks_annot:
                label.remove()
            self.tab2_fft_peaks_annot = []
        self.tab2_figfft_canvas.draw()
        self.tab2_buttons['Show peaks'].setText('Show peaks')
    
    def tab2_fft_peaks_toggle(self):
        if 'Show peaks' in self.tab2_buttons['Show peaks'].text():
            self.draw_tab2_fft_peaks()
        else:
            self.clear_tab2_fft_peaks()

    def draw_ori_unitV(self):
        if self.ori_unitV_lines_rs == []:
            self.ori_unitV_lines_rs.append(self.ax_rs.plot([0, self.x_v1_rs], [0, self.y_v1_rs], color='b'))
            self.ori_unitV_lines_rs.append(self.ax_rs.plot([0, self.x_v2_rs], [0, self.y_v2_rs], color='b'))
        else:
            self.ori_unitV_lines_rs[0][0].set_data([0, self.x_v1_rs], [0, self.y_v1_rs])
            self.ori_unitV_lines_rs[1][0].set_data([0, self.x_v2_rs], [0, self.y_v2_rs])

    def set_real_space_plot_xylim(self):
        x_axis_rs_high = self.circum_rs
        x_axis_rs_low_ex = - 0.1*self.circum_rs
        x_axis_rs_high_ex = 1.1*self.circum_rs 
        y_axis_rs_low = -1*(abs(self.y_v1_rs) + abs(self.y_v2_rs))
        y_axis_rs_high = 4.75*(abs(self.y_v1_rs) + abs(self.y_v2_rs))
        
        self.ax_rs.set_xlim(x_axis_rs_low_ex, x_axis_rs_high_ex)
        self.ax_rs.set_ylim(y_axis_rs_low, y_axis_rs_high)
        self.ax_rs.spines['left'].set_position('zero')
        self.ax_rs.spines['bottom'].set_position('zero')
        self.ax_rs.spines['right'].set_position(('data', x_axis_rs_high))
        self.ax_rs.spines['top'].set_color('none')
        self.ax_rs.spines['bottom'].set_bounds(low=0, high=x_axis_rs_high)
        
        x_tick_pos = [0, x_axis_rs_high/4, x_axis_rs_high/2, 3*x_axis_rs_high/4, x_axis_rs_high]
        x_tick_txt = ['0', '90', '180', '270', '360']
        self.ax_rs.set_xticks(x_tick_pos)
        self.ax_rs.set_xticklabels(x_tick_txt)
        self.ax_rs.format_coord = lambda x, y: f'x={x:.1f}, y={y:.1f}' 

        self.fig_rs_canvas.draw()

    def set_points_draw_range_by_spinboxes(self):
        x_low_factor = self.spin_boxes_rs['x_low'].value()
        x_low = -0.002*2**(x_low_factor/2)*self.circum_rs
        x_high_factor = self.spin_boxes_rs['x_high'].value()
        x_high = (1 + 0.002*2**(x_high_factor/2))*self.circum_rs
        y_low_factor = self.spin_boxes_rs['y_low'].value()
        y_low = -0.2*2**(y_low_factor/2)
        y_high_factor = self.spin_boxes_rs['y_high'].value()
        y_high = (5 + y_high_factor/4)*(abs(self.y_v1_rs) + abs(self.y_v2_rs))
        self.ax_rs.set_ylim(top=y_high*0.95)
        self.add_real_space_point_group(x_low, x_high, y_low, y_high)
        self.redraw_real_space_point_seq()
        
    def add_real_space_point_group(self, x_low, x_high, y_low, y_high):
        try:
            self.clear_real_space_points()
            for i in range(-50, 200):
                for j in range(-50, 200):
                    x = self.x_v1_rs*i + self.x_v2_rs*j
                    y = self.y_v1_rs*i + self.y_v2_rs*j
                    if x_low <= x <= x_high and y_low <= y <= y_high:
                        self.dots_rs = np.append(self.dots_rs, [[i, j, x, y]], axis=0)
            self.dots_rs = self.dots_rs[np.lexsort((self.dots_rs[:, 0], self.dots_rs[:,3]), axis=0)]
            self.draw_real_space_point()
        except:
            QMessageBox.information(self, 'Error', 'Something went wrong!')

    def draw_real_space_point(self):
        if self.dots_rs_plot == None:
            self.dots_rs_plot, = self.ax_rs.plot(self.dots_rs[:, 2], self.dots_rs[:, 3], 'o', color='b')
        else:
            self.dots_rs_plot.set_data(self.dots_rs[:, 2], self.dots_rs[:, 3])
        self.fig_rs_canvas.draw()
        if len(self.dots_rs) > 2 and self.circum_rs != 0:
            self.draw_strand_switch.setCheckable(True)
        else:
            self.draw_strand_switch.setChecked(False)
            self.draw_strand_switch.setCheckable(False)
            self.delete_all_strand_lines()
    
    def clear_real_space_points(self):
        self.dots_rs = np.empty((0,4))
        self.draw_real_space_point()
        self.clear_real_space_point_labels()
        self.clear_real_space_point_seq()
        self.delete_all_strand_lines()
        self.draw_strand_switch.setChecked(False)
        self.draw_strand_switch.setCheckable(False)
    
    def real_space_label_toggle(self):
        if len(self.dots_rs) > 0 and self.dots_rs_label == {}:
            self.redraw_real_space_point_lables()
            self.clear_real_space_point_seq()
        elif len(self.dots_rs) > 0 and self.dots_rs_label != {}:
            self.clear_real_space_point_labels()
    
    def clear_real_space_point_labels(self):
        for i in self.dots_rs_label:
            self.dots_rs_label[i].remove()
        self.dots_rs_label = {}
        self.fig_rs_canvas.draw()
    
    def redraw_real_space_point_lables(self):
        for i in self.dots_rs:
                name = f' [{i[0]:.0f},{i[1]:.0f}]'
                self.dots_rs_label[name] = self.ax_rs.annotate(name, (i[2], i[3]), fontsize=7)
        self.fig_rs_canvas.draw()

    def real_space_seq_toggle(self):
        if len(self.dots_rs) > 0 and self.dots_rs_seq == []:
            self.redraw_real_space_point_seq()
            self.clear_real_space_point_labels()
        elif len(self.dots_rs) > 0 and self.dots_rs_seq != []:
            self.clear_real_space_point_seq()

    def redraw_real_space_point_seq(self):
        try:
            self.rise_rs_main = 0
            self.twist_rs_main = 360
            self.n_start_main = 1
            tmp_list = self.dots_rs[:, 0:2].tolist()
            if ([self.v1_n, self.v2_n] not in tmp_list) or self.v1_n == 0 or self.v2_n ==0:
                QMessageBox.information(self,'Alert','Two points on equator must be present!')
                return

            for i in range(0,len(self.dots_rs)):
                x = self.dots_rs[i, 2]
                x_rounded =  round(x, 5)
                y = self.dots_rs[i, 3]
                y_rounded = round(self.dots_rs[i, 3], 5)
                y_m1 = self.dots_rs[i-1, 3]
                if y_rounded < 0:
                    continue
                elif y_rounded == 0:
                    seq_id = 0
                    if 0 < x_rounded < round(self.circum_rs, 5):
                        self.n_start_main += 1
                else:
                    rise = y - y_m1
                    if rise > 0.0001:
                        self.rise_rs_main = rise
                        seq_id += 1
                if seq_id == 1:
                    tmp1 = 360*x/self.circum_rs
                    tmp2 = 360*(x - self.circum_rs)/self.circum_rs
                    if abs(tmp1) <= abs(tmp2):
                        twist_tmp = tmp1 
                    else:
                        twist_tmp = tmp2
                    if abs(self.twist_rs_main) > abs(twist_tmp):
                        self.twist_rs_main = twist_tmp
                    elif self.twist_rs_main == - twist_tmp:
                        self.twist_rs_main = abs(self.twist_rs_main)
                self.dots_rs_seq.append(self.ax_rs.annotate(' '+str(seq_id),self.dots_rs[i,2:4], fontsize=8))

            for i in self.dots_rs[np.round(self.dots_rs[:, 3], 5)==0]:
                x = round(i[2], 5)
                if (x < 0 or x > round(self.circum_rs, 5) or 
                    (x == 0 and self.twist_rs_main < 0) or 
                    (x == round(self.circum_rs, 5) and self.twist_rs_main > 0)): 
                    continue
                else:
                    slope = self.rise_rs_main*360/(self.twist_rs_main*self.circum_rs)
                    y1 = -slope*i[2] 
                    y2 = slope*self.circum_rs + y1
                    if [y1, y2] not in [j[3:5] for j in self.strand_line_info]:
                        self.strand_line_info.append([self.rise_rs_main, self.twist_rs_main, self.n_start_main, y1, y2])
                        self.strand_line_rs.append((self.ax_rs.plot([0, self.circum_rs], [y1, y2], ':', color='m', linewidth=1))[0])

            self.cursor_annotate_rise_twist()
            self.fig_rs_canvas.draw()

            self.labels_rs['Rise_Twist'].setText(
                f'Rise={self.rise_rs_main:.2f} \u212B; Twist={self.twist_rs_main:.1f}\u00B0; C{self.n_start_main}')
                
        except:
            QMessageBox.information(self, 'Error', 'Something wrong!\n At least two points needed.\n (0,0) point needed.')
        
    def clear_real_space_point_seq(self):
        for i in self.dots_rs_seq:
            i.remove()
        self.dots_rs_seq = []
        self.fig_rs_canvas.draw()
    
    def strand_line_toggle(self):
        if self.draw_strand_switch.isChecked():
            self.click_count = 0
            self.click_coor = []
            self.cid_strand_line = self.fig_rs_canvas.mpl_connect('button_press_event', self.draw_strand_line)
        else:
            self.fig_rs_canvas.mpl_disconnect(self.cid_strand_line)
        
    def draw_strand_line(self, event): 
        mod = QGuiApplication.keyboardModifiers()
        if mod == QtCore.Qt.ControlModifier: 
            if event.xdata == None or event.ydata == None:
                return
            self.click_count += 1
            self.click_coor.append([event.xdata, event.ydata])
            if self.click_count ==2:
                points_para = self.match_click_with_points(self.click_coor)
                if points_para[3:5] not in [x[3:5] for x in self.strand_line_info]:
                    self.strand_line_info.append(points_para)
                    self.strand_line_rs.append((self.ax_rs.plot([0, self.circum_rs], points_para[3:5] , ':', color='m', linewidth=1))[0])
                    self.fig_rs_canvas.draw()
                    self.cursor_annotate_rise_twist()
                self.click_count =0
                self.click_coor = []

    def match_click_with_points(self, coor):
        # Find the closest points and extension
        _, index1 = spatial.KDTree(self.dots_rs[:, 2:]).query(coor[0])
        _, index2 = spatial.KDTree(self.dots_rs[:, 2:]).query(coor[1])
        if index1 == index2:
            index2 = index1 + 1
        pc1 = self.dots_rs[index1, 2:]
        pc2 = self.dots_rs[index2, 2:]
        slope =  (pc1[1] - pc2[1])/(pc1[0] - pc2[0])
        y1 = pc1[1] - pc1[0]*slope
        y2 = self.circum_rs*slope + y1
        
        rise_rs = abs(pc1[1] - pc2[1])
        if rise_rs != 0:
            if pc2[1] > pc1[1]:
                twist_rs = (pc2[0] - pc1[0])*360/self.circum_rs
            else:
                twist_rs = (pc1[0] - pc2[0])*360/self.circum_rs

            dot_strand_dist_array = []
            for i in range(len(self.dots_rs)):
                if i != index1 and i != index2:
                    dist_h = abs((self.dots_rs[i, 3] - y1)/slope - self.dots_rs[i, 2])
                    if dist_h > self.circum_rs/500:
                        dot_strand_dist_array.append(dist_h)
            dot_strand_dist_array = sorted(dot_strand_dist_array)
            n_start = int(round(self.circum_rs/dot_strand_dist_array[0]))
        else:
            n_start = 0
            twist_rs = 0

        print("\n******************************************************")
        print(f'Symmetry of this helical family:\nRise={rise_rs:.2f} \u212B; Twist={twist_rs:.1f}\u00B0; {n_start}-start')
        print("******************************************************\n")

        return [rise_rs, twist_rs, n_start, y1, y2]

    def delete_all_strand_lines(self):
        if len(self.strand_line_rs) >=1:
            for i in range(len(self.strand_line_rs)):
                self.strand_line_rs[i].remove() 
            self.strand_line_rs = []
            self.strand_line_info = []
            self.fig_rs_canvas.draw()
        self.labels_rs['Rise_Twist'].setText('')
    
    def delete_last_strand_line(self):
        if len(self.strand_line_rs) >=1:
            self.strand_line_rs[-1].remove()
            del(self.strand_line_rs[-1])
            del(self.strand_line_info[-1])
        self.fig_rs_canvas.draw()
        self.labels_rs['Rise_Twist'].setText('')

    def cursor_annotate_rise_twist(self):
        self.cursor_rise_twist = mplcursors.cursor(self.strand_line_rs, multiple=True, bindings={"select": 2})
        @self.cursor_rise_twist.connect('add')
        def _(sel):
            rise = self.strand_line_info[self.strand_line_rs.index(sel.artist)][0]
            twist = self.strand_line_info[self.strand_line_rs.index(sel.artist)][1]
            n_start = self.strand_line_info[self.strand_line_rs.index(sel.artist)][2]
            sel.annotation.set(text=f'Rise={rise:.2f} \u212B\nTwist={twist:.1f}\u00B0\n{n_start}-start helix')
            sel.annotation.arrow_patch.set(arrowstyle='simple',fc='yellow', alpha=0.5)

    def reset_real_space_plot(self):
        self.clear_real_space_points()
        self.delete_all_strand_lines()
        self.calculate_rs_para()
        self.draw_ori_unitV()
        self.set_real_space_plot_xylim()
        
    def generate_relion_command(self):
        try:
            rise, twist, pg, td, sd, bd, ps = self.get_values_from_tab3()

            txt = f'relion_helix_toolbox --simulate_helix --o init_model.mrc --subunit_diameter {sd} \
--cyl_outer_diameter {td} --angpix {ps} --rise {rise} --twist {twist} --boxdim {bd} --sym_Cn {pg}'
            info = '\n\nUse the Relion command above to genenrate the initial model\n\
(Make sure parameters make sense)'
            txt = ''.join((txt,info))
            self.tab3_text['rc'].setText(txt)
        except:
            QMessageBox.information(self, 'Error', 'Check parameters')
            
    def draw_3D_model(self):
        try:
            rise, twist, pg, td, sd, bd, ps = self.get_values_from_tab3()
            bd_ang = bd*ps
            pg_angle = 360/pg
            self.ax_3d.cla()
            self.ax_3d.set_xlabel('x (\u212B)')
            self.ax_3d.set_ylabel('y (\u212B)')
            self.ax_3d.set_zlabel('z (\u212B)')

            #Draw cylinder
            z_cyl = np.array([0, bd_ang])
            phi = np.linspace(0,np.pi*2,180)
            x_cyl = td/2*np.cos(phi)
            y_cyl = td/2*np.sin(phi)
            x_cyl_m, z_cyl_m = np.meshgrid(x_cyl, z_cyl)
            y_cyl_m, z_cyl_m = np.meshgrid(y_cyl, z_cyl)
            self.ax_3d.plot_surface(x_cyl_m, y_cyl_m, z_cyl_m, alpha=0.3, color='gray') 

            #draw strand lines
            z_1 = np.arange(bd_ang, step=rise/50)
            for i in range(pg):
                x_1 = td/2*np.cos((z_1/rise*twist+i*pg_angle)*np.pi/180)
                y_1 = td/2*np.sin((z_1/rise*twist+i*pg_angle)*np.pi/180)
                self.ax_3d.plot(x_1,y_1,z_1, alpha=0.3, lw=1)

            #Draw points
            n_points = int(bd_ang/rise+1)
            x = np.array([[td/2*np.cos((i*twist+j*pg_angle)*np.pi/180) for j in range(pg)] for i in range(n_points)])
            y = np.array([[td/2*np.sin((i*twist+j*pg_angle)*np.pi/180) for j in range(pg)] for i in range(n_points)])
            z = np.array([[rise*i for j in range(pg)] for i in range(n_points)])
            x = x.reshape(1,-1)
            y = y.reshape(1,-1)
            z = z.reshape(1,-1)
            #self.ax_3d.scatter(x, y, z, color='b', s=60)
            self.ax_3d.scatter(x[y>0], y[y>0],z[y>0], color='blue', s=80)
            self.ax_3d.scatter(x[y<=0], y[y<=0],z[y<=0], color='blue', s=80)
            self.ax_3d.view_init(elev=10, azim=-90)
            self.ax_3d.grid(False)

            self.ax_3d.set_xlim(-bd_ang/2,bd_ang/2)
            self.ax_3d.set_ylim(-bd_ang/2, bd_ang/2)
            self.ax_3d.set_zlim(0,bd_ang)
            
            self.toolbar_3d.update()
            self.fig_3d_canvas.draw()
        except:
            QMessageBox.information(self, 'Error', 'Check parameters')
    
    def get_values_from_tab3(self):
        rise = float(self.tab3_text['rise'].text())
        twist = float(self.tab3_text['twist'].text())
        td = float(self.tab3_text['td'].text())
        sd = float(self.tab3_text['sd'].text())
        bd = int(self.tab3_text['bd'].text())
        ps = float(self.tab3_text['ps'].text())
        pg = self.tab3_text['pg'].text() 
        pg = int(pg[1:]) if pg.startswith('C') else int(pg)
        return rise, twist, pg, td, sd, bd, ps 

    def update_tab3_fields(self):
        if self.radius_H != 0:
            circum = 2*math.pi*self.radius_H
        else:
            circum = self.circum_rs
        td = round(circum/math.pi)
        
        try:
            sd = round(2*math.sqrt(circum*self.rise_rs_main/self.n_start_main/math.pi))
        except:
            sd = 30
        
        try: 
            bd = self.img_xdim
        except:
            bd = 200

        self.tab3_text['rise'].setText(str(round(self.rise_rs_main,2)))
        self.tab3_text['twist'].setText(str(round(self.twist_rs_main,1)))
        self.tab3_text['pg'].setText(str(self.n_start_main))
        self.tab3_text['td'].setText(str(td))
        self.tab3_text['sd'].setText(str(sd))
        self.tab3_text['bd'].setText(str(bd))
        self.tab3_text['ps'].setText(str(round(self.angpix, 2)))
    
    def update_tab2_fft_upon_tab_switch(self, i):
        if i == 1:
            self.draw_tab2_fft()
    
    def button_push_connect(self):
        self.minSigma_chooser.valueChanged.connect(functools.partial(self.change_contrast, 'low'))
        self.maxSigma_chooser.valueChanged.connect(functools.partial(self.change_contrast, 'high'))
        self.img_rotation_chooser.valueChanged.connect(self.set_img_rotation_shift)
        self.img_shift_chooser.valueChanged.connect(self.set_img_rotation_shift)
        self.auto_align_toggle.toggled.connect(self.auto_align_toggle_signal)
        self.twoD_inv_toggle.toggled.connect(self.calc_current_img_array)
        self.ps_cmap_toggle.toggled.connect(self.draw_tab1_fft)
        self.ps_cmap_toggle.toggled.connect(self.draw_tab2_fft)
        self.tab1_buttons_col2['Set Angpix'].clicked.connect(self.set_angpix)
        self.tab1_text_col2['Angpix'].returnPressed.connect(self.set_angpix)
        self.tab1_buttons_col1['Draw LL'].clicked.connect(self.toggle_draw_tab1_LL)
        self.tab1_buttons_col1['Set LL dist'].clicked.connect(self.draw_tab1_LL)
        self.tab1_text_col1['Y_dist'].returnPressed.connect(self.draw_tab1_LL)
        self.tab1_buttons_col2['Measure'].clicked.connect(self.measure_distance)
        self.tab1_buttons_col2['Calc LL plot'].clicked.connect(self.calc_LL_plot)
        self.tab1_text_col2['helix_radius'].returnPressed.connect(self.check_LL_plots_inputs)
        self.tab1_text_col2['radius_error'].returnPressed.connect(self.check_LL_plots_inputs)
        self.tab1_text_col2['LL_Y_range'].returnPressed.connect(self.check_LL_plots_inputs)
        self.tab1_text_col2['LL_width'].returnPressed.connect(self.check_LL_plots_inputs)
        self.tab1_text_col2['Bessel_order'].returnPressed.connect(self.check_LL_plots_inputs)
        
        self.tab2_symmetrize_fft_switch.toggled.connect(self.draw_tab2_fft)
        self.tab2_buttons['Draw lattice'].clicked.connect(self.tab2_lattice_lines_toggle)
        self.tab2_buttons['Refine'].clicked.connect(self.tab2_refine_toggle)
        self.tab2_buttons['Show peaks'].clicked.connect(self.tab2_fft_peaks_toggle)
        self.buttons_rs['[h,k] label'].clicked.connect(self.real_space_label_toggle)
        self.buttons_rs['Sequence label'].clicked.connect(self.real_space_seq_toggle)
        self.buttons_rs['Delete last'].clicked.connect(self.delete_last_strand_line)
        self.buttons_rs['Delete all'].clicked.connect(self.delete_all_strand_lines)
        self.draw_strand_switch.stateChanged.connect(self.strand_line_toggle)
        self.spin_boxes_rs['x_low'].valueChanged.connect(self.set_points_draw_range_by_spinboxes)
        self.spin_boxes_rs['x_high'].valueChanged.connect(self.set_points_draw_range_by_spinboxes)
        self.spin_boxes_rs['y_low'].valueChanged.connect(self.set_points_draw_range_by_spinboxes)
        self.spin_boxes_rs['y_high'].valueChanged.connect(self.set_points_draw_range_by_spinboxes)
        
        self.tab3_buttons['Autofill'].clicked.connect(self.update_tab3_fields)
        self.tab3_buttons['Draw 3D'].clicked.connect(self.draw_3D_model)
        self.tab3_buttons['Relion command'].clicked.connect(self.generate_relion_command)

        self._CentralWidget.currentChanged.connect(self.update_tab2_fft_upon_tab_switch)

def main():
    app = QApplication(sys.argv)
    gui = MainWindow()
    gui.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
