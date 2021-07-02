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
from scipy.optimize import minimize
from scipy import ndimage, spatial, special, signal

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self._CentralWidget = QTabWidget(self)
        self.setWindowTitle('Helical indexer')
        self.setCentralWidget(self._CentralWidget)
        self.tab1 = QWidget()
        self.tab2 = QWidget()
        self.tab1.layout = QHBoxLayout()
        self.tab2.layout = QHBoxLayout()
        self.tab1.setLayout(self.tab1.layout)
        self.tab2.setLayout(self.tab2.layout)
        self._CentralWidget.addTab(self.tab1, "Power spectrum analyzer")
        self._CentralWidget.addTab(self.tab2, "Lattice generator")

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
        self.LL_distance = 1
        self.img_rotation = 0.0
        self.minSigmaValue = -1
        self.maxSigmaValue =  8
        self.angpix = 1
        self.img_cmap = 'Greys_r'

        # Tab1 default Parameters
        self.origin = [0,0]
        self.tab1_LL = []
        self.tab1_LL_label = []

        self.mrc_data_array = None
        self.current_slice = 0
        self.twoD_img_shown = None
        self.tab1_fft_shown = None
        self.measure_on = False
        self.tab1_LL_draw_on = False
        self.dist_line_in_2D = ''

        self.radius_H = 0
        self.ruler_width = 0
        self.ruler_height = 0

        self.OneD_profile_line = ''
        self.OneD_profile_peaks = ''
        self.OneD_profile_on = False
        self.OneD_profile_line_scale = 1
        self.LL_amp_plot = ''
        self.LL_bessel_plot = ''
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

        self.ori_unitV_lines_rs = []
        self.twist_rs = 0
        self.rise_rs = 0
        self.n_start = 0
        self.circum_rs = 0

        self.dots_rs = []
        self.dots_rs_label = {} 
        self.dots_rs_seq = []
        self.dots_rs_plot = None
        self.strand_line_rs = []
        self.strand_line_info = []

    def overall_layout(self):
        #Tab1
        self.twoD_img_shown = None

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
        self.tab2_ctrl_left_widget.setFixedSize(450,150)
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
    
    def tab1_fft_window(self):
        self.tab1_figfft, self.tab1_axfft = plt.subplots()
        self.tab1_figfft.suptitle('Power spectrum', fontsize=10)
        self.tab1_figfft.tight_layout()
        self.tab1_axfft.axis('off')
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
            'Rotate Image:': (3, 0, 1, 1),
            '(degree)': (3, 2, 1, 1),
            '(\u212B/pixel)': (4, 2, 1, 1),
            '(Scale)': (5, 2, 1, 1),
            '(pixel)': (6, 2, 1, 1),
            'Repeat distance:': (7, 1, 1, 2),
        }
        buttons = {
            'Set Angpix':(4, 0, 1, 1),
            '1D profile': (5, 0, 1, 1),
            'Set LL dist': (6, 0, 1, 1),
            'Draw LL': (7, 0, 1, 1),
        }

        txt_fields = {
            'Angpix': (4, 1, 1, 1, self.angpix),
            '1D_scale': (5, 1, 1, 1, self.OneD_profile_line_scale),
            'Y_dist': (6, 1, 1, 1, self.LL_distance),
        }

        self.minSigma_chooser = QSlider(QtCore.Qt.Horizontal)
        self.minSigma_chooser.setMinimum(-30.5)
        self.minSigma_chooser.setMaximum(30)
        self.minSigma_chooser.setValue(self.minSigmaValue)
        self.minSigma_chooser.setFixedWidth(200)
        layout.addWidget(self.minSigma_chooser, 0, 1, 1, 2)
        self.minSigma_chooser.setEnabled(False)

        self.maxSigma_chooser = QSlider(QtCore.Qt.Horizontal)
        self.maxSigma_chooser.setMinimum(-30)
        self.maxSigma_chooser.setMaximum(30.5)
        self.maxSigma_chooser.setValue(self.maxSigmaValue)
        self.maxSigma_chooser.setFixedWidth(200)
        layout.addWidget(self.maxSigma_chooser, 1, 1, 1, 2)
        self.maxSigma_chooser.setEnabled(False)

        self.slice_chooser = QSpinBox()
        self.slice_chooser.setValue(1)
        self.slice_chooser.setRange(1,10)
        self.slice_chooser.setEnabled(False)
        self.slice_chooser.setFixedWidth(70)
        layout.addWidget(self.slice_chooser, 2, 1, 1, 1)
        self.slice_chooser.setToolTip('Change this number to choose 2D class to display.')

        self.img_cmap_toggle = QCheckBox('B\u2B0CW')
        self.img_cmap_toggle.setCheckable(False)
        layout.addWidget(self.img_cmap_toggle, 2, 2, 1, 1)
        self.img_cmap_toggle.setToolTip('Invert the black and white of the power spectrum')

        self.img_rotation_chooser = QDoubleSpinBox()
        self.img_rotation_chooser.setValue(self.img_rotation)
        self.img_rotation_chooser.setRange(-180.0,180.0)
        self.img_rotation_chooser.setSingleStep(1)
        self.img_rotation_chooser.setEnabled(False)
        self.img_rotation_chooser.setFixedWidth(70)
        layout.addWidget(self.img_rotation_chooser, 3, 1, 1, 1)
        self.img_rotation_chooser.setToolTip('Change this number to rotate the 2D image and power spectrum.')

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

        self.tab1_text_col1['Angpix'].setToolTip('Angpix of image. (Example: 1.08)')
        self.tab1_text_col1['1D_scale'].setToolTip('Scale of the 1D profile. (Example: 1)')
        self.tab1_text_col1['Y_dist'].setToolTip('Layerline distance. (Example: 2.1)')

        self.tab1_buttons_col1['1D profile'].setToolTip('Click to draw vertical 1D profile of the power spectrum\nUseful for finding the layerline distance')
        self.tab1_buttons_col1['Set Angpix'].setToolTip('Click to set angstrom/pixel of the image\nNo need to set manually if read from MRC image correctly')
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
            'Measure': (0, 0, 1, 1),
            'Calc LL plot': (7, 0, 1, 1),
        }

        labels = {
            'Measure': (0, 1, 1, 2),
            'Radius of helix (\u212B):': (1, 0, 1, 2),
            'Layerline plot parameters:': (3, 0, 1, 3),
            'Y-coord range (pixel):': (4, 0, 1, 2),
            'Plot width (pixel):': (5, 0, 1, 2),
            'Bessel order (integer):': (6, 0, 1, 2),
            'CC=': (7, 2, 1, 1),
        }

        txt_fields = {
            'helix_radius': (1, 2, 1, 1),
            'LL_Y_range': (4, 2, 1, 1),
            'LL_width': (5, 2, 1, 1),
            'Bessel_order': (6, 2, 1, 1),
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

        self.tab1_buttons_col2['Measure'].setToolTip('''click me and then two points in 2D image to measure distance
        To get the radius of the helix, click the two side edges,
        radius will be automatically calculated and set.
        Radius can also be set manually by typing number into the field below.''')

        for txt, pos in txt_fields.items():
            self.tab1_text_col2[txt] = QLineEdit()
            self.tab1_text_col2[txt].setMaximumWidth(60)
            layout.addWidget(self.tab1_text_col2[txt], pos[0], pos[1], pos[2], pos[3])
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
        divider.setFrameShape(QFrame.HLine)
        divider.setFrameShadow(QFrame.Sunken)
        layout.addWidget(divider, 2, 0, 1, 3) 

        spacer_item = QSpacerItem(1, 30, QSizePolicy.Expanding, QSizePolicy.Minimum)
        layout.addItem(spacer_item, 0, 3, 7, 1)

        self.tab1_ctrl_layout.addLayout(layout)

    def class2D_window(self):
        self.fig2d, self.ax2d = plt.subplots()
        self.fig2d.suptitle('2D class average', fontsize=10)
        self.ax2d.axis('off')
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
        self.tab2_axfft.format_coord = lambda x, y: f'x={x-self.origin[0]:.1f}, y= {y-self.origin[1]:.1f}' 
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

        buttons = {
            'Add points:': (0, 0, 1, 1),
            'Clear points': (3, 0, 1, 1),
            'Label On/Off': (0, 3, 1, 1),
            'Seq. On/Off': (0, 4, 1, 1),
            'Delete last': (3, 3, 1, 1),
            'Delete all': (3, 4, 1, 1),
        }

        labels_rs = {
            'X range': (1, 0, 1, 1),
            'Y range': (1, 1, 1, 1),
            'Rise_Twist': (4, 3, 1, 3),
        }

        text_fields = [
            (2, 0, 1, 1),
            (2, 1, 1, 1),
        ]

        for i in range(len(text_fields)):
            self.text_fields_rs.append(QLineEdit())
            self.text_fields_rs[i].setFixedSize(80, 25)
            self.tab2_ctrl_right_layout.addWidget(self.text_fields_rs[i], text_fields[i][0], text_fields[i][1], text_fields[i][2], text_fields[i][3])
        
        for txt, pos in labels_rs.items():
            if 'Rise_Twist' in txt:
                self.labels_rs[txt] = QLabel('')
            else:
                self.labels_rs[txt] = QLabel(txt)
            self.tab2_ctrl_right_layout.addWidget(self.labels_rs[txt], pos[0], pos[1], pos[2], pos[3])

        for txt, pos in buttons.items():
            self.buttons_rs[txt] = QPushButton(txt)
            self.buttons_rs[txt].setFixedSize(100, 30)
            self.buttons_rs[txt].setFocusPolicy(QtCore.Qt.NoFocus)
            self.buttons_rs[txt].setEnabled(False)
            self.tab2_ctrl_right_layout.addWidget(self.buttons_rs[txt], pos[0], pos[1], pos[2], pos[3])

            
        self.buttons_rs['Add points:'].setToolTip('Add a group of points by specifying:\n(x_low,x_high) and (y_low, y_high) in pixel/angstrom')
        self.text_fields_rs[0].setToolTip('''Set X coordinate range of the plot (low, high)\n
        Hover mouse on plot and read the coordinates at the right side of the toolbar
        Note: this is not the azimuthal angle as labelled on the X-axis
        Normally set autmatically if refinement is successful
        ''')

        self.text_fields_rs[1].setToolTip('''Set Y coordinate range of the plot (low, high)\n
        Hover mouse on plot and read the coordinates at the right side of the toolbar
        ''')

        self.buttons_rs['Label On/Off'].setToolTip('Label points with (h, k) index')
        self.buttons_rs['Seq. On/Off'].setToolTip('Label points with 1-start helix sequential number')

        self.draw_strand_switch = QCheckBox('Draw strand')
        self.draw_strand_switch.setCheckable(False)
        self.tab2_ctrl_right_layout.addWidget(self.draw_strand_switch, 2, 3, 1, 1)
        self.draw_strand_switch.setToolTip('''Toggle switch for drawing strand\n
        After successful refine, Cmd/Ctrl-click two neighboring points to:
        Define a strand and calculate rise and twist per unit\n
        Once strands are drawn, annotation can be displayed on plot by middle-mouse button click on strand
        Hold and drag to move annotation
        Right-mouse button click to remove annotation''')

        divider1_rs = QFrame()
        divider1_rs.setFrameShape(QFrame.VLine)
        divider1_rs.setFrameShadow(QFrame.Sunken)
        self.tab2_ctrl_right_layout.addWidget(divider1_rs, 0, 2, 5, 1)
        divider2_rs = QFrame()
        divider2_rs.setFrameShape(QFrame.HLine)
        divider2_rs.setFrameShadow(QFrame.Sunken)
        self.tab2_ctrl_right_layout.addWidget(divider2_rs, 1, 3, 1, 2)

    def tab2_left_side_ctrl(self):
        self.tab2_labels = {}
        self.tab2_buttons = {}
        self.tab2_spinboxes = {}

        labels ={
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
            'Clear': (5, 1, 1, 1),
            'Refine': (5, 3, 1, 1),
            'Undo': (5, 4, 1, 1),
        }

        boxes = {
            'n_upper_v1': (1, 1, 1, 1, self.n_lines1, 0, 50),
            'n_lower_v1': (1, 2, 1, 1, self.n_lower_lines1, 0, 50),
            'lw_v1': (1, 3, 1, 1, self.v1_lattice_lw, 0, 3),
            'Bessel_v1': (1, 4, 1, 1, 3, 1, 200),
            'n_upper_v2': (2, 1, 1, 1, self.n_lines2, 0, 50),
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
        Residual after refinement (printed in terminal) is not zero (\U0001F61F).\n
        If successful (\U0001f642), real space lattice will be shown in the right panel.
        If "pix~1/\u212B" set correctly, "rise" in real space will have correct scale in angstrom.\n''') 

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

    def open_2D_classes(self):
        self.power_spec_only = 0
        mrc_file_name = QFileDialog.getOpenFileName(self, 'choose MRC', '.', filter='mrc file (*.mrc *.mrcs)')
        if mrc_file_name[0] == '':
            return
        
        self.reset_tab1_display()
        self.reset_tab2_display()
        self.setWindowTitle(f'Helical indexer: {mrc_file_name[0]}')

        with mrcfile.open(mrc_file_name[0]) as f:
            self.mrc_data_array = f.data
            if self.mrc_data_array.ndim == 2:
                self.mrc_data_array = self.mrc_data_array.reshape(1, self.mrc_data_array.shape[0], self.mrc_data_array.shape[1])

            try:
                x_pix = f.header.nx
                x_dim = f.header.cella.x
                self.angpix = x_dim/x_pix
                if self.angpix == 0:
                    self.angpix = 1
                    QMessageBox.information(self, 'Alert', 'Could not read angpix\n Set to 1!')
            except:
                self.angpix = 1
                QMessageBox.information(self, 'Alert', 'Could not read angpix\n Set to 1!')

        number_of_slice = self.mrc_data_array.shape[0]
        self.slice_chooser.setMaximum(number_of_slice)
        self.tab1_text_col1['Angpix'].setText(f'{self.angpix:3.2f}')
        self.set_initial_origin()
        self.draw_2d_image()

        self.img_rotation_chooser.setValue(0)
        self.img_rotation_chooser.setEnabled(True)
        self.set_img_rotation()
        self.draw_tab2_fft()
        self.tab2_symmetrize_fft_switch.setCheckable(True)
        self.tab2_symmetrize_fft_switch.setChecked(False)

    def load_power_spec(self):
        self.power_spec_only = 1
        power_spec_filename = QFileDialog.getOpenFileName(self, 'choose power spec image', '.', 
                                                          filter="Image file (*.tiff *tif *.jpg *.png *jpeg *mrc)")
        if power_spec_filename[0] == '':
            return

        self.setWindowTitle(f'Helical indexer: {power_spec_filename[0]}')
        self.reset_tab2_display()
        self.reset_tab1_display()
        self.tab1_buttons_col2['Measure'].setEnabled(False)
    
        if '.mrc' in power_spec_filename[0]:
            with mrcfile.open(power_spec_filename[0], permissive=True) as f:
                self.current_img_fft_amp = f.data
                try:
                    x_pix = f.header.nx
                    x_dim = f.header.cella.x
                    self.angpix = x_dim/x_pix
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
            self.angpix = 1
            QMessageBox.information(self, 'Information', 'Reading non-MRC image\nAngpix set to 1\nManually set it if you know it')

        self.current_img_fft_phase = np.zeros_like(self.current_img_fft_amp)
        self.tab1_text_col1['Angpix'].setText(f'{self.angpix:3.2f}')
        self.mrc_data_array = np.array([0]) 
        self.set_initial_origin()
        self.draw_tab1_fft()

        self.img_rotation_chooser.setValue(0)
        self.img_rotation_chooser.setEnabled(True)
        self.set_img_rotation()
        self.draw_tab2_fft()
        self.tab2_symmetrize_fft_switch.setCheckable(True)
        self.tab2_symmetrize_fft_switch.setChecked(False)
        
        QMessageBox.information(self, 'Alert', 'Loaded power spectrum only\nNo phase information\nIgnore phase plot.')

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
                f.write(f'{"Bessel order 2:":20s}{self.v2_n:10d}\n')
                f.write(f'{"RS length:":20s}{self.length_v2_rs:10.2f}\n')
                f.write(f'{"RS angle:":20s}{self.angle2_rs:10.2f}\n\n')

                f.write(f'{"Helix radius:":20s}{self.radius_H:10.2f}\n')
                f.write(f'{"Rise/subunit:":20s}{self.rise_rs:10.2f}\n')
                f.write(f'{"Twist/subunit:":20s}{self.twist_rs:10.2f}\n')
                f.write(f'{"n-start:":20s}{self.n_start:10d}')

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
            self.v1_n = para_dict['Bessel order 1']
            self.v2_n = para_dict['Bessel order 2']
            LL_distance = para_dict['Layerline distance']
            radius = para_dict['Helix radius']

            #self.orig_x_spinbox.setValue(0)
            #self.orig_y_spinbox.setValue(0)
            self.origin_x_click_value = self.origin[0]
            self.origin_y_click_value = self.origin[1]

            self.tab1_text_col1['Angpix'].setText(f'{angpix:.2f}')
            self.tab1_text_col1['Y_dist'].setText(f'{LL_distance:.2f}')
            self.set_angpix()
            self.tab1_text_col2['helix_radius'].setText(f'{radius:.2f}')
            self.check_LL_plots_inputs()
            self.draw_tab1_LL()

            self.tab2_spinboxes['Bessel_v1'].setValue(3)
            self.tab2_spinboxes['Bessel_v2'].setValue(1)
            self.draw_tab2_lattice_lines()

        except:
            QMessageBox.information(self, 'Error', 'Check parameter file format!')                
    
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
            self.tab2_figfft_canvas.draw()
        self.undo_refine()
        self.clear_tab2_lattice_lines()
        
    def draw_2d_image(self):
        if self.mrc_data_array.ndim == 1:
            return
        self.current_img_array = ndimage.rotate(self.mrc_data_array[self.current_slice], self.img_rotation, reshape=False)
        min = self.current_img_array.min()
        max = self.current_img_array.max()

        if self.twoD_img_shown == None:
            self.twoD_img_shown = self.ax2d.imshow(self.current_img_array, origin='lower', cmap='Greys_r')
            self.ax2d.set_xlim(0, self.current_img_array.shape[1])
            self.ax2d.set_ylim(0, self.current_img_array.shape[0])
            self.toolbar2d.update()
        else:
            self.twoD_img_shown.set_data(self.current_img_array)
        self.twoD_img_shown.set_clim(min, max)
        self.fig2d_canvas.draw()

        self.calculate_fft()
    
    def set_img_rotation(self):
        self.img_rotation = self.img_rotation_chooser.value() 
        if self.mrc_data_array is not None:
            self.draw_2d_image()
        if self.current_img_fft_amp is not None:
            self.draw_tab1_fft()
            self.check_LL_plots_inputs()

    def reset_tab1_display(self):
        for buttons in self.tab1_buttons_col1.values():
            buttons.setEnabled(True)
        for buttons in self.tab1_buttons_col2.values():
            buttons.setEnabled(True)
        if self.measure_on == True:
            self.measure_distance()
        self.minSigma_chooser.setEnabled(True)
        self.maxSigma_chooser.setEnabled(True)
        self.slice_chooser.setEnabled(True)
        self.img_cmap_toggle.setCheckable(True)
        self.slice_chooser.setValue(1)
        self.img_rotation = 90
        self.tab1_text_col2['LL_Y_range'].setText('')
        self.tab1_text_col2['LL_width'].setText('')
        self.tab1_text_col2['Bessel_order'].setText('')
        self.tab1_text_col2['helix_radius'].setText('')
        self.tab1_labels_col2['Measure'].setText('')
        self.contrast_high = 8
        self.contrast_low = -1
        self.minSigma_chooser.setValue(-1)
        self.maxSigma_chooser.setValue(8)
        self.clear_tab1_LL()
        if self.twoD_img_shown is not None:
            self.twoD_img_shown.remove()
            self.twoD_img_shown = None
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
            self.LL_legend.remove()
            self.LL_legend = ''
            self.LL_phase_plot.remove()
            self.LL_phase_plot = ''
            self.LL_phase_diff_ind.remove()
            self.LL_phase_diff_ind = ''
            self.fig_bessel_canvas.draw()

    def calculate_fft(self):
        try:
            current_img_fft = np.fft.fftshift(np.fft.fft2(self.current_img_array))
            self.current_img_fft_amp = np.abs(current_img_fft)
            self.current_img_fft_phase = np.angle(current_img_fft)
            self.draw_tab1_fft()    

        except:
            QMessageBox.information(self, 'Error', 'Load a 2D image first')
    
    def draw_tab1_fft(self):
        try:
            if self.power_spec_only == 1:
                self.current_img_fft_amp_rotated = ndimage.rotate(self.current_img_fft_amp, self.img_rotation, reshape=False)
            else:
                self.current_img_fft_amp_rotated = self.current_img_fft_amp
            if self.tab1_fft_shown is None:
                self.tab1_fft_shown = self.tab1_axfft.imshow(self.current_img_fft_amp_rotated, origin='lower', cmap=self.img_cmap)
                self.tab1_axfft.set_xlim(0, self.current_img_fft_amp_rotated.shape[1])
                self.tab1_axfft.set_ylim(0, self.current_img_fft_amp_rotated.shape[0])
                self.tab1_fft_toolbar.update()
            else:
                self.tab1_fft_shown.set_data(self.current_img_fft_amp_rotated)

            high = self.current_img_fft_amp_rotated.mean() + self.contrast_high*self.current_img_fft_amp_rotated.std()
            low = self.current_img_fft_amp_rotated.mean() + self.contrast_low*self.current_img_fft_amp_rotated.std()
            self.tab1_fft_shown.set_clim(vmin=low, vmax=high)
            self.tab1_figfft_canvas.draw()
            self.new_mrcs_status = 0

            if self.OneD_profile_on == True:
                self.Draw_oneD_profile()
            
            self.fftSaveButton.setEnabled(True)
            self.paramLoadButton.setEnabled(True)

        except:
            QMessageBox.information(self, 'Error', 'Load a 2D image first')

    def draw_tab2_fft(self):
        try:
            if self.tab2_symmetrize_fft_switch.isChecked():
                fft_for_tab2 = self.symmetrize_fft(self.current_img_fft_amp_rotated)
            else:
                fft_for_tab2 = self.current_img_fft_amp_rotated
            
            if self.tab2_fft_shown is None: 
                self.tab2_fft_shown = self.tab2_axfft.imshow(fft_for_tab2, origin='lower', cmap=self.img_cmap)
                self.tab2_axfft.set_xlim(0, fft_for_tab2.shape[1])
                self.tab2_axfft.set_ylim(0, fft_for_tab2.shape[0])
                self.tab2_fft_toolbar.update()
            else:
                self.tab2_fft_shown.set_data(fft_for_tab2)

            high = fft_for_tab2.mean() + self.contrast_high*fft_for_tab2.std()
            low = fft_for_tab2.mean() + self.contrast_low*fft_for_tab2.std()
            self.tab2_fft_shown.set_clim(vmin=low, vmax=high)
            self.tab2_figfft_canvas.draw()
        except:
            return

    def set_initial_origin(self):
        if self.power_spec_only == 0:
            self.origin[1] = self.mrc_data_array.shape[1]/2
            self.origin[0] = self.mrc_data_array.shape[2]/2
        else:
            self.origin[1] = self.current_img_fft_amp.shape[0]/2
            self.origin[0] = self.current_img_fft_amp.shape[1]/2
        self.tab1_axfft.format_coord = lambda x, y: f'x={round(x)-self.origin[0]:.0f}, y={round(y)-self.origin[1]:.0f}'

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

    def OneD_profile_toggle(self):
        if self.OneD_profile_on == False:
            self.Draw_oneD_profile()
            self.OneD_profile_on = True
        else:
            if self.OneD_profile_line != '':
                self.OneD_profile_line.remove()
                self.OneD_profile_line = ''
                self.OneD_profile_peaks.remove()
                self.OneD_profile_peaks = ''
                self.tab1_figfft_canvas.draw()
            self.OneD_profile_on = False
    
    def Draw_oneD_profile(self):
        try:
            self.OneD_profile_line_scale = float(self.tab1_text_col1['1D_scale'].text())
            current_fft_oneD = np.average(self.current_img_fft_amp_rotated, axis=1)
            current_fft_oneD_scaled = self.OneD_profile_line_scale*current_fft_oneD*self.current_img_fft_amp_rotated.shape[0]/current_fft_oneD.max()
            y = np.arange(0,len(current_fft_oneD))

            peaks, _ = signal.find_peaks(current_fft_oneD_scaled)
            peaks_x = np.ones_like(peaks)*self.current_img_fft_amp_rotated.shape[1]/2

            if self.OneD_profile_line == '':
                self.OneD_profile_line, = self.tab1_axfft.plot(current_fft_oneD_scaled, y, color='w', linewidth=0.5)
                self.OneD_profile_peaks, = self.tab1_axfft.plot(peaks_x, peaks,'x', color='w', markersize=3)
            else:
                self.OneD_profile_line.set_data(current_fft_oneD_scaled, y)
                self.OneD_profile_peaks.set_data(peaks_x, peaks)
            self.tab1_figfft_canvas.draw()
        except:
            return

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
            n_LL = 41

            x_label = 2*self.current_img_fft_amp.shape[1]/5

            for i in range(1, n_LL):
                y = self.origin[1] + i*self.LL_distance
                if y <= self.tab1_axfft.get_ylim()[1]:
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
            self.tab2_buttons['Clear'].setEnabled(True)
            self.tab2_buttons['Refine'].setEnabled(True)
            self.tab2_buttons['Undo'].setEnabled(True)

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

    def set_angpix(self):
        try:
            angpix_old = self.angpix
            self.angpix = float(self.tab1_text_col1['Angpix'].text())
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
            self.contrast_low = n
            if self.contrast_high <= self.contrast_low:
                self.contrast_high = self.contrast_low + 0.5
                self.maxSigma_chooser.setValue(self.contrast_high)
        elif id == 'high':
            self.contrast_high = n
            if self.contrast_high <= self.contrast_low:
                self.contrast_low = self.contrast_high - 0.5
                self.minSigma_chooser.setValue(self.contrast_low)
        min = round(self.current_img_fft_amp_rotated.mean() + self.contrast_low*self.current_img_fft_amp_rotated.std(), 1)
        max = round(self.current_img_fft_amp_rotated.mean() + self.contrast_high*self.current_img_fft_amp_rotated.std(), 1)
        self.tab1_fft_shown.set_clim(min, max)
        self.tab1_figfft_canvas.draw()
    
    def toggle_image_cmap(self):
        if self.img_cmap_toggle.isChecked():
            self.img_cmap = 'Greys'
        else:
            self.img_cmap = 'Greys_r'
        self.tab1_fft_shown.set_cmap(self.img_cmap)
        self.tab1_figfft_canvas.draw()
        self.tab2_fft_shown.set_cmap(self.img_cmap)
        self.tab2_figfft_canvas.draw()
    
    def calc_LL_plot(self):
        try:
            self.n_Bessel = int(self.tab1_text_col2['Bessel_order'].text())
            Y = self.tab1_text_col2['LL_Y_range'].text().split(',')
            self.radius_H = float(self.tab1_text_col2['helix_radius'].text())
            if self.radius_H <= 0:
                QMessageBox.information(self, 'Alert', 'To calculate layerline plot, \nHelix radius must be larger than 0!')
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
            amp_data = amp_data - amp_data.min()

            if self.angpix == 0:
                QMessageBox.information(self,'Alert', 'Angpix = 0')
                Bessel_data = np.zeros_like(amp_data)
            else:
                k = 2*math.pi*self.radius_H*(self.X_index/(self.current_img_fft_amp_rotated.shape[1]*self.angpix))
                Bessel_data = np.abs(special.jv(self.n_Bessel, k))
                scale = amp_data.max()/Bessel_data.max()
                Bessel_data = Bessel_data*scale
            cc_Bessel_vs_data = (np.corrcoef(Bessel_data, amp_data))[0,1]
            self.tab1_labels_col2['CC='].setText(f'CC={cc_Bessel_vs_data:.2f}')
    
            if self.LL_amp_plot == '':
                self.LL_amp_plot, = self.ax_amp.plot(self.X_index, amp_data, color='b')
                self.LL_bessel_plot, = self.ax_amp.plot(self.X_index, Bessel_data, color='r')
                self.LL_legend = self.ax_amp.legend((self.LL_amp_plot, self.LL_bessel_plot), ('Data', 'Predict'), loc='upper right', fontsize=8)
            else:
                self.LL_amp_plot.set_data(self.X_index, amp_data)
                self.LL_bessel_plot.set_data(self.X_index, Bessel_data)
            self.ax_amp.set_xlim(self.X_index[0], self.X_index[-1])
            self.ax_amp.set_ylim(0, amp_data.max()*1.05)
            
            X_original_index = self.X_index + int(self.current_img_fft_amp_rotated.shape[1]/2)
            phase_data = self.current_img_fft_phase[Y_phase, X_original_index[0]:X_original_index[-1]+1]
            phase_data = 180*phase_data/np.pi
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

    def change_displayed_slice(self, value):
        if type(self.mrc_data_array) == np.ndarray:
            self.current_slice = value - 1
            self.draw_2d_image()
            self.check_LL_plots_inputs()

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
        self.tab2_buttons['Undo'].setEnabled(True)
    
        self.calculate_rs_para()
        x_high = self.x_v1_rs*self.v1_n + self.x_v2_rs*self.v2_n
        x_low = -x_high*0.002
        x_high = x_high*1.002
        y_high = 5*(abs(self.y_v1_rs) + abs(self.y_v2_rs))
        self.text_fields_rs[0].setText(f'{x_low:.1f},{x_high:.1f}')
        self.text_fields_rs[1].setText(f'-0.2,{y_high:.1f}')
        self.update_real_space_plot()

        offset_new = f([self.x_v1])
        print("******************************************************")
        print(f'Residual after refinement:{offset_new:7.2f}')
        if round(offset_new,2) == 0:
            print("Looks good!\n\nIf lattice points do not match diffraction peaks,")
            print("Adjust the two base vectors and refine again.\n")
            self.tab2_buttons['Refine'].setText('Refine \U0001f642')
            self.paramSaveButton.setEnabled(True)
            for i in self.buttons_rs:
                self.buttons_rs[i].setEnabled(True)
        else:
            print("Not so good. Adjust the two base vectors and refine again.\n")
            self.tab2_buttons['Refine'].setText('Refine \U0001F61F')

    def undo_refine(self):
        try:
            self.x_v1 = self.x_v1_old
            self.x_v2 = self.x_v2_old
            for i in self.text_fields_rs:
                i.setText('')
            self.draw_tab2_lattice_lines()
            self.tab2_buttons['Refine'].setText('Refine')
            self.update_real_space_plot()
            self.tab2_buttons['Undo'].setEnabled(False)
            for i in self.buttons_rs:
                self.buttons_rs[i].setEnabled(False)
        except:
            return

    def draw_ori_unitV(self):
        if self.ori_unitV_lines_rs == []:
            self.ori_unitV_lines_rs.append(self.ax_rs.plot([0, self.x_v1_rs], [0, self.y_v1_rs], color='b'))
            self.ori_unitV_lines_rs.append(self.ax_rs.plot([0, self.x_v2_rs], [0, self.y_v2_rs], color='b'))
        else:
            self.ori_unitV_lines_rs[0][0].set_data([0, self.x_v1_rs], [0, self.y_v1_rs])
            self.ori_unitV_lines_rs[1][0].set_data([0, self.x_v2_rs], [0, self.y_v2_rs])

    def set_real_space_plot_xylim(self):
        self.x_axis_rs_low = 0
        self.x_axis_rs_high = self.x_v1_rs*self.v1_n + self.x_v2_rs*self.v2_n 
        self.x_axis_rs_low_ex = self.x_axis_rs_low - 0.1*(self.x_axis_rs_high - self.x_axis_rs_low)
        self.x_axis_rs_high_ex = self.x_axis_rs_high + 0.1*(self.x_axis_rs_high - self.x_axis_rs_low)
        self.y_axis_rs_low = -1*(abs(self.y_v1_rs) + abs(self.y_v2_rs))
        self.y_axis_rs_high = 5*(abs(self.y_v1_rs) + abs(self.y_v2_rs))
        
        self.ax_rs.set_xlim(self.x_axis_rs_low_ex, self.x_axis_rs_high_ex)
        self.ax_rs.set_ylim(self.y_axis_rs_low, self.y_axis_rs_high)
        self.ax_rs.spines['left'].set_position('zero')
        self.ax_rs.spines['bottom'].set_position('zero')
        self.ax_rs.spines['right'].set_position(('data', self.x_axis_rs_high))
        self.ax_rs.spines['top'].set_color('none')
        self.ax_rs.spines['bottom'].set_bounds(low=0, high=self.x_axis_rs_high)
        
        x_tick_pos = [0, self.x_axis_rs_high/4, self.x_axis_rs_high/2, 3*self.x_axis_rs_high/4, self.x_axis_rs_high]
        x_tick_txt = ['0', '90', '180', '270', '360']
        self.ax_rs.set_xticks(x_tick_pos)
        self.ax_rs.set_xticklabels(x_tick_txt)
        self.ax_rs.format_coord = lambda x, y: f'x={x:.1f}, y={y:.1f}' 

        self.fig_rs_canvas.draw()
    
    def add_real_space_point_group(self):
        try:
            x_range = list(map(float,(self.text_fields_rs[0].text()).split(',')))
            y_range = list(map(float,(self.text_fields_rs[1].text()).split(',')))

            if y_range[1] > self.y_axis_rs_high:
                y_range[1] = self.y_axis_rs_high
            if y_range[0] < self.y_axis_rs_low:
                y_range[0] = self.y_axis_rs_low

            for i in range(-50, 200):
                for j in range(-50, 200):
                    x = self.x_v1_rs*i + self.x_v2_rs*j
                    y = self.y_v1_rs*i + self.y_v2_rs*j
                    if x_range[0] <= x <= x_range[1] and y_range[0] <= y <= y_range[1]:
                        if [i, j] not in self.dots_rs[:, 0:2].tolist():
                            self.dots_rs = np.append(self.dots_rs, [[i, j, x, y]], axis=0)
            self.draw_real_space_point()
        except:
            QMessageBox.information(self, 'Error', 'Check "X range" and "Y range"!\nExample: -1, 10')

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
                name = f'({i[0]:.0f},{i[1]:.0f})'
                self.dots_rs_label[name] = self.ax_rs.annotate(name, (i[2], i[3]))
        self.fig_rs_canvas.draw()

    def real_space_seq_toggle(self):
        if len(self.dots_rs) > 0 and self.dots_rs_seq == []:
            self.redraw_real_space_point_seq()
            self.clear_real_space_point_labels()
        elif len(self.dots_rs) > 0 and self.dots_rs_seq != []:
            self.clear_real_space_point_seq()

    def redraw_real_space_point_seq(self):
        try:
            dots_sorted = self.dots_rs[np.lexsort((self.dots_rs[:, 0], self.dots_rs[:,3]), axis=0)]
            tmp_list = dots_sorted[:, 0:2].tolist()
            if ([self.v1_n, self.v2_n] not in tmp_list) or self.v1_n == 0 or self.v2_n ==0:
                QMessageBox.information(self,'Alert','Two points on equator must be present!')
                return
            index1 = tmp_list.index([0,0]) 
            index2 = tmp_list.index([self.v1_n, self.v2_n])
            if dots_sorted[index1,3] <= dots_sorted[index2,3]:
                index_00 = index1
            else:
                index_00 = index2
            offset = 1000000
            for i in range(1,len(dots_sorted)):
                diff = dots_sorted[i,3] - dots_sorted[i-1,3]
                if diff < offset:
                    offset = round(diff,2)
            seq_id = 0
            self.dots_rs_seq.append(self.ax_rs.annotate(' 0', dots_sorted[index_00, 2:4]))
            for i in range(index_00+1, len(dots_sorted)):
                if round(dots_sorted[i, 3] - dots_sorted[i-1, 3], 2) > offset:
                    seq_id += 1
                self.dots_rs_seq.append(self.ax_rs.annotate(' '+str(seq_id),dots_sorted[i,2:4]))
            self.fig_rs_canvas.draw()
        except:
            QMessageBox.information(self, 'Error', 'Something wrong!\n At least two points neede.\n (0,0) point needed.')
        
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
                coors = self.match_click_with_points(self.click_coor)
                self.strand_line_rs.append((self.ax_rs.plot(coors[0], coors[1], ':', color='m', linewidth=1))[0])
                self.fig_rs_canvas.draw()
                self.click_count =0
                self.click_coor = []
                self.cursor_annotate_rise_twist()

    def match_click_with_points(self, coor):
        # Find the closest points and extension
        _, index1 = spatial.KDTree(self.dots_rs[:, 2:]).query(coor[0])
        _, index2 = spatial.KDTree(self.dots_rs[:, 2:]).query(coor[1])
        if index1 == index2:
            index2 = index1 + 1
        pc1 = self.dots_rs[index1, 2:]
        pc2 = self.dots_rs[index2, 2:]
        slope =  (pc1[1] - pc2[1])/(pc1[0] - pc2[0])
        x1 = 0
        y1 = pc1[1] - pc1[0]*slope
        x2 = self.x_axis_rs_high 
        y2 = x2*slope + y1

        self.rise_rs = abs(pc1[1] - pc2[1])
        if self.rise_rs != 0:
            if pc2[1] > pc1[1]:
                self.twist_rs = (pc2[0] - pc1[0])*360/self.circum_rs
            else:
                self.twist_rs = (pc1[0] - pc2[0])*360/self.circum_rs

            dot_strand_dist_array = []
            for i in range(len(self.dots_rs)):
                if i != index1 and i != index2:
                    dist_h = abs((self.dots_rs[i, 3] - y1)/slope - self.dots_rs[i, 2])
                    if dist_h > self.circum_rs/500:
                        dot_strand_dist_array.append(dist_h)
            dot_strand_dist_array = sorted(dot_strand_dist_array)
            self.n_start = int(round(self.circum_rs/dot_strand_dist_array[0]))
        else:
            self.n_start = 0

        self.strand_line_info.append([self.rise_rs, self.twist_rs, self.n_start])
        self.labels_rs['Rise_Twist'].setText(f'Rise: {self.rise_rs:.2f} \u212B; Twist: {self.twist_rs:.2f}\u00B0; {self.n_start}-start')
        return [[x1, x2], [y1, y2]]

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
            sel.annotation.set(text=f'Rise={rise:.2f} \u212B\nTwist={twist:.2f}\u00B0\n{n_start}-start helix')
            sel.annotation.arrow_patch.set(arrowstyle='simple',fc='yellow', alpha=0.5)

    def update_real_space_plot(self):
        self.clear_real_space_points()
        self.delete_all_strand_lines()
        self.calculate_rs_para()
        self.draw_ori_unitV()
        self.set_real_space_plot_xylim()

        if [0,0] not in self.dots_rs[:,0:2].tolist():
            self.dots_rs = np.append(self.dots_rs,[[0,0,0,0]],axis=0)
            self.draw_real_space_point()    
        if self.text_fields_rs[0].text() != '' and self.text_fields_rs[1].text() != '':
            self.add_real_space_point_group()
    
    def update_tab2_fft_upon_tab_switch(self, i):
        if i == 1:
            self.draw_tab2_fft()
    
    def button_push_connect(self):
        self.minSigma_chooser.valueChanged.connect(functools.partial(self.change_contrast, 'low'))
        self.maxSigma_chooser.valueChanged.connect(functools.partial(self.change_contrast, 'high'))
        self.slice_chooser.valueChanged.connect(self.change_displayed_slice)
        self.img_rotation_chooser.valueChanged.connect(self.set_img_rotation)
        self.img_cmap_toggle.toggled.connect(self.toggle_image_cmap)
        self.tab1_buttons_col1['Set Angpix'].clicked.connect(self.set_angpix)
        self.tab1_text_col1['Angpix'].returnPressed.connect(self.set_angpix)
        self.tab1_buttons_col1['1D profile'].clicked.connect(self.OneD_profile_toggle)
        self.tab1_text_col1['1D_scale'].returnPressed.connect(self.Draw_oneD_profile)
        self.tab1_buttons_col1['Draw LL'].clicked.connect(self.toggle_draw_tab1_LL)
        self.tab1_buttons_col1['Set LL dist'].clicked.connect(self.draw_tab1_LL)
        self.tab1_text_col1['Y_dist'].returnPressed.connect(self.draw_tab1_LL)
        self.tab1_buttons_col2['Measure'].clicked.connect(self.measure_distance)
        self.tab1_buttons_col2['Calc LL plot'].clicked.connect(self.calc_LL_plot)
        self.tab1_text_col2['helix_radius'].returnPressed.connect(self.check_LL_plots_inputs)
        self.tab1_text_col2['LL_Y_range'].returnPressed.connect(self.check_LL_plots_inputs)
        self.tab1_text_col2['LL_width'].returnPressed.connect(self.check_LL_plots_inputs)
        self.tab1_text_col2['Bessel_order'].returnPressed.connect(self.check_LL_plots_inputs)

        self.tab2_symmetrize_fft_switch.toggled.connect(self.draw_tab2_fft)
        self.tab2_buttons['Draw lattice'].clicked.connect(self.draw_tab2_lattice_lines)
        self.tab2_buttons['Clear'].clicked.connect(self.clear_tab2_lattice_lines)
        self.tab2_buttons['Refine'].clicked.connect(self.opt_para)
        self.tab2_buttons['Undo'].clicked.connect(self.undo_refine)
        self.buttons_rs['Add points:'].clicked.connect(self.add_real_space_point_group)
        self.buttons_rs['Clear points'].clicked.connect(self.clear_real_space_points)
        self.buttons_rs['Label On/Off'].clicked.connect(self.real_space_label_toggle)
        self.buttons_rs['Seq. On/Off'].clicked.connect(self.real_space_seq_toggle)
        self.buttons_rs['Delete last'].clicked.connect(self.delete_last_strand_line)
        self.buttons_rs['Delete all'].clicked.connect(self.delete_all_strand_lines)
        self.draw_strand_switch.stateChanged.connect(self.strand_line_toggle)
        self._CentralWidget.currentChanged.connect(self.update_tab2_fft_upon_tab_switch)

def main():
    app = QApplication(sys.argv)
    gui = MainWindow()
    gui.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
