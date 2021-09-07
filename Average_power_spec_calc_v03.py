#!/usr/bin/python3

import mrcfile
import numpy as np
from numpy.core.fromnumeric import prod
from scipy import ndimage
import matplotlib.pyplot as plt
import glob
import re
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input_star', required=1, type=str, help='input particle star file')
parser.add_argument('-d', '--particle_dir', required=1, type=str, help='diretory of particle stack mrcs files')
parser.add_argument('-p', '--pad_factor', default=1, type=int, help='pad 2D image with zeros by this factor')
parser.add_argument('-o', '--oversample_factor', default=1, type=int, help='oversample FFT by this factor')
parser.add_argument('-s', '--symmetrize', default=1, type=int, choices=[0, 1], help='Set to 0 to not symmetrize the power spectrum')
args = parser.parse_args()

star_file_name = args.input_star
particle_stack_dir = args.particle_dir
pad_factor = args.pad_factor
oversample_factor = args.oversample_factor

output_mrc = star_file_name.split('.')[0]
output_mrc = f'{output_mrc}_pad{pad_factor}_oversample{oversample_factor}_average_power_spec.mrc'

fft_stack = 0
particle_dic = {}
total_particle_count = 0
ang_pix = 1

with open(star_file_name) as f:
    for line in f.readlines():
        if '_rlnAnglePsi ' in line:
            angle_field_number = int(line.split()[1][1:]) - 1
        if '_rlnImagePixelSize' in line:
            ang_pix_field_number = int(line.split()[1][1:]) - 1
        if 'opticsGroup1' in line:
            ang_pix = float(line.split()[ang_pix_field_number])
        elif '.mrc' in line:
            tmp_name = re.search(r'@(.+\.mrc)s', line)
            tmp_name = tmp_name.group(1)
            mrc_base_name = tmp_name.split('/')[-1].split('.')[0]
            rotation_angle = -float(line.split()[angle_field_number]) - 90
            for field in line.split():
                if '@' in field:
                    slice_number = int(field.split('@')[0]) - 1
            if mrc_base_name not in particle_dic:
                particle_dic[mrc_base_name] = []
            particle_dic[mrc_base_name].append([slice_number, rotation_angle])
            total_particle_count += 1

process_count = 0
particle_stack_files = particle_stack_dir + '/*.mrcs'
for mrcs in glob.glob(particle_stack_files):
    mrc_base_name = mrcs.split('/')[-1].split('.')[0]
    if mrc_base_name in particle_dic:
        print(f'\n{"*"*40}\nProcessing particles in {mrcs}...')
        particles_selected = particle_dic[mrc_base_name]
        with mrcfile.open(mrcs, permissive=True) as f:
            img_data = f.data 
            if img_data.ndim == 2:
                img_data = img_data.reshape(1, img_data.shape[0], img_data.shape[1])
            x_dim = img_data.shape[2]
            y_dim = img_data.shape[1]
            pad_x = int(x_dim*(pad_factor-1)/2)
            pad_y = int(y_dim*(pad_factor-1)/2)

        for i, rotation_angle in particles_selected:
            img_slice_padded = np.pad(img_data[i], ((pad_y, pad_y), (pad_x, pad_x)), mode='constant')
            img_slice_rotated = ndimage.rotate(img_slice_padded, rotation_angle, reshape=False)
            fft_slice_amp = np.abs(np.fft.fftshift(np.fft.fft2(img_slice_rotated)))
            if oversample_factor > 1:
                fft_slice_amp = ndimage.zoom(fft_slice_amp, oversample_factor)
    
            process_count += 1
            if type(fft_stack) is np.ndarray:
                fft_stack = fft_stack + fft_slice_amp
            else:
                fft_stack = fft_slice_amp
            print(f'Processed {process_count} out of {total_particle_count} particles in the star file')

fft_average = fft_stack/process_count

if args.symmetrize == 1:
    ydim = fft_average.shape[0] 
    fft_sym = (fft_average[1::1, 1::1] + fft_average[-1:0:-1, 1::1] + fft_average[1::1, -1:0:-1] + fft_average[-1:0:-1, -1:0:-1])/4
    fft_sym = np.vstack((fft_average[0, 1::], fft_sym))
    fft_sym = np.hstack((fft_average[:, 0].reshape(ydim, 1), fft_sym))
    fft_average = fft_sym

x_ang = ang_pix*fft_average.shape[1]
y_ang = ang_pix*fft_average.shape[0]
with mrcfile.new(output_mrc, overwrite=True) as f:
    f.set_data(fft_average.astype('float32'))
    f.header.cella = (x_ang, y_ang, 0)
    print(f'\nSaved average power spectrum for {process_count} particles')

fig, ax = plt.subplots()
ax.imshow(fft_average)
plt.tight_layout()
plt.show()