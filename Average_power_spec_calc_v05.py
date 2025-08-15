#!/usr/bin/python3

import mrcfile
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import glob
import re
import argparse
import multiprocessing

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input_star', required=1, type=str, help='input particle star file')
parser.add_argument('-d', '--particle_dir', required=1, type=str, help='diretory of particle stack mrcs files')
parser.add_argument('-p', '--pad_factor', default=1, type=int, help='pad 2D image with zeros by this factor')
parser.add_argument('-o', '--oversample_factor', default=1, type=int, help='oversample FFT by this factor')
parser.add_argument('-r', '--rotate', default=-90, type=float, help='rotate the paricle image by this angle')
parser.add_argument('-s', '--symmetrize', default=0, type=int, choices=[0, 1], help='Set to 1 to not symmetrize the power spectrum')
#symmetrizing the power spectrum would mess things up if the particles are not perfectly aligned, making powwer spectrum look symmetric and vertical
# although it is acutally not symmmetric
parser.add_argument('-j', '--processes', default=4, type=int, help='Number of processes to use for parallel processing')
args = parser.parse_args()

star_file_name = args.input_star
particle_stack_dir = args.particle_dir
pad_factor = args.pad_factor
oversample_factor = args.oversample_factor
rotation_angle = args.rotate
n_processes = args.processes

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
            # this angle should be adjust based on the orientation of the 2D class averge, if it is horizon, set --rotate to -90 
            # we can measure the angle in PyHI to find how much rotation is acutally needed to rotate the y-axis to align with the filament
            #(y-axis rotation in CCW direction is positive according to RELION convention)
            rotation_angle = -float(line.split()[angle_field_number]) + args.rotate
            for field in line.split():
                if '@' in field:
                    slice_number = int(field.split('@')[0]) - 1
            if mrc_base_name not in particle_dic:
                particle_dic[mrc_base_name] = []
            particle_dic[mrc_base_name].append([slice_number, rotation_angle])
            total_particle_count += 1


def process_particle(args):
    img, rotation_angle, pad_x, pad_y, oversample_factor = args
    img_slice_padded = np.pad(img, ((pad_y, pad_y), (pad_x, pad_x)), mode='constant')
    img_slice_padded = img_slice_padded.astype(np.float32)
    img_slice_rotated = ndimage.rotate(img_slice_padded, rotation_angle, reshape=False)
    fft_slice_amp = np.abs(np.fft.fftshift(np.fft.fft2(img_slice_rotated)))
    if oversample_factor > 1:
        fft_slice_amp = ndimage.zoom(fft_slice_amp, oversample_factor)
    return fft_slice_amp

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

        process_args = [
            (img_data[i], rotation_angle, pad_x, pad_y, oversample_factor)
            for i, rotation_angle in particles_selected
        ]

        with multiprocessing.Pool(processes=n_processes) as pool:
            results = pool.map(process_particle, process_args)

        for fft_slice_amp in results:
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