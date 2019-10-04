from PIL import Image
import glob
import pdb
import numpy as np
import cv2
import os


model_name = 'birds'

directorio = ('../models/%s/outputs'%model_name)

for folder in glob.glob('*'):

	image_list = []
	deconv_list = []
	number_of_dirs = 0

	try:
		number_of_dirs = len(os.walk('%s/%s'%(directorio,folder)).next()[1])
    	except StopIteration:
        	print("end")
	for x in range(0,number_of_dirs):

		for image in range(0,9):

			filename_img = ('%s/%s/unit_%04d/maxim_00%d.png' % (directorio,folder,x,image))
			img = cv2.imread(filename_img)
			image_list.append(img)
			filename_deconv = ('%s/%s/unit_%04d/deconv_00%d.png' % (directorio,folder,x,image))			
			deconv = cv2.imread(filename_deconv)
			deconv_list.append(deconv)

		aux1 = np.concatenate((image_list[0], image_list[1], image_list[2]), axis=1)
		aux2 = np.concatenate((image_list[3], image_list[4], image_list[5]), axis=1)
		aux3 = np.concatenate((image_list[6], image_list[7], image_list[8]), axis=1)
		out = np.concatenate((aux1,aux2,aux3), axis=0)
		cv2.imwrite(('%s/%s/unit_%04d/maxim.jpg' % (directorio,folder,x)), out)
		del image_list[:]
		aux1_d = np.concatenate((deconv_list[0], deconv_list[1], deconv_list[2]), axis=1)
		aux2_d = np.concatenate((deconv_list[3], deconv_list[4], deconv_list[5]), axis=1)
		aux3_d = np.concatenate((deconv_list[6], deconv_list[7], deconv_list[8]), axis=1)
		out_d = np.concatenate((aux1_d,aux2_d,aux3_d), axis=0)
		cv2.imwrite(('%s/%s/unit_%04d/deconv.jpg' % (directorio,folder,x)), out_d)
		del deconv_list[:]



