#Histogramas de entropia

#export PYTHONPATH=/home/mbm/Desktop/Visualizadores/VisualizadorModif/deep-visualization-toolbox/caffe/python/:$PYTHONPATH
	    
import numpy as np
import matplotlib.pyplot as plt
import sys
import caffe
import scipy.misc
import pdb
import os
import scipy.io
np.set_printoptions(threshold=np.inf)

MODEL_FILE = 'models/alexnet/deploy_alexnet_places365.prototxt'
PRETRAINED = 'models/alexnet/alexnet_places365.caffemodel'

net = caffe.Classifier(MODEL_FILE, PRETRAINED,
	mean=np.load('models/alexnet/places365CNN_mean.npy').mean(1).mean(1),
	channel_swap=(2,1,0),
	raw_scale=255.0)

print "successfully loaded classifier"

capas = ['data','conv1','pool1','norm1','conv2','pool2','norm2','conv3','conv4','conv5','pool5','fc6','fc7','fc8','prob']

aux = ('/home/mbm/Desktop/Aux/input_images/val_256/Places365_val_00000001.jpg')
aux_image = caffe.io.load_image(aux)
pred = net.predict([aux_image])


limite_inf = 0
limite_sup_conv1 = 954.6
limite_sup_conv2 = 208.3
limite_sup_conv3 = 160.9
limite_sup_conv4 = 96.99
limite_sup_conv5 = 106.8
limite_sup_fc6 = 61.9
limite_sup_fc7 = 15.2

region_conv1 = np.arange(0,limite_sup_conv1+1,limite_sup_conv1/100)
region_conv2 = np.arange(0,limite_sup_conv2+1,limite_sup_conv2/100)
region_conv3 = np.arange(0,limite_sup_conv3+1,limite_sup_conv3/100)
region_conv4 = np.arange(0,limite_sup_conv4+0.5,limite_sup_conv4/100)
region_conv5 = np.arange(0,limite_sup_conv5+1,limite_sup_conv5/100)
region_fc6 = np.arange(0,limite_sup_fc6+0.6,limite_sup_fc6/100)
region_fc7 = np.arange(0,limite_sup_fc7+0.1,limite_sup_fc7/100)

#intervalo = 10
#region = range(0,1501,10)
#region2 = range(0,31)
#region3 = range(0,21)
histograma_conv1 = np.zeros((net.blobs['conv1'].data.shape[1],101))
histograma_conv2 = np.zeros((net.blobs['conv2'].data.shape[1],101))
histograma_conv3 = np.zeros((net.blobs['conv3'].data.shape[1],101))
histograma_conv4 = np.zeros((net.blobs['conv4'].data.shape[1],101))
histograma_conv5 = np.zeros((net.blobs['conv5'].data.shape[1],101))
histograma_fc6 = np.zeros((net.blobs['fc6'].data.shape[1],101))
#histograma2_fc6 = np.zeros((net.blobs['fc6'].data.shape[1],30))
histograma_fc7 = np.zeros((net.blobs['fc7'].data.shape[1],101))
#histograma2_fc7 = np.zeros((net.blobs['fc7'].data.shape[1],20))


out_conv1=0
out_conv2=0
out_conv3=0
out_conv4=0
out_conv5=0
out_fc6=0
out_fc6_2=0
out_fc7=0
out_fc7_2=0
#36501
for image in range (1,1000):
	IMAGE_FILE = ('/home/mbm/Desktop/Aux/input_images/val_256/Places365_val_%08d.jpg'%image)
	input_image = caffe.io.load_image(IMAGE_FILE)
	pred = net.predict([input_image])

	#conv1

	for neurona in range (0,net.blobs['conv1'].data.shape[1]):
	#for neurona in range(94,95):	
		for reg in range(0,101):
			if reg>100:
				out_conv1 = out_conv1+1
				break
			if (np.mean(net.blobs['conv1'].data[0][neurona])>=region_conv1[reg] and np.mean(net.blobs['conv1'].data[0][neurona])<region_conv1[reg+1]):
				histograma_conv1[neurona][reg] = histograma_conv1[neurona][reg]+1	
				break	

	for neurona in range (0,net.blobs['conv2'].data.shape[1]):
	#for neurona in range(94,95):	
		for reg in range(0,101):
			if reg>100:
				out_conv2 = out_conv2+1
				break
			if (np.mean(net.blobs['conv2'].data[0][neurona])>=region_conv2[reg] and np.mean(net.blobs['conv2'].data[0][neurona])<region_conv2[reg+1]):
				histograma_conv2[neurona][reg] = histograma_conv2[neurona][reg]+1	
				break	

	for neurona in range (0,net.blobs['conv3'].data.shape[1]):
	#for neurona in range(94,95):	
		for reg in range(0,101):
			if reg>100:
				out_conv3 = out_conv3+1
				break
			if (np.mean(net.blobs['conv3'].data[0][neurona])>=region_conv3[reg] and np.mean(net.blobs['conv3'].data[0][neurona])<region_conv3[reg+1]):
				histograma_conv3[neurona][reg] = histograma_conv3[neurona][reg]+1	
				break	
				
	for neurona in range (0,net.blobs['conv4'].data.shape[1]):
	#for neurona in range(94,95):	
		for reg in range(0,101):
			if reg>100:
				out_conv4 = out_conv4+1
				break
			if (np.mean(net.blobs['conv4'].data[0][neurona])>=region_conv4[reg] and np.mean(net.blobs['conv4'].data[0][neurona])<region_conv4[reg+1]):
				histograma_conv4[neurona][reg] = histograma_conv4[neurona][reg]+1	
				break
	
	for neurona in range (0,net.blobs['conv5'].data.shape[1]):
	#for neurona in range(94,95):	
		for reg in range(0,101):
			if reg>100:
				out_conv5 = out_conv5+1
				break
			if (np.mean(net.blobs['conv5'].data[0][neurona])>=region_conv5[reg] and np.mean(net.blobs['conv5'].data[0][neurona])<region_conv5[reg+1]):
				histograma_conv5[neurona][reg] = histograma_conv5[neurona][reg]+1	
				break

	for neurona in range (0,net.blobs['fc6'].data.shape[1]):
	#for neurona in range(94,95):	
		for reg in range(0,101):
			if reg>100:
				out_fc6 = out_fc6+1
				break
			if ((net.blobs['fc6'].data[0][neurona])>=region_fc6[reg] and (net.blobs['fc6'].data[0][neurona])<region_fc6[reg+1]):
				histograma_fc6[neurona][reg] = histograma_fc6[neurona][reg]+1	
				break

		#for reg in range(0,31):
		#	if reg>29:
		#		out_fc6_2 = out_fc6_2+1
		#		break
		#	if ((net.blobs['fc6'].data[0][neurona])>=region2[reg] and (net.blobs['fc6'].data[0][neurona])<region2[reg+1]):
		#		histograma2_fc6[neurona][reg] = histograma2_fc6[neurona][reg]+1	
		#		break

	for neurona in range (0,net.blobs['fc7'].data.shape[1]):
	#for neurona in range(94,95):	
		for reg in range(0,101):
			if reg>100:
				out_fc7 = out_fc7+1
				break
			if ((net.blobs['fc7'].data[0][neurona])>=region_fc7[reg] and (net.blobs['fc7'].data[0][neurona])<region_fc7[reg+1]):
				histograma_fc7[neurona][reg] = histograma_fc7[neurona][reg]+1	
				break

		#for reg in range(0,21):
		#	if reg>19:
		#		out_fc7_2 = out_fc7_2+1
		#		break
		#	if ((net.blobs['fc7'].data[0][neurona])>=region3[reg] and (net.blobs['fc7'].data[0][neurona])<region3[reg+1]):
		#		histograma2_fc7[neurona][reg] = histograma2_fc7[neurona][reg]+1	
		#		break
				
				
	print 'Places365_val_%08d.jpg'%image

print 'out_conv1 = %d' %out_conv1
print 'out_conv2 = %d' %out_conv2
print 'out_conv3 = %d' %out_conv3
print 'out_conv4 = %d' %out_conv4
print 'out_conv5 = %d' %out_conv5
print 'out_fc6 = %d' %out_fc6
print 'out_fc6_2 = %d' %out_fc6_2
print 'out_fc7 = %d' %out_fc7
print 'out_fc7_2 = %d' %out_fc7_2

						
os.makedirs('/home/mbm/Desktop/Aux/models/alexnet/activaciones_subset/conv1/entropia')
np.savetxt('/home/mbm/Desktop/Aux/models/alexnet/activaciones_subset/conv1/entropia/conv1_entropia',histograma_conv1,fmt='%d')

os.makedirs('/home/mbm/Desktop/Aux/models/alexnet/activaciones_subset/conv2/entropia')
np.savetxt('/home/mbm/Desktop/Aux/models/alexnet/activaciones_subset/conv2/entropia/conv2_entropia',histograma_conv2,fmt='%d')

os.makedirs('/home/mbm/Desktop/Aux/models/alexnet/activaciones_subset/conv3/entropia')
np.savetxt('/home/mbm/Desktop/Aux/models/alexnet/activaciones_subset/conv3/entropia/conv3_entropia',histograma_conv3,fmt='%d')

os.makedirs('/home/mbm/Desktop/Aux/models/alexnet/activaciones_subset/conv4/entropia')
np.savetxt('/home/mbm/Desktop/Aux/models/alexnet/activaciones_subset/conv4/entropia/conv4_entropia',histograma_conv4,fmt='%d')

os.makedirs('/home/mbm/Desktop/Aux/models/alexnet/activaciones_subset/conv5/entropia')
np.savetxt('/home/mbm/Desktop/Aux/models/alexnet/activaciones_subset/conv5/entropia/conv5_entropia',histograma_conv5,fmt='%d')

os.makedirs('/home/mbm/Desktop/Aux/models/alexnet/activaciones_subset/fc6/entropia')
np.savetxt('/home/mbm/Desktop/Aux/models/alexnet/activaciones_subset/fc6/entropia/fc6_entropia',histograma_fc6,fmt='%d')
#np.savetxt('/home/mbm/Desktop/Aux/models/alexnet/activaciones/fc6/entropia/fc6_entropia2',histograma2_fc6,fmt='%d')

os.makedirs('/home/mbm/Desktop/Aux/models/alexnet/activaciones_subset/fc7/entropia')
np.savetxt('/home/mbm/Desktop/Aux/models/alexnet/activaciones_subset/fc7/entropia/fc7_entropia',histograma_fc7,fmt='%d')
#np.savetxt('/home/mbm/Desktop/Aux/models/alexnet/activaciones/fc7/entropia/fc7_entropia2',histograma2_fc7,fmt='%d')
