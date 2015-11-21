# generate test result from file_list
import caffe
import numpy as np
import scipy
import argparse

parser = argparse.ArgumentParser(description='caffe predict code')
parser.add_argument('caffe_dir', help='caffe directory')
parser.add_argument('imageFileList', help='image file list')
parser.add_argument('save1', help='save all predict score for each class')
parser.add_argument('save2', help='save predict label file')
parser.add_argument('save3', help='save the predict result: zaoyin(0) or not(1)')
parser.add_argument('deploy', help='deploy.prototxt file')
parser.add_argument('caffemodel', help='load caffe model')
args = parser.parse_args()


#read image list from file, one image one line.
dim = 1
caffe_root = args.caffe_dir
import sys
sys.path.insert(0, caffe_root + '/python/')
MODEL_FILE = args.deploy #'/mnt_data/catherine/caffe/models/animal_6class_googlenet/deploy.prototxt'
PRETRAINED = args.caffemodel #'/mnt_data/catherine/caffe/models/animal_6class_googlenet/b16_s4000/animal_6class_googlenet_iter_100000.caffemodel'

# set gpu id
caffe.set_device(0)
caffe.set_mode_gpu()

net = caffe.Classifier(MODEL_FILE, PRETRAINED,
                       mean=np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1),
                       channel_swap=(2,1,0),
                       raw_scale=255,
                       image_dims=(224, 224))

image_list_file = open(args.imageFileList)
image_list = image_list_file.readlines()
n_image = len(image_list)

output1 = open(args.save1, "w")
output2 = open(args.save2, "w")
output3 = open(args.save3, 'w')
imageList = []
imageList_name = []
imageList_label = []
for i in range(0, n_image):
        IMAGE_FILE = image_list[i]
        name_label = IMAGE_FILE.split(' ', 1)
        IMAGE_FILE = name_label[0]
        label = name_label[1]
	
	try:
		imageList.append(caffe.io.load_image(IMAGE_FILE))
		imageList_name.append(IMAGE_FILE)
		imageList_label.append(label)
	except IOError:
		print "IO error"
	except ValueError: 
		print "Value error"


	if ((i+1)%dim == 0 or i == n_image -1) and len(imageList) != 0:
		#print len(imageList), len(imageList_name), len(imageList_label)
		scores = net.predict(imageList, oversample=False)
		#print len(scores)
		for j, score in enumerate(scores):
	                print imageList_name[j], str(scores[j].argmax()), str(imageList_label[j])
                        string = imageList_name[j] + " "
                        for score_i in score:
                            string += str(score_i) + " "
	                output1.write(string +  str(imageList_label[j]))
                        output2.write(imageList_name[j]  +" " +  str(score.argmax()) +" " +  str(imageList_label[j]))
                        predict_id = 0
                        if(score.argmax() != 0):
                            predict_id = 1
                        true_id = 0
                        if(imageList_label[j] != 0):
                            true_id = 1
                        output3.write(imageList_name[j] + " " + str(predict_id) + " " + str(true_id) + "\n")
		imageList = []
		imageList_name = []
		imageList_label = []
output1.close()
output2.close()
output3.close()
