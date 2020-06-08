from PIL import Image
import os
# from libtiff import TIFF
import numpy as np
import ipdb
# import ipdb
# from osgeo import gdal
# import rasterio as rio
import shutil
import imageio
import imgaug as ia
import imgaug.augmenters as iaa
import Augmentor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
# from keras.utils.np_utils import to_categorical

class aug(object):
	def __init__(self):
		self.current_folder_path = os.path.dirname(os.path.abspath(__file__))
		self.data_dir_path = self.current_folder_path + '/data'
		self.train_dir_path = self.data_dir_path + '/train'
		self.aug_train_dir_path = self.data_dir_path + '/aug_train'
		self.aug_img_dir_path = self.aug_train_dir_path + '/aug_image'
		self.aug_label_dir_path = self.aug_train_dir_path + '/aug_label'
		self.img_dir_path = self.train_dir_path + '/image'
		self.img_play_dir = self.train_dir_path + '/image_play'
		self.img_glo_dir = self.train_dir_path + '/image_glo'
		self.img_glo_aug_dir = self.train_dir_path + '/image_glo_aug'
		self.label_play_dir = self.train_dir_path + '/label_play'
		self.label_dir_path = self.train_dir_path + '/label'
		self.label_glo_dir = self.train_dir_path + '/label_glo'
		self.label_glo_aug_dir = self.train_dir_path + '/label_glo_aug'
		self.results_dir_path = self.current_folder_path + '/results_by_name'
		self.results_dir_path_TG = self.current_folder_path + '/results_by_name_TG'
		self.results_from_training_dir_path = self.current_folder_path + '/results_from_training'
		self.test_img_dir_path = self.data_dir_path + '/test'
		self.test_labels_path = self.data_dir_path + '/test_labels'
		self.test_img_transform_path = self.data_dir_path + '/test_transform'
		self.test_labels_transform_path = self.data_dir_path + '/test_labels_transform'
		self.ground_truth_array_path = self.current_folder_path + '/' + 'ground_truth_array.npy'
		self.predicted_images_one_hot = self.current_folder_path + '/' + 'predicted_images_one_hot.npy'
	def visualise(self):
		_, _, files = next(os.walk(self.img_dir_path))
		image_path = self.img_dir_path + '/33.jpg'
		label_path = self.label_dir_path + '/33.png'
		image = imageio.imread(image_path)
		label = imageio.imread(label_path)
		# rotate = iaa.Affine(rotate=(45))
		# rotate = iaa.Rotate((90))
		aug = iaa.PerspectiveTransform(scale= (0.09, 0.12), mode='constant', seed=15)
		# aug = iaa.ElasticTransformation(alpha=(3), sigma=0.25)
		# aug = iaa.imgcorruptlike.GaussianNoise(severity=2)
		# aug = iaa.imgcorruptlike.Saturate(severity=4)
		# aug = iaa.imgcorruptlike.DefocusBlur(severity=2)
		# aug = iaa.Invert(1)
		# aug = iaa.MedianBlur(k=(11))
		# aug = iaa.imgcorruptlike.
		# aug = iaa.GammaContrast((2.0))
		# aug = iaa.SigmoidContrast(gain=(8), cutoff=(0.2), per_channel=True)
		# aug = iaa.Emboss(alpha=(1.0), strength=(0.2))
		# aug = iaa.pillike.Autocontrast((5,20), per_channel=True)
		augmented = aug.augment_image(image)
		augmented_label = aug.augment_image(label)
		# aug = iaa.imgcorruptlike.
		#Augmentor
		# p = Augmentor.Pipeline(self.img_play_dir)
		# p.random_distortion(1, 5, 5, 70)
		# p.process()
		# p = Augmentor.Pipeline(self.label_play_dir)
		# p.random_distortion(1, 5, 5, 70)
		# p.process()

		#############MAKING THE SAME AUGMENTATION FOR IMAGES AND MASKS AT ONCE!!!!!!!!!!!!!!!!!!!!!!!!!!
		seq = iaa.Sequential([aug])
		# label[label==170] = 1
		# label[label==250] = 2
		# label_arr = to_categorical(label, num_classes=3, dtype = 'uint8')
		img_arr = np.reshape(image, (1, image.shape[0], image.shape[1], 3))
		label_arr = np.reshape(label, (1, label.shape[0], label.shape[1], 1))
		# ipdb.set_trace()
		# image =
		aug_img, aug_label = seq(images = img_arr, segmentation_maps = label_arr)
		aug_img_back = np.reshape(aug_img, (image.shape[0], image.shape[1], 3))
		aug_label_back = np.reshape(aug_label, (label.shape[0], label.shape[1]))
		########################################################################

		# ia.imshow(p)
		ia.imshow(aug_img_back)
		ia.imshow(aug_label_back)
	def play(self):
		_, _, files = next(os.walk(self.label_dir_path))
		counter = 1
		for filename in files:
			# file_path = self.img_dir_path + '/' + filename
			# img_new_name = str(counter) + '.jpg'
			# new_image_path = self.train_dir_path + '/image_2/' + img_new_name
			# shutil.copyfile(file_path, new_image_path)
			label_path = self.label_dir_path + '/' + filename[:-4] + '.png'
			label_new_name = str(counter) + '.png'
			new_label_path = self.train_dir_path + '/label_2/' + label_new_name
			shutil.copyfile(label_path, new_label_path)
			counter += 1
	
	def img_rotate(self):
		_, _, glo_files = next(os.walk(self.img_glo_dir))
		_, _, img_files = next(os.walk(self.img_dir_path))
		#image_count - will be used to save new augmented images in a way that name of the first image
		#will be +1 comparing to the number of already existing images
		image_count = len(img_files)
		for filename in glo_files:
			image_path = self.img_glo_dir + '/' + filename
			image = imageio.imread(image_path)
			label_path = self.label_glo_dir + '/' + filename[:-4] + '.png'
			label = imageio.imread(label_path)
			rotate_angles = [90, 180, 270]
			for angle in rotate_angles:
				rotate = iaa.Affine(rotate=(angle))
				rotated_image = rotate.augment_image(image)
				rotated_label = rotate.augment_image(label)
				image_count += 1
				new_path_image = self.img_glo_aug_dir + '/' + str(image_count) + '.jpg'
				rotated_image = Image.fromarray(rotated_image.astype(np.uint8))
				rotated_image.save(new_path_image)
				new_path_label = self.label_glo_aug_dir + '/' + str(image_count) + '.png'
				rotated_label = Image.fromarray(rotated_label.astype(np.uint8))
				rotated_label.save(new_path_label)

#PERSPECTIVE TRANSFORMATION IS NOT WORKING THE SAME ON IMAGES AND LABELS!
	def perspective_transform(self, scale_range):
		_, _, glo_files = next(os.walk(self.img_glo_dir))
		_, _, img_files = next(os.walk(self.img_dir_path))
		# ipdb.set_trace()
		image_count = len(img_files)
		for filename in glo_files:
			image_path = self.img_glo_dir + '/' + filename
			image = imageio.imread(image_path)
			label_path = self.label_glo_dir + '/' + filename[:-4] + '.png'
			label = imageio.imread(label_path)
			aug = iaa.PerspectiveTransform(scale = scale_range, mode='constant', seed=15)
			seq = iaa.Sequential([aug])
			img_arr = np.reshape(image, (1, image.shape[0], image.shape[1], 3))
			label_arr = np.reshape(label, (1, label.shape[0], label.shape[1], 1))
			aug_img, aug_label = seq(images = img_arr, segmentation_maps = label_arr)
			aug_img_back = np.reshape(aug_img, (image.shape[0], image.shape[1], 3))
			aug_label_back = np.reshape(aug_label, (label.shape[0], label.shape[1]))
			image_count += 1
			new_path_image = self.img_glo_aug_dir + '/' + str(image_count) + '.jpg'
			rotated_image = Image.fromarray(aug_img_back.astype(np.uint8))
			rotated_image.save(new_path_image)
			new_path_label = self.label_glo_aug_dir + '/' + str(image_count) + '.png'
			rotated_label = Image.fromarray(aug_label_back.astype(np.uint8))
			rotated_label.save(new_path_label)

	def test_perspective_transform(self, scale_range):
		# _, _, glo_files = next(os.walk(self.img_glo_dir))
		_, _, img_files = next(os.walk(self.test_img_dir_path))
		# ipdb.set_trace()
		# image_count = len(img_files)
		for filename in img_files:
			image_path = self.test_img_dir_path + '/' + filename
			image = imageio.imread(image_path)
			label_path = self.test_labels_path + '/' + filename[:-4] + '.png'
			label = imageio.imread(label_path)
			aug = iaa.PerspectiveTransform(scale = scale_range, mode='constant', seed=15)
			seq = iaa.Sequential([aug])
			img_arr = np.reshape(image, (1, image.shape[0], image.shape[1], 3))
			label_arr = np.reshape(label, (1, label.shape[0], label.shape[1], 1))
			aug_img, aug_label = seq(images = img_arr, segmentation_maps = label_arr)
			aug_img_back = np.reshape(aug_img, (image.shape[0], image.shape[1], 3))
			aug_label_back = np.reshape(aug_label, (label.shape[0], label.shape[1]))
			# image_count += 1
			new_path_image = self.test_img_transform_path + '/' + filename
			rotated_image = Image.fromarray(aug_img_back.astype(np.uint8))
			rotated_image.save(new_path_image)
			new_path_label = self.test_labels_transform_path + '/' + filename[:-4] + '.png'
			rotated_label = Image.fromarray(aug_label_back.astype(np.uint8))
			rotated_label.save(new_path_label)

	def img_invert(self):
		_, _, files = next(os.walk(self.img_dir_path))
		image_count = len(files) + len(os.listdir(self.img_glo_aug_dir))
		for filename in files:
			image_path = self.img_dir_path + '/' + filename
			image = imageio.imread(image_path)
			label_path = self.label_dir_path + '/' + filename[:-4] + '.png'
			# label = imageio.imread(label_path)
			aug = iaa.Invert(1)
			augmented = aug.augment_image(image)
			image_count += 1
			new_path_image = self.img_glo_aug_dir + '/' + str(image_count) + '.jpg'
			augmented = Image.fromarray(augmented.astype(np.uint8))
			augmented.save(new_path_image)
			new_path_label = self.label_glo_aug_dir + '/' + str(image_count) + '.png'
			shutil.copyfile(label_path, new_path_label)

	def img_contrast(self):
		_, _, files = next(os.walk(self.img_dir_path))
		image_count = len(files) + len(os.listdir(self.img_glo_aug_dir))
		for filename in files:
			image_path = self.img_dir_path + '/' + filename
			image = imageio.imread(image_path)
			label_path = self.label_dir_path + '/' + filename[:-4] + '.png'
			# label = imageio.imread(label_path)
			aug = iaa.pillike.Autocontrast((5,20), per_channel=True)
			augmented = aug.augment_image(image)
			image_count += 1
			new_path_image = self.img_glo_aug_dir + '/' + str(image_count) + '.jpg'
			augmented = Image.fromarray(augmented.astype(np.uint8))
			augmented.save(new_path_image)
			new_path_label = self.label_glo_aug_dir + '/' + str(image_count) + '.png'
			shutil.copyfile(label_path, new_path_label)

	def img_gauss_noise(self):
		_, _, files = next(os.walk(self.img_dir_path))
		image_count = len(files) + len(os.listdir(self.img_glo_aug_dir))
		for filename in files:
			image_path = self.img_dir_path + '/' + filename
			image = imageio.imread(image_path)
			label_path = self.label_dir_path + '/' + filename[:-4] + '.png'
			# label = imageio.imread(label_path)
			aug = iaa.imgcorruptlike.GaussianNoise(severity=2)
			augmented = aug.augment_image(image)
			image_count += 1
			new_path_image = self.img_glo_aug_dir + '/' + str(image_count) + '.jpg'
			augmented = Image.fromarray(augmented.astype(np.uint8))
			augmented.save(new_path_image)
			new_path_label = self.label_glo_aug_dir + '/' + str(image_count) + '.png'
			shutil.copyfile(label_path, new_path_label)

	def img_saturate(self):
		_, _, files = next(os.walk(self.img_dir_path))
		image_count = len(files) + len(os.listdir(self.img_glo_aug_dir))
		for filename in files:
			image_path = self.img_dir_path + '/' + filename
			image = imageio.imread(image_path)
			label_path = self.label_dir_path + '/' + filename[:-4] + '.png'
			# label = imageio.imread(label_path)
			aug = iaa.imgcorruptlike.Saturate(severity=4)
			augmented = aug.augment_image(image)
			image_count += 1
			new_path_image = self.img_glo_aug_dir + '/' + str(image_count) + '.jpg'
			augmented = Image.fromarray(augmented.astype(np.uint8))
			augmented.save(new_path_image)
			new_path_label = self.label_glo_aug_dir + '/' + str(image_count) + '.png'
			shutil.copyfile(label_path, new_path_label)

	def img_defocus_blur(self):
		_, _, files = next(os.walk(self.img_dir_path))
		image_count = len(files) + len(os.listdir(self.img_glo_aug_dir))
		for filename in files:
			image_path = self.img_dir_path + '/' + filename
			image = imageio.imread(image_path)
			label_path = self.label_dir_path + '/' + filename[:-4] + '.png'
			# label = imageio.imread(label_path)
			aug = iaa.imgcorruptlike.DefocusBlur(severity=2)
			augmented = aug.augment_image(image)
			image_count += 1
			new_path_image = self.img_glo_aug_dir + '/' + str(image_count) + '.jpg'
			augmented = Image.fromarray(augmented.astype(np.uint8))
			augmented.save(new_path_image)
			new_path_label = self.label_glo_aug_dir + '/' + str(image_count) + '.png'
			shutil.copyfile(label_path, new_path_label)



if __name__ == '__main__':
	augmentation = aug()
	# augmentation.visualise()

	# augmentation.img_rotate()
	augmentation.test_perspective_transform(scale_range = (0.15, 0.20) )

	# augmentation.img_defocus_blur()
	# augmentation.img_saturate()
	# augmentation.img_gauss_noise()
	# augmentation.img_contrast()
	# augmentation.img_invert()
	

'''
DONE AUGMENTATIONS:
- ROTATION - 90, 180, 270
Original images and rotated images are now used as base for other augmentations.
- INVERT
- CONTRAST
- GAUSS NOISE
- SATURATE
- DEFOCUS BLUR
rotate = iaa.Affine(rotate=(90))
rotated_image = rotate.augment_image(image)
'''