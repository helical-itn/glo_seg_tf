from PIL import Image
import os
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
import albumentations as A
import cv2
import random
# from keras.utils.np_utils import to_categorical
# from keras.preprocessing.image import ImageDataGenerator


class aug(object):
	def __init__(self):
		self.current_folder_path = os.path.dirname(os.path.abspath(__file__))
		self.data_dir_path = self.current_folder_path + '/data'
		self.train_dir_path = self.data_dir_path + '/train'
		self.aug_train_dir_path = self.data_dir_path + '/aug_train'
		self.aug_img_dir_path = self.aug_train_dir_path + '/aug_image'
		self.aug_label_dir_path = self.aug_train_dir_path + '/aug_label'
		# self.img_dir_path = self.train_dir_path + '/image'
		self.img_dir_path = '/media/mihael/Hard/MUW NEW SLIDES/train_set_downsampled/images'
		# self.label_dir_path = self.train_dir_path + '/label'
		self.label_dir_path = '/media/mihael/Hard/MUW NEW SLIDES/train_set_downsampled/labels_3_cls'
		self.results_dir_path = self.current_folder_path + '/results_by_name'
		self.results_dir_path_TG = self.current_folder_path + '/results_by_name_TG'
		self.results_from_training_dir_path = self.current_folder_path + '/results_from_training'
		# self.test_img_dir_path = self.data_dir_path + '/test'
		# self.test_labels_path = self.data_dir_path + '/test_labels'
		self.test_img_dir_path = '/media/mihael/Hard/MUW NEW SLIDES/test_set/img_test'
		self.test_labels_path = '/media/mihael/Hard/MUW NEW SLIDES/test_set/lbl_test'
		self.augmented_images_path = '/media/mihael/Hard/MUW NEW SLIDES/train_set_downsampled/augmented_images'
		self.augmented_labels_path = '/media/mihael/Hard/MUW NEW SLIDES/train_set_downsampled/augmented_labels'
		self.test_img_transform_path = self.data_dir_path + '/test_transform'
		self.test_labels_transform_path = self.data_dir_path + '/test_labels_transform'
		self.ground_truth_array_path = self.current_folder_path + '/' + 'ground_truth_array.npy'
		self.predicted_images_one_hot = self.current_folder_path + '/' + 'predicted_images_one_hot.npy'

	def visualise_augmentations(self):
		# _, _, files = next(os.walk(self.img_dir_path))
		image_path = self.img_dir_path + '/9e837e99567dfd5fdb8cdfb9ab38da7c_20191223_132908_726 [x=16384,y=8192,w=8192,h=8192]_01_02.png'
		label_path = self.label_dir_path + '/9e837e99567dfd5fdb8cdfb9ab38da7c_20191223_132908_726 [x=16384,y=8192,w=8192,h=8192]_01_02.png'
		image = imageio.imread(image_path)
		# label = imageio.imread(label_path)
		label = np.asarray(Image.open(label_path))
		# label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
		# rotate = iaa.Affine(rotate=(45))
		# rotate = iaa.Rotate((90))
		# aug = iaa.PerspectiveTransform(scale= (0.09, 0.12), mode='constant', seed=15)
		# pers_aug = iaa.PerspectiveTransform(scale= (0.08), mode='replicate', fit_output=False, seed=15)
		# aug = iaa.ElasticTransformation(alpha=(5), sigma=(2.5), mode='nearest')
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
		# augmented = aug.augment_image(image)
		# augmented_label = aug.augment_image(label)
		# aug = iaa.imgcorruptlike.


		#############MAKING THE SAME AUGMENTATION FOR IMAGES AND MASKS AT ONCE WHEN USING imgaug
		# seq = iaa.Sequential([aug, pers_aug])
		# # label[label==170] = 1
		# # label[label==250] = 2
		# # label_arr = to_categorical(label, num_classes=3, dtype = 'uint8')
		# img_arr = np.reshape(image, (1, image.shape[0], image.shape[1], 3))
		# label_arr = np.reshape(label, (1, label.shape[0], label.shape[1], 4))
		# # ipdb.set_trace()
		# # image =
		# aug_img, aug_label = seq(images = img_arr, segmentation_maps = label_arr)
		# aug_img_back = np.reshape(aug_img, (image.shape[0], image.shape[1], 3))
		# aug_label_back = np.reshape(aug_label, (label.shape[0], label.shape[1], 4))
		########################################################################

		#USING AUGMENTOR #################################################################
		# p = Augmentor.Pipeline('/media/mihael/Hard/MUW NEW SLIDES/slide3-disease-name/tiles2/aug_test')
		# p.ground_truth('/media/mihael/Hard/MUW NEW SLIDES/slide3-disease-name/tiles2/aug_test/lbl')
		# p.random_distortion(probability=1, grid_width=16, grid_height=16, magnitude=20)
		# aug_img, aug_lbl = p.sample(1)
		# p.ImageDataGenerator.flow()
		# g = p.keras_generator(batch_size=1)
		# aug_img, aug_lbl = next(g)
		##################################################################################

		#USING AULBUMENTATIONS #################################################################
		# aug = A.ElasticTransform(p=1, alpha=400, sigma=20, alpha_affine=0.5,
		# 						 border_mode=cv2.BORDER_REPLICATE, interpolation=cv2.INTER_LINEAR)
		aug = A.RandomRotate90(p=1)
		# aug = A.Blur(blur_limit=(5,8), always_apply=False, p=1)
		# aug = A.CLAHE (clip_limit=(4.0, 16.0), tile_grid_size=(4, 4), always_apply=False, p=1)
		# aug = A.Downscale (scale_min=0.25, scale_max=0.4, interpolation=0, always_apply=False, p=1)
		# aug = A.Equalize (mode='cv', by_channels=True, mask=None, mask_params=(), always_apply=False, p=1)
		# aug = A.GaussNoise (var_limit=(50, 100), mean=0, always_apply=False, p=1)
		# aug = A.HueSaturationValue (hue_shift_limit=30, sat_shift_limit=30, val_shift_limit=30, always_apply=False, p=1)
		# aug = A.IAAEmboss (alpha=(0.5, 0.8), strength=(0.5, 0.8), always_apply=False, p=1)
		# aug = A.IAASharpen (alpha=(0.5, 0.8), lightness=(0.8, 1), always_apply=False, p=1)
		# aug = A.Solarize (threshold=(100,200), always_apply=False, p=1)
		augmented = aug(image=image, mask=label)

		# aug_label = self.elastic_transform(label, alpha=120, sigma=120 * 0.05, random_state=None)
		image_elastic = augmented['image']
		mask_elastic = augmented['mask']
		##################################################################################

		# image_reshaped = np.reshape(aug_img, (image.shape[1],image.shape[2],image.shape[3]))
		# label_reshaped = np.reshape(aug_lbl, (image.shape[1],image.shape[2],image.shape[3]))
		# plt.axis('off')
		plt.style.use('classic')
		plt.imshow(image)
		plt.show()
		plt.imshow(image_elastic)
		plt.show()
		plt.imshow(mask_elastic)
		plt.show()
		# ia.imshow(img_deformed)
		# ia.imshow(lbl_deformed)

		image_from_array = Image.fromarray(mask_elastic.astype(np.uint8))
		image_from_array.save('/media/mihael/Hard/MUW NEW SLIDES/Augmented_images/aug_testing/aug.png')

	def save_original_images(self, img_source_folder, lbl_source_folder, img_destination_folder, lbl_destination_folder):
		_, _, files = next(os.walk(img_source_folder))
		counter = 1
		for filename in files:
			image_path = img_source_folder + '/' + filename
			label_path = lbl_source_folder + '/' + filename
			aug_img_path = img_destination_folder + '/' + str(counter) + '_org.png'
			aug_lbl_path = lbl_destination_folder + '/' + str(counter) + '_org.png'
			counter+=1
			shutil.copyfile(image_path, aug_img_path)
			shutil.copyfile(label_path, aug_lbl_path)

	def save_augmented_images(self, aug_name, augmented_items, image_name):
		image_from_array = Image.fromarray(augmented_items['image'].astype(np.uint8))
		new_image_path = self.augmented_images_path + '/' + str(image_name) + '_' + aug_name +'.png'
		image_from_array.save(new_image_path)
		label_from_array = Image.fromarray(augmented_items['mask'].astype(np.uint8))
		new_label_path = self.augmented_labels_path + '/' + str(image_name) + '_' + aug_name + '.png'
		label_from_array.save(new_label_path)

	# def copy_images(self):

	def augment_and_save_images(self):
		_, _, files = next(os.walk(self.img_dir_path))
		counter = 1
		rotate_and_flip = A.Compose([
			A.VerticalFlip(p=1),
			A.RandomRotate90(p=1)
		])
		elastic_aug = A.ElasticTransform(p=1, alpha=400, sigma=20, alpha_affine=0.5,
										 border_mode=cv2.BORDER_REPLICATE, interpolation=cv2.INTER_LINEAR)
		clahe_aug = A.CLAHE (clip_limit=(4.0, 16.0), tile_grid_size=(4, 4), always_apply=False, p=1)
		blur_aug = A.Blur(blur_limit=(5,8), always_apply=False, p=1)
		downscale_aug = A.Downscale (scale_min=0.25, scale_max=0.4, interpolation=0, always_apply=False, p=1)
		equalize_aug = A.Equalize (mode='cv', by_channels=True, mask=None, always_apply=False, p=1)
		gaussnoise_aug = A.GaussNoise (var_limit=(50, 100), mean=0, always_apply=False, p=1)
		saturate_aug = A.HueSaturationValue (hue_shift_limit=30, sat_shift_limit=30, val_shift_limit=30, p=1)
		embos_aug = A.IAAEmboss (alpha=(0.5, 0.8), strength=(0.5, 0.8), always_apply=False, p=1)
		sharpen_aug = A.IAASharpen (alpha=(0.5, 0.8), lightness=(0.8, 1), always_apply=False, p=1)
		solarize_aug = A.Solarize (threshold=(100,200), always_apply=False, p=1)
		aug_dict = {'rotate': rotate_and_flip, 'elastic':elastic_aug, 'clahe':clahe_aug, 'blur':blur_aug,
							 'downscale': downscale_aug, 'equalize': equalize_aug, 'noise': gaussnoise_aug, 'saturate':saturate_aug,
							 'embos':embos_aug, 'sharpen':sharpen_aug, 'solarize': solarize_aug}
		required_aug_list = ['rotate', 'elastic', 'noise', 'saturate', 'solarize']

		for filename in files:
			image_path = self.img_dir_path + '/' + filename
			image = imageio.imread(image_path)
			label_path = self.label_dir_path + '/' + filename
			label = np.asarray(Image.open(label_path))
			for aug_type in required_aug_list:
				self.save_augmented_images(aug_name = aug_type,augmented_items=aug_dict[aug_type](image=image, mask=label),
										   image_name=counter)
			#Following 3 if loops are separated to get extra randomness
			if bool(random.getrandbits(1)):
				self.save_augmented_images(aug_name = 'clahe', augmented_items=aug_dict['clahe'](image=image, mask=label),
										   image_name=counter)
			else:
				self.save_augmented_images(aug_name = 'equalize', augmented_items=aug_dict['equalize'](image=image, mask=label),
										   image_name=counter)
			if bool(random.getrandbits(1)):
				self.save_augmented_images(aug_name = 'blur', augmented_items=aug_dict['blur'](image=image, mask=label),
										   image_name=counter)
			else:
				self.save_augmented_images(aug_name = 'downscale', augmented_items=aug_dict['downscale'](image=image, mask=label),
										   image_name=counter)
			if bool(random.getrandbits(1)):
				self.save_augmented_images(aug_name = 'embos', augmented_items=aug_dict['embos'](image=image, mask=label),
										   image_name=counter)
			else:
				self.save_augmented_images(aug_name = 'sharpen', augmented_items=aug_dict['sharpen'](image=image, mask=label),
										   image_name=counter)
			counter+=1


if __name__ == '__main__':
	augmentation = aug()
	# augmentation.visualise_augmentations()
	#FIRST ORIGINAL IMAGES NEEDS TO BE COPIED WITH FUNCTION BELOW, THAN ORIGINAL IMAGES ARE AUGMENTED
	augmentation.save_original_images(augmentation.img_dir_path, augmentation.label_dir_path,
									  augmentation.augmented_images_path, augmentation.augmented_labels_path)
	augmentation.augment_and_save_images()

	# augmentation.img_rotate()
	# augmentation.test_perspective_transform(scale_range = (0.15, 0.20) )

