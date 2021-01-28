from PIL import Image
# from libtiff import TIFF
import numpy as np
import ipdb
# import pdb
import os
# from osgeo import gdal
import rasterio as rio
import shutil
import cv2
import imageio
import image_slicer
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical

# im = gdal.Open('crop_1.tif')
# myarray = np.array(im.GetRasterBand(1).ReadAsArray())
# current_folder_path = os.path.dirname(os.path.abspath(__file__))

# ipdb.set_trace()

# with rio.open('/home/mihael/ML/data_slides/glomeruli_sections/4_bg_Mask.jpg', 'r') as ds:
# 	arr = ds.read()
class read_n_play_now():
    def __init__(self, img_rows = 1024, img_cols = 1024):

        self.img_rows = img_rows
        self.img_cols = img_cols
        self.current_folder_path = os.path.dirname(os.path.abspath(__file__))
        self.data_dir_path = self.current_folder_path + '/data'
        self.results_dir_path = self.current_folder_path + '/results'
        self.result_img_path = self.results_dir_path + '/mgs_mask_test.npy'
        self.train_dir_path = self.data_dir_path + '/train'
        self.slides_dir = "/media/mihael/Hard/MUW NEW SLIDES/slide8-disease-name/tiles2"
        self.img_dir_path = '/home/mihael/ML/glo_seg_tensorflow/MAIN/data/train'
        self.label_dir_path = self.train_dir_path + '/label'
        self.img_play_dir = self.train_dir_path + '/image_play'
        self.label_play_dir = self.train_dir_path + '/label_play'
        self.img_repair_dir = self.train_dir_path + '/image_repaired'
        self.label_repair_dir = self.train_dir_path + '/label_repaired'
        self.test_img_dir_path = self.data_dir_path + '/test'
        self.test_512 = self.data_dir_path + '/test_512'
        self.npy_dir_path = self.current_folder_path + '/npydata'
        self.augmented_images_path = '/media/mihael/Hard/MUW NEW SLIDES/Augmented_images/augmented_images'
        self.augmented_labels_path = '/media/mihael/Hard/MUW NEW SLIDES/test_set_downsampled/labels'
        self.augmented_labels_3cls_path = '/media/mihael/Hard/MUW NEW SLIDES/test_set_downsampled/labels_3_cls'

    # img_path = masks_altering_path + '/' + 'HE_3_Mask_bg_01_01.jpg'
    # a = np.asarray(Image.open(img_path))

    #CONVERT LABELS TO ONE HOT LABELS##################################################
    # for filename in labels_1024_path:
    # 	file_path = labels_1024_path + '/' + filename
    # 	arr_mask = np.asarray(Image.open(file_path))
    # 	arr_mask[arr_mask==160] = [1, 0, 0]
    # 	arr_mask[arr_mask==60] = [0, 1, 0]
    # 	arr_mask[arr_mask==140] = [0, 0, 1]
    ##################################################################################
    def lower_resolution(self, basewidth):
        #LOWER IMAGE RESOLUTION
        # self.new_image_dir = self.train_dir_path + '/image_' + str(basewidth)
        # self.new_label_dir = self.train_dir_path + '/label_' + str(basewidth)
        self.new_image_dir = self.slides_dir + '/image_down_' + str(basewidth)
        self.new_label_dir = self.slides_dir + '/label_down_' + str(basewidth)

        try:
            if not os.path.exists(self.new_image_dir):
                os.mkdir(self.new_image_dir)
            if not os.path.exists(self.new_label_dir):
                os.mkdir(self.new_label_dir)
        except OSError:
            print ("Creation of the directory failed" )

        # self.img_dir_path = self.slides_dir + "/img"
        # self.label_dir_path = self.slides_dir + "/lbl"
        _, _, files = next(os.walk(self.img_dir_path))
        for filename in files:
            image_path = self.img_dir_path + "/" + filename
            img = Image.open(image_path)
            wpercent = (basewidth / float(img.size[0]))
            hsize = int((float(img.size[1]) * float(wpercent)))
            # img = img.resize((basewidth, hsize), Image.NEAREST)
            img = img.resize((basewidth, basewidth), Image.BICUBIC)
            new_file_path = self.new_image_dir + '/' + filename
            img.save(new_file_path)

            label_path = self.label_dir_path + "/" + filename[:-4] + "-labelled.png"
            img = Image.open(label_path)
            # wpercent = (basewidth / float(img.size[0]), Image.NEAREST)
            wpercent = (basewidth / float(img.size[0]))
            hsize = int((float(img.size[1]) * float(wpercent)))
            # img = img.resize((basewidth, hsize))
            img = img.resize((basewidth, basewidth), Image.NEAREST)
            # ipdb.set_trace()
            label_file_path = self.new_label_dir + '/' + filename[:-4] + '.png'
            img.save(label_file_path)
    def load_norm_save(self):
        _, _, files = next(os.walk(self.img_dir_path))
        imgs_array = []
        for filename in files:
            image_path = self.img_dir_path + '/' + filename

    def slice(self, portions):
        self.img_dir_path = self.slides_dir + "/image_2048"
        self.label_dir_path = self.slides_dir + "/label_2048"
        self.sliced_img_path = self.slides_dir + "/image_1024"
        self.sliced_lbl_path = self.slides_dir + "/label_1024"

        for filename in os.listdir(self.img_dir_path):
            file_path = self.img_dir_path + '/' + filename
            tiles = image_slicer.slice(file_path, portions, save=False)
            filename_no_sufix = filename[:-4]
            # slice_name = staining + '_' + filename_no_sufix + '_Mask'
            image_slicer.save_tiles(tiles, directory=self.sliced_img_path, prefix=filename_no_sufix, format='PNG')
            portion_name = filename[-9:-4]
            mask_file_path = self.label_dir_path + "/" + filename #[:-4] + "-labelled.png"
            # ipdb.set_trace()
            # mask_file_path_to_save = test_labels_1024_path + '/' + filename_no_sufix + '.png'

            tiles = image_slicer.slice(mask_file_path, portions, save=False)
            image_slicer.save_tiles(tiles, directory=self.sliced_lbl_path, prefix=filename_no_sufix, format='PNG')

    def labels_to_3_classes(self):
        for filename in os.listdir(self.augmented_labels_path):
            item_path = self.augmented_labels_path + '/' + filename
            label_arr = np.asarray(Image.open(item_path))
            label_arr = label_arr.copy()
            label_arr[label_arr == 3] = 2
            label_arr[label_arr == 4] = 2
            label_arr[label_arr == 5] = 2
            label_from_array = Image.fromarray(label_arr.astype(np.uint8))
            new_label_path = self.augmented_labels_3cls_path + '/' + filename
            label_from_array.save(new_label_path)



def copy_paste_files(self):
    counter = 1
    for filename in os.listdir(self.img_dir_path):
        img_path = self.img_dir_path + '/' + filename
        label_path = self.label_dir_path + '/' + filename[:-4] + '.png'
        new_img_path = self.img_repair_dir + '/' + str(counter) + '.jpg'
        new_label_path = self.label_repair_dir + '/' + str(counter) + '.png'
        shutil.copyfile(img_path, new_img_path)
        shutil.copyfile(label_path, new_label_path)
        counter += 1

    # im1 = Image.open(file_path)
    # im1.save(png_image_path)
def save_array(self):
    counter = 1
    imgs_array = []
    labels_array = []
    for filename in os.listdir(self.img_play_dir):
        # if counter <= 2:
        img_path = self.img_dir_path + '/' + filename
        label_path = self.label_play_dir + '/' + filename[:-4] + '.png'
        img = np.asarray(Image.open(img_path))
        imgs_array.append(img)
        label = np.asarray(Image.open(label_path))
        labels_array.append(label)
        counter += 1
    print ('there are images: ', counter)
    imgs_array = np.asarray(imgs_array)
    img_arr_path = self.data_dir_path + '/22_imgs.npy'
    ipdb.set_trace()
    np.save(img_arr_path, imgs_array)
    labels_array = np.asarray(labels_array)
    label_arr_path = self.data_dir_path + '/22_labels.npy'
    np.save(label_arr_path, labels_array)

class extracting_seg_objects():
    def __init__(self):
        self.test_img_folder = '/media/mihael/Hard/MUW NEW SLIDES/slide8-disease-name/tiles2/img'
        self.test_lbl_folder = '/media/mihael/Hard/MUW NEW SLIDES/slide8-disease-name/tiles2/lbl'
        self.data_folder = '/media/mihael/Hard/MUW NEW SLIDES/data'

    def crop_part_outside_image(self, image_dimension, bot_x, top_x, bot_y, top_y):
        if bot_x < 0:
            bot_x = 0
        if top_x > image_dimension - 1:
            top_x = image_dimension -1
        if bot_y < 0:
            bot_y = 0
        if top_y > image_dimension - 1:
            bot_y = image_dimension - 1
        return bot_x, top_x, bot_y, top_y

    def extracting_objects_from_mask(self, label_list):
        for label in label_list:
            label_dir = self.data_folder + '/' + str(label)
            if not os.path.exists(label_dir):
                os.mkdir(label_dir)
        for filename in os.listdir(self.test_lbl_folder):
            mask_path = self.test_lbl_folder + '/' + filename
            img_path = self.test_img_folder + '/' + filename.replace('-labelled', '')
            # mask=cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = np.asarray(Image.open(mask_path))
            img = np.asarray(Image.open(img_path))#.astype(np.float32)
            # empty_mask = np.zeros(shape = (1024, 1024), dtype=np.int8)
            # empty_mask_3D = np.zeros(shape = (1024, 1024, 3), dtype=np.int8)
            for label in label_list:
                if int(label) in mask:
                    label_filtered = cv2.inRange(mask, int(label), int(label))
                    contours,hierarchy = cv2.findContours(label_filtered, 1, 2)
                    for cnt in contours:
                        cnt_count = 0
                        empty_mask = np.zeros(shape = (8192, 8192), dtype=np.int8)
                        # empty_mask_3D = np.zeros(shape = (1024, 1024, 3), dtype=np.int8)
                        (x, y), radius = cv2.minEnclosingCircle(cnt)
                        center = (int(x), int(y))
                        radius = int(radius) + int(int(radius)*0.2)
                        glo_mask = cv2.circle(empty_mask, center, radius, (1), -1)

                        mask_float = np.float32(glo_mask)
                        mask_RGB = cv2.cvtColor(mask_float, cv2.COLOR_GRAY2RGB)
                        out = np.zeros_like(img)
                        out[mask_RGB==1] = img[mask_RGB==1]

                        bot_x = int(x) - radius
                        top_x = int(x) + radius
                        bot_y = int(y) - radius
                        top_y = int(y) + radius
                        bot_x, top_x, bot_y, top_y = self.crop_part_outside_image(8192, bot_x, top_x, bot_y, top_y)
                        out = out[bot_y:top_y, bot_x:top_x]

                        glo_path = self.data_folder + '/' + str(label) + '/' + filename.replace('-labelled.png', '') \
                                   +'_'+ str(cnt_count) + '.png'
                        cnt_count+=1
                        imageio.imwrite(glo_path, out)



#SLICING IMAGES OR MASKS##########################################################################
# portions = 4
# for filename in os.listdir(test_images_path):
# 	file_path = test_images_path + '/' + filename
# 	tiles = image_slicer.slice(file_path, portions, save=False)
# 	filename_no_sufix = filename[:-4]
# 	# slice_name = staining + '_' + filename_no_sufix + '_Mask'
# 	image_slicer.save_tiles(tiles, directory=test_images_1024_path, prefix=filename_no_sufix, format='JPEG')
# 	portion_name = filename[-9:-4]
# 	mask_file_path = test_labels_path + '/' + filename[:-9] + 'Mask_' + portion_name + '.png'
# 	# ipdb.set_trace()
# 	# mask_file_path_to_save = test_labels_1024_path + '/' + filename_no_sufix + '.png'

# 	tiles = image_slicer.slice(mask_file_path, portions, save=False)
# 	image_slicer.save_tiles(tiles, directory=test_labels_1024_path, prefix=filename_no_sufix, format='PNG')
##################################################################################

#TAKIN NAMES OF IMAGES AND MAKING THEM SHORTER#######################################
# counter = 1
# for filename in os.listdir(train_images_path):
# 	file_path = train_images_path + '/' + filename
# 	stain_name = filename.split('_')
# 	stain_name = stain_name[0]
# 	new_name_file_path = masks_altering_path + '/' + stain_name + '_' + str(counter) + '.jpg'
# 	# og_img_name = filename.replace('_Mask', '')
# 	# I = np.asarray(Image.open(file_path))
# 	# I.shape = [2048, 2048, 1]
# 	# ipdb.set_trace()
# 	# masks_altering_path_path = masks_altering_path + '/' + og_img_name
# 	shutil.copyfile(file_path, new_name_file_path)
# 	counter += 1
###############################################################################
# 	file_path = masks_altering_path + '/' + filename
# 	arr_mask = np.asarray(Image.open(file_path))
# 	arr_mask = arr_mask.copy()
# 	print (filename)
# 	# arr_mask[arr_mask==240] = 160
# 	for pix in np.nditer(arr_mask):
# 		if pix != 60 and pix != 140 and pix != 160 and pix != 150 and pix != 50:
# 			counter += 1
# 	print(counter)
# print (counter)
# im_array = np.array(im)

#CREATING MASKS WITH 0 AND 255 AS VALUES###################################################
# w, h, d = 4096, 4096, 3
# list_0 = [[[0 for x in range(d)] for y in range(h)] for z in range(w)]
# arr_0 = np.asarray(list_0)
# list_255 = [[[255 for x in range(d)] for y in range(h)] for z in range(w)]
# arr_255 = np.asarray(list_255)
# arr_0 = [[[0, 0, 0] for y in range(h)] for z in range(w)]
# im_0 = Image.fromarray(arr_0.astype(np.uint8))
# im_0.save("arr_0.jpg")
# im_255 = Image.fromarray(arr_255.astype(np.uint8))
# im_255.save("arr_255.jpg")
############################################################################################

#3D masks to 2D  masks converter -> RGB to grayscale ###################################
# all_images_path = current_folder_path + '/all_images'
# masks_path = all_images_path + '/masks_no_glo'
# masks_altering_path = all_images_path + '/masks_altering'
# for filename in os.listdir(masks_path):
# 	filename_path = masks_path + '/' + filename
# 	img = Image.open(filename_path).convert('L')
# 	new_path = masks_altering_path + '/' + filename
# 	img.save(new_path)
############################################################################################

# 2D MASKS TO 3D MASKS converter -> grayscale to RGB ############################
# all_images_path = current_folder_path + '/all_images'
# train_images_path = all_images_path + '/train_labels'
# masks_path = all_images_path + '/masks_no_glo'
# masks_altering_path = all_images_path + '/masks_altering'
# for filename in os.listdir(test_labels_1024_path):
# filename_path = test_labels_1024_path + '/' + filename
# img = Image.open(filename_path).convert('L')
# rgb_img = cv2.cvtColor(filename_path,cv2.COLOR_GRAY2RGB)
# new_path = masks_altering_path + '/' + filename
# rgb_img.save(new_path)
# Image.open(filename_path).convert('RGB').save(new_path)
#######################################################################

#Copying and pasting files from one folder to another - change image format
# he_images = all_large_images_path + '/' + 'HE_labels'
# for filename in os.listdir(he_images):
# 	file_path = he_images + '/' + filename
# 	img_name = filename[:-4] + '.png'

# 	png_image_path = all_large_images_path + '/images/' + img_name

# # 	shutil.copyfile(file_path, summ_img_path)

# 	im1 = Image.open(file_path)
# 	im1.save(png_image_path)
##################################################################################

######################################################Elastic transform
# from scipy.ndimage.interpolation import map_coordinates
# from scipy.ndimage.filters import gaussian_filter
# def elastic_transform(self, image, alpha, sigma, random_state=None):
#     """Elastic deformation of images as described in [Simard2003].
#
#     .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
#        Convolutional Neural Networks applied to Visual Document Analysis", in
#        Proc. of the International Conference on Document Analysis and
#        Recognition, 2003.
#     """
#     if random_state is None:
#         random_state = np.random.RandomState(None)
#
#     h, w = image.shape[:2]
#     x, y = np.meshgrid(np.arange(w), np.arange(h))
#     dx = gaussian_filter((random_state.rand(h,w) * 2 - 1), sigma, mode="constant", cval=0) * alpha
#     dy = gaussian_filter((random_state.rand(h,w) * 2 - 1), sigma, mode="constant", cval=0) * alpha
#     indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))
#
#     if len(image.shape) > 2:
#         c = image.shape[2]
#         distored_image = [map_coordinates(image[:,:,i], indices, order=1, mode='reflect') for i in range(c)]
#         distored_image = np.concatenate(distored_image, axis=1)
#     else:
#         distored_image = map_coordinates(image, indices, order=1, mode='reflect')
#
#     return distored_image.reshape(image.shape)
###################################################################################
# ipdb.set_trace()
# pdb.set_trace()
if __name__ == '__main__':
    r_n_p = read_n_play_now()
    # r_n_p.copy_paste_files()
    # r_n_p.lower_resolution(1024)
    # r_n_p.labels_to_3_classes()
    # ext = extracting_seg_objects()
    # ext.extracting_objects_from_mask(label_list=[2,3,4,5])
    # arr = np.load('/home/mihael/ML/glo_seg_tensorflow/ground_truth_array.npy')
    img = cv2.imread('/home/mihael/ML/glo_seg/data/labels_part/200209761_09_SFOG [x=32768,y=0,w=4096,h=4096]-labelled.png', cv2.IMREAD_GRAYSCALE)#/255SAssociates
    train_mask = to_categorical(img, num_classes=3, dtype='uint8')
    print ('jebiga')
