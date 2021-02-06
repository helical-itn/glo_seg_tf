# TODO: USING MAP/FILTER ... - test execution speed with and without loops

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from PIL import Image
from keras.utils.np_utils import to_categorical
from tensorflow.keras.utils import plot_model

from unet_elements import *
from segmentation_results_and_metrics import prediction_and_metrics
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger, TensorBoard
import mlflow

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import random
import time

import tensorflow as tf
tf.compat.v1.disable_eager_execution()

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


import mlflow.tensorflow
# mlflow.set_tracking_uri("/home/mihael/ML/glo_seg_tensorflow/MAIN/mlruns")
# mlflow.tensorflow.autolog(every_n_iter=1)

class unet_training(object):

    def __init__(self, continue_training = False, old_model_name = '', new_model_name = '', train_from_array = True):

        # self.img_rows = img_rows
        # self.img_cols = img_cols
        self.new_model_name = new_model_name
        self.old_model_name = old_model_name
        self.continue_training = continue_training
        self.train_from_array = train_from_array
        self.current_folder_path = os.path.dirname(os.path.abspath(__file__))
        self.trained_models_path = '/home/mihael/ML/glo_seg_tensorflow/MAIN/trained_models'
        self.trained_models_server = self.trained_models_path + '/Trained_On_Server'
        self.data_dir_path = self.current_folder_path + '/data'
        self.results_dir_path = '/media/mihael/Hard/ML/glo_seg_tensorflow/MAIN/results_by_name'
        self.result_img_path = self.results_dir_path + '/mgs_mask_test.npy'
        self.train_dir_path = self.data_dir_path + '/train'
        self.img_dir_path = '/home/mihael/ML/glo_seg/data/train_256'
        self.label_dir_path = '/home/mihael/ML/glo_seg/data/masks_256'
        self.test_img_dir_path = self.data_dir_path + '/test'
        self.npy_dir_path = self.current_folder_path + '/npydata'
        self.statistics_path = '/home/mihael/ML/glo_seg/narrow_deep_Anet/statistics'
        self.aug_img = '/home/mihael/ML/glo_seg_tensorflow/MAIN/data/aug_train/aug_image'
        self.aug_lbl = '/home/mihael/ML/glo_seg_tensorflow/MAIN/data/aug_train/aug_label'

    def train(self, unet_model, batch_size, epochs):
        # with mlflow.start_run():
        time_stamp = time.strftime("%Y%m%d-%H%M%S")
        #model loading
        new_model_path = self.trained_models_path + "/" + self.new_model_name + '_'  + time_stamp + '.hdf5'
        if self.continue_training == True:
            model_path = self.trained_models_path + "/" + self.old_model_name
            model = load_model(model_path)
        else:
            model = unet_model
        model_checkpoint = ModelCheckpoint(new_model_path, monitor='loss', verbose=1, save_best_only=True)
        plot_model(model, to_file= self.results_dir_path + "/" + self.new_model_name + '.png',
                   show_shapes=True)
        #data loading
        if self.train_from_array == True:
            big_training_path = self.data_dir_path + '/22_imgs.npy'
            big_training = np.load(big_training_path)
            big_label_path = self.data_dir_path + '/22_labels.npy'
            big_label = np.load(big_label_path)
        else:
            _, _, files = next(os.walk(self.img_play_dir))
            random.shuffle(files)
            big_training = []
            big_label = []

            for item_name in files:
                item_path = self.img_play_dir + '/' + item_name
                label_png_name = item_name[:-4] + '.png'
                label_path = self.label_play_dir + '/' + label_png_name
                item_arr = np.asarray(Image.open(item_path))
                big_training.append(item_arr)
                label_arr = np.asarray(Image.open(label_path))
                label_arr = label_arr.copy()
                # IN CASE OF 2D MASKS:
                # label_arr.shape = [1024, 1024, 1]
                # label_one_hot = np.zeros((1024, 1024, 3))
                # categorical_labels = to_categorical(label_arr, num_classes=3)

                # label_one_hot[np.arange(1024),label_arr] = 1
                # label_3D = label_arr[:,:,newaxis = 3]
                # label_arr[label_arr == 2] = 1
                # label_arr[label_arr == 3] = 2
                label_arr[label_arr == 170] = 1
                label_arr[label_arr == 250] = 2

                # NEXT LINE IS USED TO GET MASK INPUT ARRAY IN THE RIGHT SHAPE (WHEN 2D MASK IS USED)
                # label_arr.shape = [1024, 1024, 1]
                label_arr = to_categorical(label_arr, num_classes=3, dtype='uint8')
                big_label.append(label_arr)
            big_training = np.asarray(big_training)
            big_label = np.asarray(big_label)
        # class_weight_manual = [1, 1, 5]
        mlflow.set_tracking_uri('/home/mihael/ML/glo_seg_tensorflow/MAIN/mlruns/')
        mlflow.set_experiment('Test_1')
        with mlflow.start_run():
            # mlflow.tensorflow.autolog(every_n_iter=1)
            mlflow.log_param("batch_size", batch_size)
            model.fit(big_training, big_label, batch_size=batch_size, epochs=epochs, verbose=1, shuffle=True, callbacks=[(model_checkpoint)])
        # model.fit(train_batch, label_batch, verbose=1, validation_split=0.25,  callbacks=[model_checkpoint]) #validation_split=0.25,
        model_stat_folder = self.statistics_path + "/stats_" + self.new_model_name + '_'  + time_stamp
        os.makedirs(model_stat_folder)
        return new_model_path, model_stat_folder
        # mlflow.end_run()

if __name__ == '__main__':
    def timer(start,end):
        hours, rem = divmod(end-start, 3600)
        minutes, seconds = divmod(rem, 60)
        print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))

    image_height = None
    image_width = None
    image_channels = 3
    output_classes = 3 #in case of 3 labels predicted is glomeruli, the rest of the tissue and backgorund
    #in case of 2 labels, predicted is glomeruli and the rest is background
    learning_rate = 0.0001
    batch_size = 1
    epochs = 10
    continue_training = False #False - it will build model from scratch and then train it, True - it will take pretrained model
    #and continue training, if True write name of the model you want to use below
    old_model_name = 'top-weights_22_ep.h5'
    new_model_name = 'TEST_image_play_Anet_TEST'
    train_from_array = False #True - it will load training samples and labels which are saved as numpy array,
    # False - it will load training samples and labels which are saved as images in self.img_dir_path and  self.label_dir_path
    train_and_predict = True
    just_train = False
    just_predict = False

    if train_and_predict or just_train:
        start = time.time()
        unet_model = narrow_deep_Anet(IMG_HEIGHT = image_height, IMG_WIDTH = image_width, IMG_CHANNELS = image_channels,
                                        No_Classes = output_classes, LearnRate = learning_rate)
        myunet = unet_training(continue_training = continue_training, old_model_name = old_model_name, new_model_name = new_model_name, train_from_array = train_from_array)
        model_path, model_stat_folder = myunet.train(unet_model, batch_size = batch_size, epochs = epochs)
        end = time.time()
        print ('::::::::::::::::::::::::::::Time needed for data loading and training::::::::::::::::::::::::::::')
        timer(start, end)

    if just_predict:
        model_path = '/home/mihael/ML/glo_seg_tensorflow/MAIN/trained_models/A_Net/Narrow_Deep_Anet_3_cls_train_set_20210119-204245.hdf5'
        model_stat_folder = '/home/mihael/ML/glo_seg/narrow_deep_Anet/statistics'

    if train_and_predict or just_predict:
        start = time.time()
        predict_and_print_metrics = prediction_and_metrics(model_path = model_path, model_stat_folder = model_stat_folder, No_Classes = output_classes)
        ## prediction_and_metrics(model_path, model_stat_folder).predict()
        predict_and_print_metrics.predict()
        predict_and_print_metrics.metrics_for_whole_test()
        predict_and_print_metrics.save_predictions_as_images()
        end = time.time()
        print (':::::::Time needed for prediction, metrics calculation and converting to images:::::::')
        timer(start, end)
