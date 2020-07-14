# TODO: USING MAP/FILTER ... - test execution speed with and without loops

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from PIL import Image
from keras.utils.np_utils import to_categorical

from unet_elements import *
from segmentation_results_and_metrics import prediction_and_metrics

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import random
import time

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

class unet_training(object):

    def __init__(self, continue_training = False, old_model_name = '', new_model_name = '', train_from_array = True):

        # self.img_rows = img_rows
        # self.img_cols = img_cols
        self.new_model_name = new_model_name
        self.old_model_name = old_model_name
        self.continue_training = continue_training
        self.train_from_array = train_from_array
        self.current_folder_path = os.path.dirname(os.path.abspath(__file__))
        self.trained_models_path = self.current_folder_path + '/trained_models'
        self.data_dir_path = self.current_folder_path + '/data'
        self.results_dir_path = self.current_folder_path + '/results'
        self.result_img_path = self.results_dir_path + '/mgs_mask_test.npy'
        self.train_dir_path = self.data_dir_path + '/train'
        self.img_dir_path = self.train_dir_path + '/image'
        self.label_dir_path = self.train_dir_path + '/label'
        self.img_play_dir = self.train_dir_path + '/image_play'
        self.label_play_dir = self.train_dir_path + '/label_play'
        self.img_128_path = self.train_dir_path + '/image_128'
        self.label_128_path = self.train_dir_path + '/label_128'
        self.img_256_path = self.train_dir_path + '/image_256'
        self.label_256_path = self.train_dir_path + '/label_256'
        self.img_512_path = self.train_dir_path + '/image_512'
        self.label_512_path = self.train_dir_path + '/label_512'
        self.test_img_dir_path = self.data_dir_path + '/test'
        self.npy_dir_path = self.current_folder_path + '/npydata'
        self.statistics_path = self.current_folder_path + '/statistics'

    def train(self, unet_model, batch_size, epochs):
        time_stamp = time.strftime("%Y%m%d-%H%M%S")
        #model loading
        new_model_path = self.trained_models_path + "/" + self.new_model_name + '_'  + time_stamp + '.hdf5'
        if self.continue_training == True:
            model_path = self.trained_models_path + "/" + self.old_model_name + '.hdf5'
            model = load_model(model_path)
        else:
            model = unet_model
        model_checkpoint = ModelCheckpoint(new_model_path, monitor='loss', verbose=1, save_best_only=True)

        #data loading
        if self.train_from_array == True:
            big_training_path = self.data_dir_path + '/all_images.npy'
            big_training = np.load(big_training_path)
            big_label_path = self.data_dir_path + '/all_labels_3cls.npy'
            big_label = np.load(big_label_path)
        else:
            _, _, files = next(os.walk(self.img_dir_path))
            random.shuffle(files)
            big_training = []
            big_label = []

            for item_name in files:
                item_path = self.img_dir_path + '/' + item_name
                label_png_name = item_name[:-4] + '.png'
                label_path = self.label_dir_path + '/' + label_png_name
                item_arr = np.asarray(Image.open(item_path))
                big_training.append(item_arr)
                label_arr = np.asarray(Image.open(label_path))
                label_arr = label_arr.copy()
                # IN CASE OF 2D MASKS:
                # label_arr.shape = [1024, 1024, 1]
                # ipdb.set_trace()
                # label_one_hot = np.zeros((1024, 1024, 3))
                # ipdb.set_trace()
                # categorical_labels = to_categorical(label_arr, num_classes=3)

                # label_one_hot[np.arange(1024),label_arr] = 1
                # label_3D = label_arr[:,:,newaxis = 3]
                # label_arr[label_arr==160] = 0
                label_arr[label_arr == 170] = 1
                label_arr[label_arr == 250] = 2
                # NEXT LINE IS USED TO GET MASK INPUT ARRAY IN THE RIGHT SHAPE (WHEN 2D MASK IS USED)
                # label_arr.shape = [1024, 1024, 1]
                label_arr = to_categorical(label_arr, num_classes=3, dtype='uint8')
                big_label.append(label_arr)
            big_training = np.asarray(big_training)
            big_label = np.asarray(big_label)
        class_weight_manual = [5, 1, 1]
        model.fit(big_training, big_label, batch_size=batch_size, nb_epoch=epochs, class_weight = class_weight_manual, verbose=1, shuffle=True, callbacks=[(model_checkpoint)])
        # model.fit(train_batch, label_batch, verbose=1, validation_split=0.25,  callbacks=[model_checkpoint]) #validation_split=0.25,
        model_stat_folder = self.statistics_path + "/stats_" + self.new_model_name + '_'  + time_stamp
        os.makedirs(model_stat_folder)
        return new_model_path, model_stat_folder

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
    batch_size = 2
    epochs = 100
    continue_training = True #False - it will build model from scratch and then train it, True - it will take pretrained model
    #and continue training, if True write name of the model you want to use below
    old_model_name = '512_norm_progressive_20200714-084432'
    new_model_name = '1024_norm_progressive'
    train_from_array = True #True - it will load training samples and labels which are saved as numpy array,
    # False - it will load training samples and labels which are saved as images in self.img_dir_path and  self.label_dir_path

    start = time.time()
    unet_model = my_unet_batch_norm(IMG_HEIGHT = image_height, IMG_WIDTH = image_width, IMG_CHANNELS = image_channels,
                                    No_Classes = output_classes, LearnRate = learning_rate)
    myunet = unet_training(continue_training = continue_training, old_model_name = old_model_name, new_model_name = new_model_name, train_from_array = train_from_array)
    model_path, model_stat_folder = myunet.train(unet_model, batch_size = batch_size, epochs = epochs)
    end = time.time()
    print ('::::::::::::::::::::::::::::Time needed for data loading and training::::::::::::::::::::::::::::')
    timer(start, end)

    # model_path = '/home/mihael/ML/glo_seg_tensorflow/MAIN/trained_models/progressive_1024_150ep_20200710-160535.hdf5'
    # model_stat_folder = '/home/mihael/ML/glo_seg_tensorflow/MAIN/stat_progress'

    start = time.time()
    predict_and_print_metrics = prediction_and_metrics(model_path = model_path, model_stat_folder = model_stat_folder, No_Classes = output_classes)
    ## prediction_and_metrics(model_path, model_stat_folder).predict()
    predict_and_print_metrics.predict()
    predict_and_print_metrics.metrics_for_whole_test()
    predict_and_print_metrics.save_predictions_as_images()
    end = time.time()
    print (':::::::Time needed for prediction, metrics calculation and converting to images:::::::')
    timer(start, end)



'''
LIST OF SOME BUGS AND HOW THEY WERE HANDLED

1)
self.results[0] /= self.num_samples_or_steps
IndexError: list index out of range
- i think there was a problem because I had batch size 1, after I changed it to
8 error was gone, but then next error happened:

2)
tensorflow.python.framework.errors_impl.UnknownError: Failed to get convolution algorithm. 
This is probably because cuDNN failed to initialize, so try looking to see if a warning 
log message was printed above.
	 [[{{node conv2d/Conv2D}}]]
	 [[{{node loss/mul}}]]
https://github.com/tensorflow/tensorflow/issues/24828
SOLVED WITH ADDING:
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

3)
TypeError: Input 'y' of 'Equal' Op has type float32 that does not match type int32 of argument 'x'.

maybe the wrong optimizer was used, so instead of:
model.compile(optimizer = Adam(lr=LearnRate), loss= 'sparse_categorical_crossentropy' , metrics=['acc'])

I use:
model.compile(
            optimizer=keras.optimizers.Adadelta(),
            loss='sparse_categorical_crossentropy',
            metrics=['sparse_categorical_accuracy'])
'''
