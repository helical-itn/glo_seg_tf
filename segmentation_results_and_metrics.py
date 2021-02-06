import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import pandas as pd
from tensorflow.python.keras.models import *
from PIL import Image
from keras.utils.np_utils import to_categorical
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import jaccard_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report
import time
from tensorflow.keras.layers import  Lambda
# config = ConfigProto()
# config.gpu_options.allow_growth = True
# session = InteractiveSession(config=config)

def timer(start,end):
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))

class prediction_and_metrics(object):

    def __init__(self, model_path, model_stat_folder, No_Classes):
        self.model_path = model_path
        self.model_stat_folder = model_stat_folder
        self.number_of_classes = No_Classes
        self.current_folder_path = os.path.dirname(os.path.abspath(__file__))
        self.data_dir_path = self.current_folder_path + '/data'
        self.train_dir_path = self.data_dir_path + '/train'
        self.img_dir_path = self.train_dir_path + '/image'
        self.results_dir_path = '/home/mihael/ML/glo_seg/narrow_deep_Anet/results'
        self.results_iou_path = self.current_folder_path + '/results_by_name_iou'
        self.results_dir_path_TG = self.current_folder_path + '/results_by_name_TG'
        self.results_from_training_dir_path = self.current_folder_path + '/results_from_training'
        self.test_img_dir_path = '/home/mihael/ML/glo_seg/data/img_from_train'
        self.test_labels_path = '/home/mihael/ML/glo_seg/data/lbl_from_train'
        self.test_img_transform_path = self.data_dir_path + '/test_transform'
        self.test_labels_transform_path = self.data_dir_path + '/test_labels_transform'
        self.ground_truth_array_path = '/home/mihael/ML/glo_seg/data/ground_truth_array.npy'
        self.predicted_images_one_hot = '/home/mihael/ML/glo_seg/data/predicted_images_one_hot.npy'
        self.filenames_array_path = '/home/mihael/ML/glo_seg/data/filenames_array.npy'
        self.results_form_colab = self.current_folder_path + '/results_form_colab/first_run'
        self.statistics_path = self.current_folder_path + '/statistics'
        if self.number_of_classes == 2:
            self.labels = [0, 1]
            # self.conf_matrix_labels = ['glo', 'bg']
            self.conf_matrix_labels = ['bg', 'glo'] # prepared for hubmap data
        elif self.number_of_classes == 3:
            self.labels = [0, 1, 2]
            self.conf_matrix_labels = ['glo', 'cells', 'bg']
        else:
            raise AssertionError ("You have more than 3 classes. This metrics calculation is suited only for 2 or 3 classes.")

    def predict(self):
        model = load_model(self.model_path, compile=False)
        _, _, files = next(os.walk(self.test_img_dir_path))
        images_to_predict = []
        ground_truth_array = []
        for filename in files:
            test_file_path = self.test_img_dir_path + '/' + filename
            ground_truth_path = self.test_labels_path + '/' + filename[:-4] + '.png'
            item_arr = np.asarray(Image.open(test_file_path))
            images_to_predict.append(item_arr)
            label_arr = np.asarray(Image.open(ground_truth_path))
            label_arr = label_arr.copy()
            #THIS IS SPECIFIC TO THIS CASE BECAUSE LABELS WERE 3D IMAGES WHERE CLASSES WERE ORGANIZED:
            # 0 - glomeruli, 170 - other parts of kidney tissue, 250 - background
            if self.number_of_classes == 2:
            #     label_arr[label_arr==170] = 1
            #     label_arr[label_arr==250] = 1
                label_arr = to_categorical(label_arr, num_classes=self.number_of_classes, dtype = 'uint8')
            else:
            #     label_arr[label_arr==170] = 1
            #     label_arr[label_arr==250] = 2
                label_arr = to_categorical(label_arr, num_classes=self.number_of_classes, dtype = 'uint8')
            ground_truth_array.append(label_arr)
        images_to_predict = np.asarray(images_to_predict)
        ground_truth_array = np.asarray(ground_truth_array)
        # ground_truth_array[ground_truth_array==170] = 1
        # ground_truth_array[ground_truth_array==250] = 2
        # label_arr = to_categorical(label_arr, num_classes=3, dtype = 'uint8')
        # images_to_predict = np.divide(images_to_predict, 255) #if inputs have to be normalized
        predicted_images = model.predict(images_to_predict, batch_size=1, verbose=1)
        # _, _, labels = next(os.walk(test_labels_path))
        # for label_name in labels:
        # 	test_file_path = test_labels_path + '/' + label_name
        # 	item_arr = np.asarray(Image.open(test_file_path))
        # 	ground_truth_array.append(item_arr)
        predicted_images_one_hot = []
        for i in range(predicted_images.shape[0]):
            img = predicted_images[i]
            #FOR STATISTICS***********************************************
            if self.number_of_classes == 2:
                for row in img:
                    for cell in row:
                        if cell[0] > cell[1]:
                            cell[0] = 1
                            cell[1] = 0
                        elif cell[1] > cell[0]:
                            cell[1] = 1
                            cell[0] = 0
            else:
                for row in img:
                    for cell in row:
                        if cell[0] > cell[1] and cell[0] > cell[2]:
                            cell[0] = 1
                            cell[1] = 0
                            cell[2] = 0
                        elif cell[1] > cell[0] and cell[1] > cell[2]:
                            cell[1] = 1
                            cell[0] = 0
                            cell[2] = 0
                        elif cell[2] > cell[1] and cell[2] > cell[0]:
                            cell[2] = 1
                            cell[0] = 0
                            cell[1] = 0
            predicted_images_one_hot.append(img)
        predicted_images_one_hot = np.asarray(predicted_images_one_hot)
        filenames_array = np.asarray(files)
        np.save(self.ground_truth_array_path, ground_truth_array)
        np.save(self.predicted_images_one_hot, predicted_images_one_hot)
        np.save(self.filenames_array_path, filenames_array)


    def metrics_for_each_image(self, predicted_folder, ground_truth_folder):
        '''CALCULATING METRICS BASED ON SAVED IMAGES'''
        print('**********************METRICS**********************')
        if self.number_of_classes == 2:
            print ('Confusion matrix - rows = actual, columns = predicted: 0 - {}, 1 - {}'.format(self.conf_matrix_labels[0],
                                                                                                          self.conf_matrix_labels[1]))
        else:
            print ('Confusion matrix - rows = actual, columns = predicted: 0 - {}, 1 - {}, 2 - {}'.format(self.conf_matrix_labels[0],
                                                                                                          self.conf_matrix_labels[1],
                                                                                                          self.conf_matrix_labels[2]))
        # _, _, files = next(os.walk(ground_truth_folder))
        filenames = np.load(self.filenames_array_path)
        filenames = filenames.tolist()
        for filename in filenames:
            ground_truth_item_path = self.ground_truth_folder + '/' + filename[:-4] + '.png'
            ground_truth_item = np.asarray(Image.open(ground_truth_item_path))
            predicted_item_path = self.predicted_folder + '/' + filename[:-4] + '.png'
            predicted_item = np.asarray(Image.open(predicted_item_path))
            predicted_item = predicted_item.copy()
            ground_truth_item = ground_truth_item.copy()
            #PREPARING ARRAYS TO CALCULATE CONFUSION MATRIX
            ground_truth_item[ground_truth_item==170] = 1
            if self.number_of_classes == 2:
                ground_truth_item[ground_truth_item==250] = 1
            else:
                ground_truth_item[ground_truth_item==250] = 2
            predicted_item_reshaped = np.reshape(predicted_item, (predicted_item.shape[0]**2, self.number_of_classes))
            predicted_item_a = np.argmax(predicted_item_reshaped, axis=1)
            ground_truth_item_reshaped = np.reshape(ground_truth_item, (ground_truth_item.shape[0]**2, 1))
            acc = accuracy_score(ground_truth_item_reshaped, predicted_item_a)
            class_report = classification_report(ground_truth_item_reshaped, predicted_item_a, labels = self.labels, target_names = self.conf_matrix_labels)
            conf_matrix = confusion_matrix(ground_truth_item_reshaped, predicted_item_a, labels = self.labels)
            #IN CALCULATING F1 SCORE, IT IS IMPORTANT TO CHECK RESULTS WHEN average = 'macro'!
            # f1_score_calc = f1_score(ground_truth_item_reshaped, predicted_item_a, labels = [0,1,2], average='micro')
            jacc_score = jaccard_score(ground_truth_item_reshaped, predicted_item_a, labels = self.labels,  average=None)
            print ('FOR IMAGE *************** {} ***************'.format(filename))
            print ('Confusion matrix')
            print (conf_matrix)
            print ('Accuracy')
            print (acc)
            print ('Classification report')
            print (class_report)
            print ('Jaccard index (IoU): {:.4f}'.format(jacc_score))

    def metrics_for_predicted_items(self):
        '''CALCULATING METRICS BASED ON EACH PREDICTED ITEM (ARRAY OF AN IMAGE)'''
        if self.number_of_classes == 2:
            print ('Confusion matrix - rows = actual, columns = predicted: 0 - {}, 1 - {}'.format(self.conf_matrix_labels[0],
                                                                                                  self.conf_matrix_labels[1]))
        else:
            print ('Confusion matrix - rows = actual, columns = predicted: 0 - {}, 1 - {}, 2 - {}'.format(self.conf_matrix_labels[0],
                                                                                                          self.conf_matrix_labels[1],
                                                                                                          self.conf_matrix_labels[2]))
        # _, _, files = next(os.walk(ground_truth_folder))
        statistics_per_image_path = self.model_stat_folder + '/for_each_image'
        os.makedirs(statistics_per_image_path)
        filenames = np.load(self.filenames_array_path)
        ground_truth_array = np.load(self.ground_truth_array_path)
        predicted_images_array = np.load(self.predicted_images_one_hot)
        filenames = filenames.tolist()
        image_counter = 0
        for filename in filenames:
            ground_truth_item = ground_truth_array[image_counter]
            ground_truth_item_reshaped = np.reshape(ground_truth_item, (ground_truth_item.shape[0]**2, self.number_of_classes))
            predicted_item = predicted_images_array[image_counter]
            predicted_item_reshaped = np.reshape(predicted_item, (predicted_item.shape[0]**2, self.number_of_classes))
            ground_truth_item_a = np.argmax(ground_truth_item_reshaped, axis=1)
            predicted_item_a = np.argmax(predicted_item_reshaped, axis=1)
            acc = accuracy_score(ground_truth_item_a, predicted_item_a)
            class_report = classification_report(ground_truth_item_a, predicted_item_a, labels = self.labels, target_names = self.conf_matrix_labels)
            conf_matrix = confusion_matrix(ground_truth_item_a, predicted_item_a, labels = self.labels)
            jacc_score = jaccard_score(ground_truth_item_a, predicted_item_a, labels = self.labels,  average=None)
            # dice_score = prediction.dice_coef(ground_truth_item_reshaped, predicted_item_reshaped)
            print ('FOR IMAGE *************** {} ***************'.format(filename))
            print ('Confusion matrix')
            print (conf_matrix)
            print ('Accuracy')
            print (acc)
            print ('Classification report')
            print (class_report)
            # print ('Dice score (F1_score): {:.4f}'.format(dice_score))
            # print ('Jaccard score (IoU): {:.4f}'.format(jacc_score))
            class_report_dict = classification_report(ground_truth_item_a, predicted_item_a, labels = self.labels, target_names = self.conf_matrix_labels,
                                                      output_dict = True)
            class_report_df = pd.DataFrame(class_report_dict).transpose().round(4)
            jacc_list = np.around(jacc_score, decimals = 4).tolist()
            jacc_list = jacc_list + ['NaN', 'NaN', 'NaN']
            if self.number_of_classes == 2:
                acc_list = [acc.round(4), 'NaN', 'NaN', 'NaN', 'NaN']
            else:
                acc_list = [acc.round(4), 'NaN', 'NaN', 'NaN', 'NaN', 'NaN']
            class_report_df['Accuracy'],  class_report_df['Jaccard'] = [acc_list, jacc_list]
            confusion_matrix_df = pd.DataFrame(data = conf_matrix, index = self.conf_matrix_labels, columns = self.conf_matrix_labels)
            # df_combined = pd.concat([class_report_df, confusion_matrix_df], axis =0, ignore_index=True)
            # ipdb.set_trace()
            class_report_csv_path = statistics_per_image_path + '/' + 'class_report_' + str(filename[:-4]) + '.csv'
            class_report_df.to_csv(class_report_csv_path, index = True, header=True)
            confusion_matrix_csv_path = statistics_per_image_path + '/' + 'confusion_matrix_' + str(filename[:-4]) + '.csv'
            confusion_matrix_df.to_csv(confusion_matrix_csv_path, index = True, header=True)
            image_counter += 1

    def metrics_for_whole_test(self):
        ground_truth_array = np.load(self.ground_truth_array_path)
        predicted_images_array = np.load(self.predicted_images_one_hot)
        ground_truth_array_reshaped = np.reshape(ground_truth_array, (ground_truth_array.shape[0]*ground_truth_array.shape[1]**2, self.number_of_classes))
        predicted_images_array_reshaped = np.reshape(predicted_images_array, (predicted_images_array.shape[0]*predicted_images_array.shape[1]**2, self.number_of_classes))
        ground_truth_array_a = np.argmax(ground_truth_array_reshaped, axis=1)
        predicted_images_array_a = np.argmax(predicted_images_array_reshaped, axis=1)
        acc = accuracy_score(ground_truth_array_a, predicted_images_array_a)
        class_report = classification_report(ground_truth_array_a, predicted_images_array_a, labels = self.labels, target_names = self.conf_matrix_labels)
        conf_matrix = confusion_matrix(ground_truth_array_a, predicted_images_array_a, labels = self.labels)
        jacc_score = jaccard_score(ground_truth_array_a, predicted_images_array_a, labels = self.labels, average=None)
        # dice_score = prediction.dice_coef(ground_truth_array_reshaped, predicted_images_array_reshaped)
        # print ('***************FOR ALL IMAGES METRICS ARE***************')
        # print ('Confusion matrix')
        # print (conf_matrix)
        # print ('Accuracy')
        # print (acc)
        # print ('Classification report')
        # print (class_report)
        # print ('Dice score (F1_score): {:.4f}'.format(dice_score))
        # print ('Jaccard index (IoU): {:.4f}'.format(jacc_score))
        class_report_dict = classification_report(ground_truth_array_a, predicted_images_array_a, labels = self.labels, target_names = self.conf_matrix_labels,
                                                  output_dict = True)
        class_report_df = pd.DataFrame(class_report_dict).transpose().round(4)
        jacc_list = np.around(jacc_score, decimals = 4).tolist()
        jacc_list = jacc_list + ['NaN', 'NaN', 'NaN']
        if self.number_of_classes == 2:
            acc_list = [acc.round(4), 'NaN', 'NaN', 'NaN', 'NaN']
        else:
            acc_list = [acc.round(4), 'NaN', 'NaN', 'NaN', 'NaN', 'NaN']
        class_report_df['Accuracy'],  class_report_df['Jaccard'] = [acc_list, jacc_list]
        confusion_matrix_df = pd.DataFrame(data = conf_matrix, index = self.conf_matrix_labels, columns = self.conf_matrix_labels)
        # df_combined = pd.concat([class_report_df, confusion_matrix_df], axis =0, ignore_index=True)
        class_report_csv_path = self.model_stat_folder + '/' + 'class_report.csv'
        class_report_df.to_csv(class_report_csv_path, index = True, header=True)
        confusion_matrix_csv_path = self.model_stat_folder + '/' + 'confusion_matrix.csv'
        confusion_matrix_df.to_csv(confusion_matrix_csv_path, index = True, header=True)

    def save_predictions_as_images(self):
        predicted_images_array = np.load(self.predicted_images_one_hot)
        filenames = np.load(self.filenames_array_path)
        filenames = filenames.tolist()
        for i in range(predicted_images_array.shape[0]):
            img = predicted_images_array[i]
            #FOR PRINTING IMAGES**************************************************
            if self.number_of_classes == 2:
                for row in img:
                    for cell in row:
                        if cell[0] >= 0.5:
                            cell[0] = 0
                            cell[1] = 50
                        elif cell[1] >= 0.5:
                            cell[1] = 150
                            cell[0] = 0
            else:
                for row in img:#np.nditer(img, op_flags=['readwrite']):
                    for cell in row:
                        if cell[0] >= 0.9:
                            cell[0] = 50
                            cell[1] = 0
                            cell[2] = 0
                        elif cell[1] >= 0.9:
                            cell[1] = 150
                            cell[0] = 0
                            cell[2] = 0
                        elif cell[2] >= 0.9:
                            cell[2] = 250
                            cell[0] = 0
                            cell[1] = 0
                # arr_mask[arr_mask==160]
                # img = array_to_img(img)
            img = Image.fromarray(img.astype(np.uint8))
            results_by_name_path = self.results_dir_path + '/' + filenames[i][:-4] + '.png'
            img.save(results_by_name_path)

if __name__ == '__main__':
    prediction = prediction_and_metrics()
    start = time.time()
    prediction.predict()
    end = time.time()
    print (':::::::::::::::::::::::::::::::Time needed for prediction:::::::::::::::::::::::::::::::')
    timer(start, end)

    start = time.time()
    prediction.metrics_for_predicted_items()
    end = time.time()
    print (':::::::::::::::::::::::::::::::Time needed for caclulating metrics:::::::::::::::::::::::::::::::')
    timer(start, end)

    start = time.time()
    prediction.save_predictions_as_images()
    end = time.time()
    print (':::::::::::::::::::::::::::::::Time needed for saving images:::::::::::::::::::::::::::::::')
    timer(start, end)

    prediction.metrics_for_whole_test()