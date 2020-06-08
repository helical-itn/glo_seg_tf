# Glomeruli segmentation (U-Net)
Using deep learning (U-Net) in segmenting glomeruli on histopathological images of kidney tissue

## Used data

Images which are used as inputs are crops of histopathological WSI's (whole slide images) of the kidney  tissue. Used WSI's are brightfield microscopic images. Used stainings: Haematoxylin and Eosin (HE), Periodic Acid–Schiff's (PAS) and Acid Fuchsin–Orange G Stain (SFOG).

## Labels

Labels are 3D images (meaning images with more than 1 band/channel) where labels are organized in following order: 0 - glomeruli, 170 - other parts of kidney tissue, 250 - background. Those numbers where chosen for practical reasons  during data preprocessing. (e.g. when images with pixels with those values are visualized it is easy to distinguish between classes)

### train_and_predict.py

- main script, it will do: data loading, training, predicting on test set and calculating metrics for predicted labels

### segmentation_results_and_metrics.py

- it consists of functions which are used to do prediction and calculate metrics

### unet_elemetns.py

- unet architectures, segmentation loss functions...

### img_augmentation.py

- image augmentation wasn't done live (during training) even though that is the most efficient way of doing it. I wanted to check how each augmentation will affect each of the image. Also I wanted to completely randomize images and their augmented versions in training set before training starts.
def visualize is just for testing different augmentations
- script could and should be written more efficient but this was written just to get the job done asap

### read_n_play.py

- quick and dirty solutions


