## Steps followed to calculate the pixel wise accuracy for each object instance
 - Convert the hex color code to RGB, which is easier to work with
 - For each object instance
   - Get the ground truth (labelled by annotators) mask for object instance, identified uniquely by the RGB color code.
   - Extract the pixels from the predictions array from the same roi as that of the mask got in the previous step.
   - Use the channel with the maximum number of foreground predictions, this can be done by summing the prediction array along the first axis and then picking the index with maximum sum.
   - Keep track of the channels that are previously used.
   - The required prediction mask for the current object instance can then be got from the prediction array by using the channel got in the previous step.
   - Compute the pixel wise accuracy with the formula.
       <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;accuracy=\frac{TP&plus;TN}{TP&plus;TN&plus;FP&plus;FN}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;accuracy=\frac{TP&plus;TN}{TP&plus;TN&plus;FP&plus;FN}" title="accuracy=\frac{TP+TN}{TP+TN+FP+FN}" /></a>
    - TP represents foreground pixels that are correctly predicted
    - TN represents background pixels that are correctly predicted
    - FP represents background pixels that are wrongly predicted
    - FN represents foreground pixels that are wrongle predictes
  - The accuracy formula can be simplfied to (correctly classified pixels) / (total pixels). This can be computed using element wise comparision with the ground truth, and then taking the mean along first axis.
  - Get the predicted class name using the channel to index the class_ids list.
  - If the normalized class name and predicted class name do no match, set accuracy to zero.
    - If a given class name cannot be mapped to coco dataset, all its instances should have their accuracies set to zero
  - If the current channel has already be used by a previous object
   - Set the accuracy to 0 if the respective class in present in the multiple instance class list

## Running the code
 - git clone https://github.com/matterport/Mask_RCNN.git
 - python3 setup.py install
 - download the maskrcnn pretrained weights and place it in the root of this repository [download weights](https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5)
 - python calculate_accuracy.py, to get the output json or 
 - pip install jupyterlab, and then run the calculate_accuracy.ipynb notebook
