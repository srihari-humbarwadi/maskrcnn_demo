## Running the code
 - git clone https://github.com/matterport/Mask_RCNN.git
 - python3 setup.py install
 - download the maskrcnn pretrained weights and place it in the root of this repository [download weights](https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5)
 - python calculate_accuracy.py, to get the output json or python calculate_accuracy_only_foreground.py, to get the output
 - pip install jupyterlab, and then run the required noteboooks

## Notes
 - Since majority of the instances can be of a small scale with respect to the full image, pixel wise accuracy that would take into account both the foreground and background will not give us usefull insights, it would simply measure how good the model is at detecting the majority class, here the majority class would be the background.
 - An alternative way to calculate the pixel wise accuracy would be including only the foreground pixels, any misclassification would be penalized and would add up in the final score. This would serve as a more robust method when the scales of object instances are small compared to the entire image.
 
 
 ```
 pixel wise accuracy for foreground
{
  "Car # 10": 0.42379958246346555,
  "Car # 21": 0.5816869300911854,
  "Car # 11": 0.8622589531680441,
  "Car # 22": 0.9636505904111538,
  "Car # 16": 0.94824016563147,
  "Car # 2": 0.9203966764942375,
  "Car # 23": 0.9587470373454615,
  "Car # 12": 0.6556962025316456,
  "building": 0,
  "ground": 0,
  "licenseplate": 0,
  "Car # 4": 0.7362986497220015,
  "vegetation": 0,
  "Car # 17": 0.9210992907801419,
  "road": 0,
  "Car # 20": 0.5851950650216738,
  "Car # 24": 0.9482000409081611,
  "Car # 18": 0.8512396694214877,
  "Car # 13": 0.7192028985507246,
  "Car # 9": 0.8086560364464692,
  "Car # 7": 0.7785588752196837,
  "Car # 19": 0.548951048951049,
  "tram track": 0,
  "Car # 1": 0.8594942748091603,
  "Car # 3": 0.8814432989690721,
  "car_group": 0.18532695374800637,
  "Car # 15": 0.27439024390243905,
  "rail track": 0,
  "Car # 6": 0.5714285714285714,
  "Car # 5": 0.8701298701298701,
  "Truck # 2": 0,
  "sidewalk": 0,
  "Car # 14": 0.35809018567639256,
  "Car # 8": 0.7613168724279835
}
``` 
 
 
```
 pixel wise accuracy for both background and foreground
 {
  "Car # 10": 0.9997907021604938,
  "Car # 21": 0.9988884066358025,
  "Car # 11": 0.9998398919753086,
  "Car # 22": 0.9992626350308642,
  "Car # 16": 0.9998562885802469,
  "Car # 2": 0.9998172260802469,
  "Car # 23": 0.9991661844135803,
  "Car # 12": 0.9997309027777778,
  "building": 0,
  "ground": 0,
  "licenseplate": 0,
  "Car # 4": 0.9998008294753087,
  "vegetation": 0,
  "Car # 17": 0.9997815393518519,
  "road": 0,
  "Car # 20": 0.9993065200617284,
  "Car # 24": 0.9989940200617284,
  "Car # 18": 0.9998292824074074,
  "Car # 13": 0.9997882908950617,
  "Car # 9": 0.9996903935185185,
  "Car # 7": 0.9996990740740741,
  "Car # 19": 0.9994261188271605,
  "tram track": 0,
  "Car # 1": 0.9989395254629629,
  "Car # 3": 0.9997559799382716,
  "car_group": 0.9987683256172839,
  "Car # 15": 0.9991584683641975,
  "rail track": 0,
  "Car # 6": 0.9995187114197531,
  "Car # 5": 0.9996913580246913,
  "Truck # 2": 0,
  "sidewalk": 0,
  "Car # 14": 0.9989853395061729,
  "Car # 8": 0.9997005208333334
}
```
