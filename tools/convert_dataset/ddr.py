import os
import sys
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import argparse
import cv2
import numpy as np

'''
Please, first rename the original directories of DDR dataset to:

DDR
|——images
|   |——test
|   |——val
|   |——train
|——labels
|   |——test
|   |   |——EX
|   |   |——HE
|   |   |——MA
|   |   |——SE
|   |——val
|   |   |——EX
|   |   |——HE
|   |   |——MA
|   |   |——SE
|   |——train
|   |   |——EX
|   |   |——HE
|   |   |——MA
|   |   |——SE

Run the following code to generate .png labels for training
'''

CLASSES = dict(
    EX=1,
    HE=2,
    SE=3,
    MA=4,
)

def parse_args():
    parser = argparse.ArgumentParser(description='Reconstruct the DDR dataset')
    parser.add_argument('--root', default='./data/DDR')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    root = args.root

    image_root = os.path.join(root,'images')
    label_root = os.path.join(root,'labels')
    for subset in os.listdir(image_root):
        for eachimage in os.listdir(os.path.join(image_root,subset)):
            
            img = cv2.imread(os.path.join(image_root,subset,eachimage))
            fused_gt = np.zeros((img.shape[0],img.shape[1]),dtype=np.uint8)
            for each_class in CLASSES:
                each_class_gt_path = os.path.join(label_root,subset,each_class,eachimage.replace('.jpg','.tif'))
                each_class_gt = cv2.imread(each_class_gt_path,cv2.IMREAD_GRAYSCALE)
                fused_gt[each_class_gt==255] = CLASSES[each_class]

            label_path = os.path.join(label_root,subset,eachimage.replace('.jpg','.png'))
            cv2.imwrite(label_path,fused_gt)

if __name__ == '__main__':
    main()
