#!/usr/bin/env python


import os.path
import shelve
import pickle
import numpy as np
from collections import defaultdict
import cv2
from pprint import pprint
import warnings
import glob
import json
import argparse


def getUserKeyWithConfirmation(image, windowName, imageClasses):
    import cv2
    while True:
        userKey1 = cv2.waitKey(0)
        if(userKey1 in imageClasses):
            pass
            #newImage = putTextOnImage(image, imageClasses[userKey1])
            #cv2.imshow(windowName, newImage)
        elif(userKey1==27):
            newImage = putTextOnImage(image, 'quit')
            cv2.imshow(windowName, newImage)
        else:
            newImage = putTextOnImage(image, 'Try again')
            cv2.imshow(windowName, image)
            continue

        return imageClasses[userKey1]

def draw_images(images, max_y, max_x):
    import cv2

    blank_image = np.zeros((max_y * 2, max_x * 2, 3), dtype=np.uint8)
    for i,image in enumerate(images):
        x = image.shape[1]
        y = image.shape[0]
        x_offset = i * max_x
        y_offset = 0
        if x_offset > (max_x):
            x_offset = (i - 2) * max_x
            y_offset = max_y
        if int((max_y / y) * x) <= max_x and y < max_y:
            new_y = int((max_y / y) * y)
            new_x = int((max_y / y) * x)
        elif int((max_x / x) * y) <= max_y and x < max_x:
            new_x = int((max_x / x) * x)
            new_y = int((max_x / x) * y)
        else:
            new_x = int((max_x / float(x)) * x)
            new_y = int((max_y / float(y)) * y)
        blank_image[y_offset: y_offset + new_y, x_offset: x_offset + new_x] = cv2.resize(image, ( new_x, new_y))
    return blank_image


def read_image(image_name):
    return cv2.imread(image_name)


def deep_clone(obj):
    return pickle.loads(pickle.dumps(obj))

def draw_polygons_on_image(image, polygons, color):
    import cv2
    for polygon in polygons:
        points = np.array([polygon], dtype=np.int32)
        cv2.polylines(image, points, color=color, isClosed=True, thickness=2)

def putTextOnImage(image, text):
    import cv2
    newImage = np.copy(image)
    length = 40
    counter = 0
    for t in [text[i:i + length] for i in range(0, len(text), length)]:
        cv2.putText(newImage, t, (5, (50 + (50 * counter))), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), thickness=2)
        counter += 1
    return newImage

def select_from_image_tiles(images):
    import cv2
    max_y = 450
    max_x = 600
    
    # max_y = 700
    # max_x = 1000

    selected_index = None
    tile_image = draw_images(images, max_y, max_x)
    while True:
        cv2.imshow('choose the best', tile_image)
        result = getUserKeyWithConfirmation(tile_image,'aoeu', {63234: 'left', 63232:'up', 63235:'right', 63233:'down', 13:'accept', 83:'superSkip', 32:'skip', 116:'imageTitle', 109:'misc', 104:'headless',114:'regionLabel', 127:'back', 99:'imageCaption', 82:'regionLabel', 111:'ownObject', 115:'sectionTitle', 105:'intraobjectlinkage'})
        complete_results = ['accept', 'misc', 'headless', 'reverse', 'skip', 'back', 'imageCaption', 'regionLabel', 'ownObject', 'sectionTitle', 'intraobjectlinkage', 'superSkip', 'imageTitle']
        if selected_index is not None and result in complete_results:
            return (selected_index, result)
        else:
            selected_index = selected_index or 0
            if result == 'down':
                selected_index = selected_index + 2 if selected_index < 2 else selected_index
            elif result == 'right':
                selected_index = selected_index + 1 if selected_index < 3 else selected_index
            elif result == 'left':
                selected_index = selected_index - 1 if selected_index > 0 else selected_index
            elif result == 'up':
                selected_index = selected_index - 2 if selected_index >= 1 else selected_index

            tile_image = draw_images(images, max_y, max_x)
            if selected_index > 1:
                start_x = (selected_index - 2) * max_x
                end_x = start_x + max_x
                start_y = max_y
                end_y = max_y *2 
            else:
                start_x = selected_index * max_x
                end_x = start_x + max_x
                start_y = 0
                end_y = max_y
            cv2.rectangle(tile_image, (start_x, start_y), (end_x, end_y), color=(0, 0, 178), thickness=5)


def review_images(cat_dir_path, outfile, resume):
    diagram_images = defaultdict(list)
    cat_dir_path = './all_turkers_agree/'
    for g in glob.glob(cat_dir_path + '*'): 
        image_name = os.path.splitext(g.split('/')[-1])[0]
        image_id = image_name.split('_')[-2]
        diagram_images[image_id].append(image_name)
    idx = int(resume)
    for image_id, images in diagram_images.items()[idx:]:
        print images[0]
        read_images = [putTextOnImage(read_image(cat_dir_path + image + '.png'), image.split('_')[-1]) for image in images][::-1]
        read_images[1], read_images[3] = read_images[3], read_images[1]        
        selected_idx = select_from_image_tiles(read_images)
        if selected_idx[1] in ['skip', 'accept']:
            with open(outfile, 'a') as f:
                f.write(', '.join([str(idx), images[0].replace('_1', ''), str(selected_idx[0]), '\n']))
        elif selected_idx[1] == 'back':
            with open(outfile, 'a') as f:
                f.write(', '.join([str(idx), images[0].replace('_1', ''), 'redo', '\n']))
        idx += 1

def main():
    parser = argparse.ArgumentParser(description='Review text localization')
    parser.add_argument('imgdir', help='path to diagram images', type=str)
    parser.add_argument('outfile', help='file to write output', type=str)
    parser.add_argument('resume', help='resume from index', type=str)    
    args = parser.parse_args()
    review_images(args.imgdir, args.outfile, args.resume)

if __name__ == "__main__":
    main()

