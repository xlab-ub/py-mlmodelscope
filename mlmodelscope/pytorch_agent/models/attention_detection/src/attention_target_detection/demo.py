import argparse, os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
import pandas as pd
import numpy as np
import dlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import cv2
from glob import glob
from model import ModelSpatial
from utils import imutils, evaluation
from config import *


#parser = argparse.ArgumentParser()
#parser.add_argument('--model_weights', type=str, help='model weights', default='model_demo.pt')
#parser.add_argument('--image_dir', type=str, help='images', default='data/demo/frames')
#parser.add_argument('--head', type=str, help='head bounding boxes', default='data/demo/person1.txt')
#parser.add_argument('--vis_mode', type=str, help='heatmap or arrow', default='heatmap')
#parser.add_argument('--out_threshold', type=int, help='out-of-frame target dicision threshold', default=100)
#args = parser.parse_args()


def _get_transform():
    transform_list = []
    transform_list.append(transforms.Resize((input_resolution, input_resolution)))
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    return transforms.Compose(transform_list)



def move_figure(f, x, y): #https://matplotlib.org/stable/users/explain/figure/backends.html
    """Move figure's upper left corner to pixel (x, y)"""
    backend = plt.get_backend()
    if backend == 'TkAgg':
        f.canvas.manager.window.wm_geometry("+%d+%d" % (x, y))
    elif backend == 'WXAgg':
        f.canvas.manager.window.SetPosition((x, y))
    else:
        f.canvas.manager.window.move(x, y)



def makeFrames(maxFrames = 10, video_path="gordon_ramsay.avi", frames_dir="data/frames"):
    cap = cv2.VideoCapture(video_path)
    
    # Ensure the frames directory exists
    os.makedirs(frames_dir, exist_ok=True)

    frame_count = 0
    success, image = cap.read()

    while success and frame_count < maxFrames:
        # Define frame path
        frame_path = os.path.join(frames_dir, "frame%d.jpg" % frame_count)
        
        # Saves the frames with frame-count
        cv2.imwrite(frame_path, image)
        
        # Read the next frame
        success, image = cap.read()
        frame_count += 1

    return frame_count

def makeCSV():
    cnn_face_detector = dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")

    frames = glob("data/frames/*.jpg")
    frames.sort()

    # Prepare a list to hold all the bounding box data
    data = []

    count = 0
    for frame_path in frames:
        print("\nWorking on frame: " + str(count))
        frame = cv2.imread(frame_path)
        frame_raw = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        dets = cnn_face_detector(frame_raw, 1)
        if dets:
            # Assuming the first detection is what we want
            d = dets[0]
            l = d.rect.left()
            r = d.rect.right()
            t = d.rect.top()
            b = d.rect.bottom()
            # Expand the bounding box a bit (here using the same factor you provided)
            l -= (r-l)*0.2
            r += (r-l)*0.2
            t -= (b-t)*0.2
            b += (b-t)*0.2

            # Append the frame identifier and bounding box to the data list
            frame_id = os.path.basename(frame_path)  # Or however you wish to identify frames
            data.append([frame_id, l, t, r, b])
        count += 1
    df = pd.DataFrame(data, columns=['frame', 'left', 'top', 'right', 'bottom'])
    df.to_csv("data/csv/head.csv", index=False)

    return df

def cleanUp():
    # Remove intermediate files
    files = glob('data/frames/*.jpg')
    for f in files:
        os.remove(f)

    os.remove("data/csv/head.csv")

    return None

    



def run(df):
    vis_mode = "heatmap"


    # set up data transformation
    test_transforms = _get_transform()

    model = ModelSpatial()
    model_dict = model.state_dict()
    pretrained_dict = torch.load("mlmodelscope/pytorch_agent/models/attention_detection/src/attention_target_detection/model_demo.pt")
    pretrained_dict = pretrained_dict['model']
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    model.cuda()
    model.train(False)

    with torch.no_grad():
        for i in df.index:
            frame_raw = Image.open(os.path.join("data/frames", i))
            frame_raw = frame_raw.convert('RGB')
            width, height = frame_raw.size

            head_box = [df.loc[i,'left'], df.loc[i,'top'], df.loc[i,'right'], df.loc[i,'bottom']]

            head = frame_raw.crop((head_box)) # head crop

            head = test_transforms(head) # transform inputs
            frame = test_transforms(frame_raw)
            head_channel = imutils.get_head_box_channel(head_box[0], head_box[1], head_box[2], head_box[3], width, height,
                                                        resolution=input_resolution).unsqueeze(0)

            head = head.unsqueeze(0).cuda()
            frame = frame.unsqueeze(0).cuda()
            head_channel = head_channel.unsqueeze(0).cuda()

            # forward pass
            raw_hm, _, inout = model(frame, head_channel, head)

            # heatmap modulation
            raw_hm = raw_hm.cpu().detach().numpy() * 255
            raw_hm = raw_hm.squeeze()
            inout = inout.cpu().detach().numpy()
            inout = 1 / (1 + np.exp(-inout))
            inout = (1 - inout) * 255
            norm_map = cv2.resize(raw_hm, (height, width)) - inout

            # vis
            plt.close()
            fig = plt.figure()
            move_figure(fig, 0, 0)
            plt.axis('off')
            plt.imshow(frame_raw)

            ax = plt.gca()
            rect = patches.Rectangle((head_box[0], head_box[1]), head_box[2]-head_box[0], head_box[3]-head_box[1], linewidth=2, edgecolor=(0,1,0), facecolor='none')
            ax.add_patch(rect)

            if vis_mode == 'arrow':
                if inout < 100: # in-frame gaze
                    pred_x, pred_y = evaluation.argmax_pts(raw_hm)
                    norm_p = [pred_x/output_resolution, pred_y/output_resolution]
                    circ = patches.Circle((norm_p[0]*width, norm_p[1]*height), height/50.0, facecolor=(0,1,0), edgecolor='none')
                    ax.add_patch(circ)
                    plt.plot((norm_p[0]*width,(head_box[0]+head_box[2])/2), (norm_p[1]*height,(head_box[1]+head_box[3])/2), '-', color=(0,1,0,1))
            else:
                plt.imshow(norm_map, cmap = 'jet', alpha=0.2, vmin=0, vmax=255)

            plt.show(block=False)
            plt.pause(0.2)

        print('DONE!')


if __name__ == "__main__":
    run()




