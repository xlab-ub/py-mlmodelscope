from ..pytorch_abc import PyTorchAbstractClass

import torch, cv2, dlib, os, sys

import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from torchvision import transforms

sys.path.append(os.path.dirname(__file__))
from src.attention_target_detection.model import ModelSpatial
from src.attention_target_detection.utils import imutils, evaluation
sys.path.pop()



class Attention_Target_Detection(PyTorchAbstractClass): 

    def __init__(self):
        current_dir = os.path.dirname(__file__)
        self.model_weights = os.path.join(current_dir,'src/attention_target_detection/model_demo.pt')
        self.frame_dir = os.path.join(current_dir,'src/attention_target_detection/data/frames')
        self.input_resolution = 224
        self.output_resolution = 64
        self.vis_mode = "heatmap"
        self.out_threshold = 100

        
        self.CNN_FACE_MODEL = os.path.join(current_dir, 'src/attention_target_detection/mmod_human_face_detector.dat')
        # from http://dlib.net/files/mmod_human_face_detector.dat.bz2

        # Initialize Transformations
        transform_list = []
        transform_list.append(transforms.Resize((self.input_resolution, self.input_resolution)))
        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

        self.test_transform = transforms.Compose(transform_list)

        self.model = ModelSpatial()
        model_dict = self.model.state_dict()
        pretrained_dict = torch.load(self.model_weights)
        pretrained_dict = pretrained_dict['model']
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict)

        return None
       
    def preprocess(self, input_data):
        src_video_path = input_data
        print(src_video_path)

        cap = cv2.VideoCapture("src/gordon_ramsay.avi")
        

        #EXTRACT FRAMES INTO TEMP FRAME FOLDER
        frame_count = 0
        success = 1
    
        while success:
            # cap object calls read 
            # function extract frames 
            success, image = cap.read()
     
            # Saves the frames with frame-count 
            cv2.imwrite("src/attention_target_detection/data/frames/frame%d.jpg" % frame_count, image)
            frame_count += 1
             
        return [frame_count]
        

    def predict(self, input_data):

        self.model.cuda()
        self.model.train(False)
       
        cnn_face_detector = dlib.cnn_face_detection_model_v1(self.CNN_FACE_MODEL)
        frame_count = input_data[0]

        with torch.no_grad():
            for i in range(frame_count):
                frame_raw = Image.open(os.path.join("src/attention_target_detection/data/frames/frame", str(i), ".*"))
                frame_raw = frame_raw.convert('RGB')
                width, height = frame_raw.size

                
                dets = cnn_face_detector(frame_raw, 1)
                d = dets[0]
                l = d.rect.left()
                r = d.rect.right()
                t = d.rect.top()
                b = d.rect.bottom()
                # expand a bit
                l -= (r-l)*0.2
                r += (r-l)*0.2
                t -= (b-t)*0.2
                b += (b-t)*0.2

                head_box = [l,t,r,b]


                head = frame_raw.crop((head_box)) # head crop

                head = self.test_transform(head) # transform inputs
                frame = self.test_transform(frame_raw)
                head_channel = imutils.get_head_box_channel(head_box[0], head_box[1], head_box[2], head_box[3], width, height,
                                                            resolution=self.input_resolution).unsqueeze(0)

                head = head.unsqueeze(0).cuda()
                frame = frame.unsqueeze(0).cuda()
                head_channel = head_channel.unsqueeze(0).cuda()

                # forward pass
                raw_hm, _, inout = self.model(frame, head_channel, head)

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

                backend = plt.get_backend()
                if backend == 'TkAgg':
                    fig.canvas.manager.window.wm_geometry("+%d+%d" % (0, 0))
                elif backend == 'WXAgg':
                    fig.canvas.manager.window.SetPosition((0, 0))
                else:
                    fig.canvas.manager.window.move(0, 0)

                plt.axis('off')
                plt.imshow(frame_raw)

                ax = plt.gca()
                rect = patches.Rectangle((head_box[0], head_box[1]), head_box[2]-head_box[0], head_box[3]-head_box[1], linewidth=2, edgecolor=(0,1,0), facecolor='none')
                ax.add_patch(rect)

                if self.vis_mode == 'arrow':
                    if inout < self.out_threshold: # in-frame gaze
                        pred_x, pred_y = evaluation.argmax_pts(raw_hm)
                        norm_p = [pred_x/self.output_resolution, pred_y/self.output_resolution]
                        circ = patches.Circle((norm_p[0]*width, norm_p[1]*height), height/50.0, facecolor=(0,1,0), edgecolor='none')
                        ax.add_patch(circ)
                        plt.plot((norm_p[0]*width,(head_box[0]+head_box[2])/2), (norm_p[1]*height,(head_box[1]+head_box[3])/2), '-', color=(0,1,0,1))
                else:
                    plt.imshow(norm_map, cmap = 'jet', alpha=0.2, vmin=0, vmax=255)

                plt.show(block=False)
                plt.pause(0.2)

        print('DONE!')
        
        return None
    
    def postprocess(self, model_output):


        # Remove intermediate files
        files = glob.glob('src/attention_target_detection/data/frames/*.jpg')
        for f in files:
            os.remove(f)

        print("\nYour outputs are saved at: \n")
        return [0]

