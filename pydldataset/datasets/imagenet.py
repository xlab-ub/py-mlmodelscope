import os 
import re 
import time

class ImageNet: 
    dataset = {}

    def __init__(self, count):
        data_dir = os.environ['DATA_DIR']
        self.image_list = []
        data_dir = os.path.expanduser(data_dir)
        val_path = os.path.join(data_dir, "val_map.txt") 
        self.last_loaded = -1
        self.image_list = []
        # if not os.path.exists(val_path): 
        #     with open(os.path.join(data_dir, "ILSVRC2012_validation_ground_truth.txt"), 'r') as f: 
        #         labels = f.readlines()
        #     vals = [None] * 50000 
        #     with open(val_path, 'w') as f: 
        #         # for root, dirs, files in os.walk(os.path.join(data_dir, "ILSVRC2012_val_")): 
        #         for root, dirs, files in os.walk(data_dir): 
        #             # print(f'root: {root}, dirs: {dirs}, files: {files}')
        #             for file in files: 
        #                 if file.endswith(".JPEG"): 
        #                     image_name = os.path.join(root, file) 
        #                     file_num = int(file.split("_")[-1].split(".")[0]) 
        #                     label = labels[file_num - 1].strip() 
        #                     # f.write(image_name + " " + label + "\n")
        #                     vals[file_num - 1] = image_name + " " + label + "\n"
        #         f.writelines(vals)
        
        res = 0
        with open(val_path, "r") as f: 
            for s in f: 
                image_name, label = re.split(r"\s+", s.strip())
                src = os.path.join(data_dir, image_name)
                self.image_list.append(src)
                res += 1
                if count > 0 and res >= count:
                    break
        # return res
    
    def __len__(self):
        return len(self.image_list) 
    
    def __getitem__(self, idx):
        return self.image_list[idx] 
    
    def get_item_count(self):
        return len(self.image_list) 

    def load(self, sample_list):
        for sample in sample_list:
            self.dataset[sample] = self.image_list[sample]
        self.last_loaded = time.time()


    def unload(self, sample_list):
        self.dataset = {}
    
    def get_samples(self, id_list):
        data = [self.dataset[id] for id in id_list] 
        return data 

def init(count):
  return ImageNet(count) 