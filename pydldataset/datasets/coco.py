import os 
import json 

class COCO: 
    dataset = {}
    image_list = []

    def __init__(self, count):
        data_dir = os.environ['DATA_DIR']
        data_dir = os.path.expanduser(data_dir)
        annotation_file = os.path.join(data_dir, "annotations/instances_val2017.json")
        with open(annotation_file, "r") as f:
            coco = json.load(f)
        res = 0
        for i in coco["images"]:
            self.image_list.append(os.path.join(data_dir, "val2017", i["file_name"]))
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

    def unload(self, sample_list):
        self.dataset = {}
    
    def get_samples(self, id_list):
        data = [self.dataset[id] for id in id_list] 
        return data 

def init(count):
  return COCO(count) 