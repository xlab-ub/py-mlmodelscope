class DataLoader:
  def __init__(self, dataset, batch_size):
    self.dataset = dataset
    # print(dataset)
    self.batch_size  = batch_size
    self.idx = 0

  ### Changjae Lee @ 2023-02-16 
  def __len__(self): 
    return -(-len(self.dataset) // self.batch_size) 
  
  def __iter__(self):
    return self
  
  def __next__(self):
    if self.idx >= len(self.dataset):
      raise StopIteration
    else:
      start = self.idx
      end = min(self.idx + self.batch_size, len(self.dataset))
      data = [None] * (end - start)
      for i in range(end - start):
        data[i] = self.dataset[start + i]
      self.idx += self.batch_size
      return data

  def reset(self): 
    self.idx = 0 
    return 