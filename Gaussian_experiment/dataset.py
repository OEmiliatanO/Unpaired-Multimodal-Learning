from torch.utils.data import Dataset, DataLoader

class UnpairedDataset(Dataset):
    def __init__(self, data_x, data_y):
        self.data_x = data_x
        self.data_y = data_y
        self.len_x = len(data_x)
        self.len_y = len(data_y)
        self.length = max(self.len_x, self.len_y)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        idx_x = idx % self.len_x
        idx_y = idx % self.len_y
        
        return {'x': self.data_x[idx_x], 'y': self.data_y[idx_y]}