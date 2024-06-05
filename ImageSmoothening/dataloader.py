from torch.utils.data import DataLoader , Dataset
from PIL import Image
import pandas as pd
import utils



class CustData(Dataset):
    def __init__(self ,
                 root_dir ,
                 csv_file ,
                 transform ,
                 start_lim,
                 end_lim):
        super(CustData , self).__init__()

        self.root_dir = root_dir
        self.csv = pd.read_csv(csv_file).iloc[start_lim:end_lim,:]
        self.transform = transform

        
    def __len__(self):
        return len(self.csv)
    
    def __getitem__(self, idx):

        x = Image.open(self.root_dir + '/' + self.csv.iloc[idx , 0])
        y = Image.open(self.root_dir + '/' + self.csv.iloc[idx , 1])

        x = self.transform(x)
        y = self.transform(y)

        return (x, y)

# dataloader = DataLoader(
#     CustData(
#         'Face',
#         'data.csv',
#         utils.TRANSFORM
#         ,
#         0,
#         -1
#     )
# )

# count = 0
# for _ , (x , y) in enumerate(dataloader):
#     count+=1

# print(count)
    
