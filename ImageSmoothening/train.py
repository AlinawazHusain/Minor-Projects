import torch
import torch.optim as optim
import torch.nn as nn
import model
from torch.utils.data import random_split
import dataloader
from torch.utils.data import DataLoader
from tqdm import tqdm
import utils
import numpy as np
import os
from PIL import Image
import pandas as pd

LEARNING_RATE = 1e-4
EPOCHS = 501
ROOT_DIR = 'Face'
CSV = 'data.csv'

data = pd.read_csv(CSV)
train_size = int(0.8 * len(data))
val_size = len(data) - train_size


Model = model.Model()
optimizer = optim.Adam(Model.parameters() , lr = LEARNING_RATE )
loss_fn = nn.MSELoss()


train_loader = DataLoader(
    dataloader.CustData(
        ROOT_DIR,
        CSV,
        utils.TRANSFORM,
        0,
        train_size+1
    ),
    batch_size = 8,
    shuffle = True ,
)

val_loader = DataLoader(
    dataloader.CustData(
        ROOT_DIR,
        CSV,
        utils.TRANSFORM,
        train_size+1,
        -1
    ),
    batch_size = 1,
    shuffle = False ,
)


Model.train()
for epoch in range(EPOCHS):
    pbar = tqdm(train_loader , desc = f"Epoch {epoch}/{EPOCHS}")
    epoch_loss = 0
    for _ , (x , y) in enumerate(pbar):
        x = x.to(torch.float32)  
        y = y.to(torch.float32) 
        pred = Model(x)
        loss = loss_fn(pred , y) 
        epoch_loss += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pbar.set_postfix({'Loss' : loss.item()})
    print(f"Loss ->.. {epoch_loss}")
    if epoch%25 == 0 and epoch !=0:
        print("================>>>>>>>>>>>>>>>> SAVING TESTING IMAGES  <<<<<<<<<<<<<<<================")
        with torch.no_grad():
            Model.eval()
            os.makedirs(f'Outputs/epoch{epoch}' , exist_ok=True)
            val_loss = 0
            for i , (x , y) in enumerate(val_loader):
                x = x.to(torch.float32) 
                y = y.to(torch.float32) 

                pred = Model(x)
                loss = loss_fn(pred, y)
                val_loss += loss.item()

                if i>10:
                    continue
                else:
                    numpy_image = pred.squeeze(0).permute(1, 2, 0).detach().numpy()
                    numpy_image = (numpy_image).astype(np.uint8)
                    pil_image = Image.fromarray(numpy_image)
                    pil_image.save(f"Outputs/epoch{epoch}/output_image{i}.jpg")
        Model.train()
        print(f"Val Loss :=========>>>>>  {val_loss}")

    if epoch %50 == 0 and epoch !=0:
        state = {
            "state_dict" : Model.state_dict(),
            "optimizer" : optimizer.state_dict()
        }

        torch.save(state ,f'saved_model_{epoch}.pth.tar')

    