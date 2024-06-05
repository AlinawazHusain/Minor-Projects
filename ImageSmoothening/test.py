import model
import utils
from PIL import Image
import torch
import numpy as np

path_to_img = 'Face/1.jpg'
model_path = 'saved_model_700.pth.tar'


print("============>>>>>>>  Loading Model")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
Model = model.Model().to(device)
Model.load_state_dict(torch.load(model_path , map_location=device)['state_dict'])

print('===============>>>>>> Loading Image')

img = Image.open(path_to_img)
img = utils.TRANSFORM(img)
img = img.to(torch.float32).to(device)

print("==========>>>>>>> transformation Started ")
with torch.no_grad():
    Model.eval()
    pred = Model(img)
    # real_img = img.squeeze(0).permute(1, 2, 0).detach().numpy()
    numpy_image_pred = pred.squeeze(0).permute(1, 2, 0).detach().numpy()
    # read_img = (real_img).astype(np.uint8)
    numpy_image_pred = (numpy_image_pred).astype(np.uint8)
    # real_img = Image.fromarray(real_img)
    pil_image_pred = Image.fromarray(numpy_image_pred)
    # real_img.save('real_image.jpg')
    pil_image_pred.save("output_image.jpg")

