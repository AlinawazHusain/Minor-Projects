import torchvision.transforms as transform
from PIL import Image
import torch
from tqdm import trange
import torch.optim as optim


image_size = 256
device = 'cuda' if torch.cuda.is_available else 'cpu'

def load_image(image):
    img = Image.open(image)
    img = Transform(img).unsqueeze(0)
    return img


Transform = transform.Compose(
    [
        transform.Resize((image_size , image_size)),
        transform.ToTensor()
    ]
)



