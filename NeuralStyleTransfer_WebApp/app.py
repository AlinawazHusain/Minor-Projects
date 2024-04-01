import streamlit as st
import torch
import numpy as np
import torch.optim as optim
from tqdm import trange
from PIL import Image
import warnings
from utils import load_image
from model import Model
warnings.filterwarnings('ignore')


def tensor_to_image(tensor):
    numpy_img = tensor.cpu().detach().numpy()
    # Remove batch dimension and rearrange axes to (H, W, C)
    numpy_img = np.transpose(numpy_img.squeeze(0), (1, 2, 0))
    numpy_img = numpy_img.clip(0, 1)
    # Rescale values to range [0, 255] and convert to uint8
    numpy_img = (numpy_img * 255).astype('uint8')
    # Create PIL image
    img = Image.fromarray(numpy_img)
    return img




def start_transform(img , style , generated , model , epoch):
    device = 'cuda' if torch.cuda.is_available else 'cpu'
    total_steps = epoch
    learning_rate = 0.01
    alpha = 1
    beta = 0.01
    optimizer = optim.Adam([generated], lr=learning_rate)

    for step in trange(total_steps):
        
        generated_features = model(generated)
        original_img_features = model(img)
        style_features = model(style)

        
        style_loss = original_loss = 0


        for gen_feature, orig_feature, style_feature in zip(
            generated_features, original_img_features, style_features
        ):

        
            batch_size, channel, height, width = gen_feature.shape
            original_loss += torch.mean((gen_feature - orig_feature) ** 2)
            
            G = gen_feature.view(channel, height * width).mm(
                gen_feature.view(channel, height * width).t()
            )
            
            A = style_feature.view(channel, height * width).mm(
                style_feature.view(channel, height * width).t()
            )
            style_loss += torch.mean((G - A) ** 2)

        total_loss = alpha * original_loss + beta * style_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        label_placeholder.write(f"loading..... {(step*100)/(total_steps-1)} %")
        out = tensor_to_image(generated)
        Image_placeholder.image(out, use_column_width=True)





st.title("Neural Style Transfer")

st.write("Upload images and Style for style transfer:")

img = st.file_uploader("Choose the original image...", type=["jpg", "jpeg", "png"])
style = st.file_uploader("Choose the style image...", type=["jpg", "jpeg", "png"])
epoch = abs(st.number_input("Enter number of epochs:", value=10, step=1))

st.write("Output Image")
label_placeholder = st.empty()
Image_placeholder = st.empty()




if img is not None and style is not None:
    img = load_image(img)
    style = load_image(style)
    

    generated = img.clone().requires_grad_(True)
    model = Model().eval()

    if st.button("Start"):
        start_transform(img , style , generated , model , epoch)

    


    

    



