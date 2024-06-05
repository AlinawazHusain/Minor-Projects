import torchvision.transforms as transforms


TRANSFORM = transforms.Compose(
    [   
        transforms.Resize((460 , 460)),
        transforms.PILToTensor(),
    ]
)

