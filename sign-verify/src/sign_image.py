import torch as T
import torch.utils.data as data
from torchvision import transforms
import models

from config import IMG_SIZE, device

transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor()
])

model = models.VggConvNet().to(device)
model.load_state_dict(T.load('model-vgg.pt'))

def predict(sign, vsign):
    tsign = transform(sign)
    tvsign  = transform(vsign)
    input = T.cat([tsign, tvsign]).float().to(device)
    output = model(input.unsqueeze(0))
    return output.squeeze().max(dim=-1)[1].cpu().item()


