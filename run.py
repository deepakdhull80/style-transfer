import os
import argparse
from datetime import datetime
import uuid

from tqdm import tqdm
import torch
from torchvision import transforms
from PIL import Image

from model.neural_style import StyleTransfer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-ci","--content", help="readable content image in local path", required=True)
    parser.add_argument("-si","--style", help="readable style image in local path",required=True)
    parser.add_argument("-d","--device", help="default device is cpu", default="cpu")
    parser.add_argument("-it","--iteration", help="default iteration is 200", default=200)
    parser.add_argument("--image_size", help="image_size of resultant image",default=512)
    parser.add_help("[hint] if image_size:[default=512] not able to load in cuda, reduce image_size or device=cpu")
    args = parser.parse_args()
    
    SAVE_PATH = "results"
    image_size = (int(args.image_size), int(args.image_size))

    transform = transforms.Compose([
        transforms.PILToTensor(),
        transforms.Resize(image_size),
        transforms.ConvertImageDtype(torch.float)
    ])

    
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
    try:
        device = torch.device(args.device)
    except:
        print(f"[==Warning] {args.device} is not found, using cpu")
        device = torch.device("cpu")

    model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg19', pretrained=True)
    model = model.eval().to(device)

    style_image = Image.open(args.style)
    content_image = Image.open(args.content)

    style_image_tensor = transform(style_image).unsqueeze(0).to(device)
    content_image_tensor = transform(content_image).unsqueeze(0).to(device)

    assert style_image_tensor.shape[1] == 3, f"having problem with style image :{args.style}"
    assert content_image_tensor.shape[1] == 3, f"having problem with content image :{args.content}"

    st_model = StyleTransfer.get_model(model.features, style_image_tensor, content_image_tensor)
    target_image = content_image_tensor.clone()
    target_image.requires_grad = True

    optimizer = torch.optim.LBFGS([target_image])

    SAVE_PATH = f"{SAVE_PATH}/{uuid.uuid4().urn.split(':')[-1]}"

    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
    
    i = [0]
    def r():
        with torch.no_grad():
            target_image.clamp_(0, 1)
        optimizer.zero_grad()
        loss = st_model.loss(target_image)
        loss.backward()
        if i[0] % 20 == 0:
            print(f"Epoch: {i[0]}, Total Loss",loss.detach().cpu().numpy())
        if (i[0]+1) % 20 == 0:
            transforms.ToPILImage()(target_image[0]).save(f"{SAVE_PATH}/st_image_[{i[0]+1}].jpg")
        i[0]+=1
        return loss
    
    for _ in tqdm(range(int(args.iteration)),total=int(args.iteration)):
        optimizer.step(r)
    
    transforms.ToPILImage()(target_image[0]).save(f"{SAVE_PATH}/st_image_[{i[0]+1}].jpg")
    print(f"images saved in :{SAVE_PATH}")
