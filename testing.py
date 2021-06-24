import torch
import torchvision.transforms as transforms
from PIL import Image


def print_test_examples(model, device, dataset):
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    model.eval()
    tst_1 = transform(Image.open(
        "TestImages/dog.jpg").convert("RGB")).unsqueeze(0)
    print("Caption-1:"+" ".join(model.captionize(tst_1.to(device), dataset.vocab)))
    tst_2 = transform(Image.open(
        "TestImages/surf.jpg").convert("RGB")).unsqueeze(0)
    print("Caption-1:"+" ".join(model.captionize(tst_2.to(device), dataset.vocab)))

    model.train()


def save_checkpoint(state, filename="Image_Captionizer.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    step = checkpoint["step"]
    return step
