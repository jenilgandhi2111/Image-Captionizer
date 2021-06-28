# Final Version of main.py
# To use this Image captionizer please download this dataset
# Link: https://www.kaggle.com/adityajn105/flickr8k
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
from preprocessing import get_loader
from testing import print_test_examples
from model import ImageCaptionizer
from tqdm import tqdm


def train():
    loss_points = []
    transform = transforms.Compose(
        [
            # Resizing to a larger dimension just to enhance features
            transforms.Resize((356, 356)),
            # Inception V3 Input size is (299,299)
            transforms.RandomCrop((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    # Getting the dataset from getloader
    loader, dataset = get_loader(
        root_folder="Image-Captionizer\Flickr8K\Images",
        annotation_file="Image-Captionizer\Flickr8K\captions.txt",
        transform=transform, num_workers=2
    )

    # HyperParams
    device = torch.device("cpu")
    embed_size = 256
    hidden_size = 256
    vocab_size = len(dataset.vocab)
    num_layers = 1
    lr = 3e-4
    num_epochs = 1

    model = ImageCaptionizer(embed_size, hidden_size, vocab_size, num_layers)
    lossfn = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<PAD>"])
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for name, param in model.encoder.inception.named_parameters():
        if "fc.weight" in name or "fc.bias" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    model.train()

    for epoch in range(num_epochs):
        print("> Epoch:", str(epoch+1))

        for idx, (img, capt) in tqdm(enumerate(loader), total=len(loader), leave=False):
            plt.plot(loss_points)
            print_test_examples(model, device, dataset)
            img = img.to(device)
            capt = capt.to(device)
            outputs = model(img, capt[:-1])
            loss = lossfn(
                outputs.reshape(-1, outputs.shape[2]), capt.reshape(-1)
            )
            loss_points.append(loss)
            optimizer.zero_grad()
            loss.backward(loss)
            optimizer.step()


train()
