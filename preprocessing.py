import os
import torch
import spacy
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, dataset
from PIL import Image
import torchvision.transforms as transforms

spacy_en = spacy.load("en_core_web_sm")


class TextPreprocessor:
    def __init__(self, threshhold=5):
        print("> Text Preprocessor initialized")
        self.threshhold = threshhold
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}

    def __len__(self):
        return len(self.stoi)

    @staticmethod
    def tokenizer_eng(text):
        return [token.text.lower() for token in spacy_en(text)]

    def build_vocab(self, captions):
        print("> Building Vocab")
        freq = {}
        idx = 4

        for sent in captions:
            for word in self.tokenizer_eng(sent):
                if word not in freq:
                    freq[word] = 1
                else:
                    freq[word] += 1

                # Now check if this word has the threshhold
                # frequency if yes then add it to stoi and itos dict
                if freq[word] == self.threshhold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

        print("> Finished Building Vocab")

    def numericalize(self, text):

        tokenized_txt = self.tokenizer_eng(text)
        ret_list = []
        ret_list.append(self.stoi["<SOS>"])
        for token in tokenized_txt:
            if token in self.stoi:
                ret_list.append(self.stoi[token])
            else:
                ret_list.append(self.stoi["<UNK>"])
        ret_list.append(self.stoi["<EOS>"])
        return ret_list


class FlickrDataset(Dataset):
    def __init__(self,
                 image_dir,
                 caption_file,
                 transforms=None,
                 threshhold=5):
        print("> Dataset initialized")
        self.root_dir = image_dir
        self.caption_file = caption_file
        self.df = pd.read_csv(caption_file)
        self.transform = transforms
        # The df would have a image and a caption mapped to it
        self.images = self.df["image"]
        self.captions = self.df["caption"]

        self.vocab = TextPreprocessor(threshhold=threshhold)
        self.vocab.build_vocab(self.captions.tolist())

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, index):
        print("> Index:", str(index))
        caption = self.captions[index]

        # This a image path we need to open that image by PIL
        image_idx = self.images[index]

        image = Image.open(os.path.join(
            self.root_dir, image_idx)).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        capt = []
        capt += self.vocab.numericalize(caption)

        return image, torch.tensor(capt)


# tx = TextPreprocessor(1)
# tx.build_vocab(["jenil is going home", "parishi is goint to school"])
# print(tx.numericalize("jenil is fine going"))

class Connector:
    def __init__(self, padding_id):
        self.pad_id = padding_id

    def __call__(self, batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)
        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first=False,
                               padding_value=self.pad_id)
        return imgs, targets


def get_loader(
    root_folder,
    annotation_file,
    transform=True,
    batch_size=32,
    num_workers=8,
    shuffle=True,
    pin_memory=True,
):
    print("> Loader Called")
    transform = transforms.Compose([
        transforms.Resize((356, 356)),
        transforms.RandomCrop((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = FlickrDataset(root_folder, annotation_file, transforms=transform)

    pad_idx = dataset.vocab.stoi["<PAD>"]

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        collate_fn=Connector(padding_id=pad_idx),
    )

    return loader, dataset
