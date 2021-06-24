import torch
import torch.nn as nn
import torchvision.models as models


class Encoder(nn.Module):
    '''
    Init Params:    Embedding_dimension , train_cnn
    Inputs:         Image
    Outputs:        Feature Vectors of the image mapped to the Embedding dimension of 
                    the image.
    Description:    > The model is supplied with a image where it is passed through a 
                      a pretrained model Here I have taken InceptionV3 but it could be 
                      any like VGG16,VGG19 etc. 
                    > The inception model last layer is removed and the second last layer
                      in the inception model (2048 features) is then mapped to the embedding
                      size. i.e we remove the top layer and then we attach a embedding size 
                      number of layer
                    > These features are then passed through relu activation functions and then
                      passed thorugh a dropout layer.                     
    '''

    def __init__(self, embed_size, train_cnn=False):
        super(Encoder, self).__init__()
        self.train_cnn = train_cnn

        self.inception = models.inception_v3(pretrained=True, aux_logits=False)
        # Adds a Embed size layer at the top of final features layer
        self.inception.fc = nn.Linear(
            self.inception.fc.in_features, embed_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, image):
        # Already reized image
        features = self.inception(image)
        return self.dropout(self.relu(features))


class Decoder(nn.Module):
    '''
    Init params:    Embedding_size,Hidden_size,Vocab_size,Num_layers.
    Inputs:         Feature vectors,Caption(Numericalized)
    Output:         Predicted Caption in numericalized form
    Description:    > Decoder is a simple LSTM network which is fed with a feature
                      vector from the encoder (of embedding dimension) and a embeddding
                      vectors from the embedding layer. Now these two  are concatenated.
                      i.e a feature vector is appended at the first of the embedding vectors
                    > Now these Are then passed through a LSTM layer and the hidden size vectors
                      are formed. Now another linear layer maps it to the vocab size.
                    > And these outputs of the vocab size vectors are returned , for each word.

    '''

    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(Decoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, features, caption):
        embedding = self.embed(caption)
        embedding = torch.cat((features.unsqueeze(0), embedding), dim=0)
        ops, _ = self.lstm(embedding)
        output = self.linear(ops)

        return output


class ImageCaptionizer(nn.Module):
    '''
    Init Params:    Embed_size , Hidden_size , Vocab_size , Number_of_layer_of_lstm
    Inputs:         Images and true captions
    Outputs:        Predicted Caption from the decoder
    Description:    > The Images are fed into the decoder and then feature vector is obtained.
                    > Now this feature vector is concated along the dimension of the embedding vector
                      so that the feature vector is the first input to the decoder lstm.
                      i.e
                        example:
                        feature vector:[1,2,3]
                        Embedding vectors: [[2,3,4],[3,4,5],[4,5,6]]
                        After Concating: [[1,2,3],[2,3,4],[3,4,5],[4,5,6]]
                    > Now the predicted captions are returned from the ImageCaptionizer.
    '''

    def __init__(self, embed_size, hidden_size, vocab_size, num_layer):
        super(ImageCaptionizer, self).__init__()
        self.encoder = Encoder(embed_size)
        self.decoder = Decoder(embed_size=embed_size, hidden_size=hidden_size,
                               vocab_size=vocab_size, num_layers=num_layer)

    def forward(self, image, captions):
        features = self.encoder(image)
        output = self.decoder(features, captions)

        return output

    def captionize(self, image, vocab, max_len=50):
        retcap = []
        with torch.no_grad():
            x = self.encoder(image).unsqueeze(0)
            previous_states = None
            for i in range(max_len):
                hiden_state, previous_states = self.decoder.lstm(
                    x, previous_states)
                op = self.decoder.linear(hiden_state.squeeze(0))
                pred = op.argmax(1)
                retcap.append(pred.item())
                x = self.decoder.embed(pred).unsqueeze(0)

                if vocab.itos[pred.item()] == "<EOS>":
                    break
        return [vocab.itos[idx] for idx in retcap]
