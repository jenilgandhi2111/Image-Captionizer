# Image Captionizer
### Image captionizer is a beautiful application of deep neural networks. This helps us generate the description of the image given an input image.<br>

## How it works?
#### > The Image is first fed to any of the pretrained model like VGG16,VGG19,InceptionV3 for this implementation I have used InceptionV3. Now we remove the final output layer from the network and take the second last features network and then we add another dense layer after the feature layer which maps the features to embed size. This completes our definition of the Encoder.<br>> Now the decoder part is a LSTM network in which we provide the index of the tokens,which are then mapped to the embed size. Now before feeding this tensor of embeddings to the network we concat the output features (with the embedding tensor) from the encoder to produce the start token.
<br/>
<b>
    Architecture of the system is as below<br/>
</br>
<kbd>
<img src = "https://github.com/jenilgandhi2111/Image-Captionizer/blob/master/Assets/Diagram.png">
</kbd>

## Further improvements:
<ul>
<li>We could add visual attention to make it perform better</li>
<li>We could use transformer networks for better caption generation</li>
<li>We could also add teacher forcing and fine tune the teacher_force_ratio to get good results</li>
</ul>