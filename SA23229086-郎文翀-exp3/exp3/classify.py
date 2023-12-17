import torch
from torch import nn
from utils import preprocess, rev_label_map
import json
import os
from nltk.tokenize import PunktSentenceTokenizer, TreebankWordTokenizer
from PIL import Image, ImageDraw, ImageFont

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
checkpoint = 'checkpoint_han.pth.tar'
checkpoint = torch.load(checkpoint)
model = checkpoint['model']
model = model.to(device)
model.eval()

# Pad limits, can use any high-enough value since our model does not compute over the pads
sentence_limit = 15
word_limit = 20

# Word map to encode with
data_folder = r'/data2/lwc/PythonProjects/exp3/dataset'
with open(os.path.join(data_folder, 'word_map.json'), 'r') as j:
    word_map = json.load(j)

# Tokenizers
sent_tokenizer = PunktSentenceTokenizer()
word_tokenizer = TreebankWordTokenizer()


def classify(document):
    """
    Classify a document with the Hierarchial Attention Network (HAN).

    :param document: a document in text form
    :return: pre-processed tokenized document, class scores, attention weights for words, attention weights for sentences, sentence lengths
    """
    # A list to store the document tokenized into words
    doc = list()

    # Tokenize document into sentences
    sentences = list()
    for paragraph in preprocess(document).splitlines():
        sentences.extend([s for s in sent_tokenizer.tokenize(paragraph)])

    # Tokenize sentences into words
    for s in sentences[:sentence_limit]:
        w = word_tokenizer.tokenize(s)[:word_limit]
        if len(w) == 0:
            continue
        doc.append(w)

    # Number of sentences in the document
    sentences_in_doc = len(doc)
    sentences_in_doc = torch.LongTensor([sentences_in_doc]).to(device)  # (1)

    # Number of words in each sentence
    words_in_each_sentence = list(map(lambda s: len(s), doc))
    words_in_each_sentence = torch.LongTensor(words_in_each_sentence).unsqueeze(0).to(device)  # (1, n_sentences)

    # Encode document with indices from the word map
    encoded_doc = list(
        map(lambda s: list(map(lambda w: word_map.get(w, word_map['<unk>']), s)) + [0] * (word_limit - len(s)),
            doc)) + [[0] * word_limit] * (sentence_limit - len(doc))
    encoded_doc = torch.LongTensor(encoded_doc).unsqueeze(0).to(device)

    # Apply the HAN model
    scores, word_alphas, sentence_alphas = model(encoded_doc, sentences_in_doc,
                                                 words_in_each_sentence)  # (1, n_classes), (1, n_sentences, max_sent_len_in_document), (1, n_sentences)
    scores = scores.squeeze(0)  # (n_classes)
    scores = nn.functional.softmax(scores, dim=0)  # (n_classes)
    word_alphas = word_alphas.squeeze(0)  # (n_sentences, max_sent_len_in_document)
    sentence_alphas = sentence_alphas.squeeze(0)  # (n_sentences)
    words_in_each_sentence = words_in_each_sentence.squeeze(0)  # (n_sentences)

    return doc, scores, word_alphas, sentence_alphas, words_in_each_sentence


def visualize_attention(doc, scores, word_alphas, sentence_alphas, words_in_each_sentence):
    """
    Visualize important sentences and words, as seen by the HAN model.

    :param doc: pre-processed tokenized document
    :param scores: class scores, a tensor of size (n_classes)
    :param word_alphas: attention weights of words, a tensor of size (n_sentences, max_sent_len_in_document)
    :param sentence_alphas: attention weights of sentences, a tensor of size (n_sentences)
    :param words_in_each_sentence: sentence lengths, a tensor of size (n_sentences)
    """
    # Find best prediction
    score, prediction = scores.max(dim=0)
    prediction = '{category} ({score:.2f}%)'.format(category=rev_label_map[prediction.item()], score=score.item() * 100)

    # For each word, find it's effective importance (sentence alpha * word alpha)
    alphas = (sentence_alphas.unsqueeze(1) * word_alphas * words_in_each_sentence.unsqueeze(
        1).float() / words_in_each_sentence.max().float())
    # alphas = word_alphas * words_in_each_sentence.unsqueeze(1).float() / words_in_each_sentence.max().float()
    alphas = alphas.to('cpu')

    # Determine size of the image, visualization properties for each word, and each sentence
    min_font_size = 15  # minimum size possible for a word, because size is scaled by normalized word*sentence alphas
    max_font_size = 55  # maximum size possible for a word, because size is scaled by normalized word*sentence alphas
    font = ImageFont.truetype("./calibril.ttf", max_font_size)
    space_size = (font.getbbox(' ')[2], font.getbbox(' ')[3])

    line_spacing = 15  # spacing between sentences
    left_buffer = 100  # initial empty space on the left where sentence-rectangles will be drawn
    top_buffer = 2 * min_font_size + 3 * line_spacing  # initial empty space on the top where the detected category will be displayed
    image_width = left_buffer  # width of the entire image so far
    image_height = top_buffer + line_spacing  # height of the entire image so far
    word_loc = [image_width, image_height]  # top-left coordinates of the next word that will be printed
    rectangle_height = 0.75 * max_font_size  # height of the rectangles that will represent sentence alphas
    max_rectangle_width = 0.8 * left_buffer  # maximum width of the rectangles that will represent sentence alphas, scaled by sentence alpha
    rectangle_loc = [0.9 * left_buffer,
                     image_height + rectangle_height]  # bottom-right coordinates of next rectangle that will be printed
    word_viz_properties = list()
    sentence_viz_properties = list()
    for s, sentence in enumerate(doc):
        # Find visualization properties for each sentence, represented by rectangles
        # Factor to scale by
        sentence_factor = sentence_alphas[s].item() / sentence_alphas.max().item()

        # Color of rectangle
        rectangle_saturation = str(int(sentence_factor * 100))
        rectangle_lightness = str(25 + 50 - int(sentence_factor * 50))
        rectangle_color = 'hsl(0,' + rectangle_saturation + '%,' + rectangle_lightness + '%)'

        # Bounds of rectangle
        rectangle_bounds = [rectangle_loc[0] - sentence_factor * max_rectangle_width,
                            rectangle_loc[1] - rectangle_height] + rectangle_loc

        # Save sentence's rectangle's properties
        sentence_viz_properties.append({'bounds': rectangle_bounds.copy(),
                                        'color': rectangle_color})

        for w, word in enumerate(sentence):
            # Find visualization properties for each word
            # Factor to scale by
            word_factor = alphas[s, w].item() / alphas.max().item()

            # Color of word
            word_saturation = str(int(word_factor * 100))
            word_lightness = str(25 + 50 - int(word_factor * 50))
            word_color = 'hsl(0,' + word_saturation + '%,' + word_lightness + '%)'

            # Size of word
            word_font_size = int(min_font_size + word_factor * (max_font_size - min_font_size))
            word_font = ImageFont.truetype("./calibril.ttf", word_font_size)

            # Save word's properties
            word_viz_properties.append({'loc': word_loc.copy(),
                                        'word': word,
                                        'font': word_font,
                                        'color': word_color})

            # Update word and sentence locations for next word, height, width values
            word_size = (word_font.getmask(word).getbbox()[2], word_font.getmask(word).getbbox()[3])

            word_loc[0] += word_size[0] + space_size[0]
            image_width = max(image_width, word_loc[0])
        word_loc[0] = left_buffer
        word_loc[1] += max_font_size + line_spacing
        image_height = max(image_height, word_loc[1])
        rectangle_loc[1] += max_font_size + line_spacing

    # Create blank image
    img = Image.new('RGB', (image_width, image_height), (255, 255, 255))

    # Draw
    draw = ImageDraw.Draw(img)
    # Words
    for viz in word_viz_properties:
        draw.text(xy=viz['loc'], text=viz['word'], fill=viz['color'], font=viz['font'])
    # Rectangles that represent sentences
    for viz in sentence_viz_properties:
        draw.rectangle(xy=viz['bounds'], fill=viz['color'])
    # Detected category/topic
    category_font = ImageFont.truetype("./calibril.ttf", min_font_size)
    draw.text(xy=[line_spacing, line_spacing], text='Detected Category:', fill='grey', font=category_font)
    category_size = (
        category_font.getmask('Detected Category:').getbbox()[2],
        category_font.getmask('Detected Category:').getbbox()[3])
    draw.text(xy=[line_spacing, line_spacing + category_size[1] + line_spacing],

              text=prediction.upper(), fill='black',
              font=category_font)
    del draw

    # Display
    img.save("vis.png")



if __name__ == '__main__':
    document = "*Artichoke Gratine: The corn chips were amazing, lightly salted and crisp. The dip was a bit too garlicky and runny for my liking. Ate a few bites of this but could not see myself eating the entire thing solo.*Spicy Buffalo ""Wings"": first things first... do not let looks dismay you... true it looks gross but they taste legit! The flavor of the buffalo sauce was perfect, although could have been spicier. And the cucumber ranch dipping sauce was perfectly creamy and lightly flavored as to not overpower the ""wings"". This dish is a must try!!*Vegan Chili Fries: the fries are thin cut and tasty. The chili sauce was good, at first, but I quickly got sick of the flavor. This could be because I was never a huge chili fan even back when I ate meat. Hmm, I think you are better off ordering the thyme fries.*Crab Puffs: Perfectly crisp with a delicious creamy filling. Another must try!"
    # document = "Quiessence is, simply put, beautiful.  Full windows and earthy wooden walls give a feeling of warmth inside this restaurant perched in the middle of a farm.  The restaurant seemed fairly full even on a Tuesday evening; we had secured reservations just a couple days before."
    # document = "I think the acquisition by Republic has helped them overall. After Republic took over the in-flight cookies started. It still tends to suck if you aren't Ascent club (like just about any budget-centric airline does), but once you get there it's a good value. When I've had to fly Southwest or USAir I've been disappointed in comparison."
    # document = "Solid food, great device, and casual environment.  They will not leave you hungry.  Overall, it's really good."
    # document = "After my last review, somewhat scathing regarding the burgers, I got an invite from the management to come down, meet the chef and give him another try.  I thought to myself, ""how nice"", they read my review and they are concerned about the quality of their product.  I think that is admirable.  I did not take them up on the offer, because I don't feel that would be appropriate - I'm interested in doing yelp reviews because I have benefitted from reading the reviews of others, and I want to share my thoughts and feelings too :-) - I'm not looking for freebies."
    visualize_attention(*classify(document))