from PIL import Image
import requests
from transformers import FlavaProcessor, FlavaModel
from transformers import FlavaFeatureExtractor
from transformers import BertTokenizer


class CustomFlavaModel:
    def __init__(self):
        self.model = FlavaModel.from_pretrained("facebook/flava-full")
        self.processor = FlavaProcessor.from_pretrained("facebook/flava-full")
        self.feature_extractor = FlavaFeatureExtractor.from_pretrained("facebook/flava-full")
        self.tokenizer = BertTokenizer.from_pretrained("facebook/flava-full")

    def train(self, data):
        # Training logic for Flava model
        pass

    def predict(self, input_data):
        # Prediction logic for Flava model
        pass

    def evaluate(self, url=None):
        if url is None:
            url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image = Image.open(requests.get(url, stream=True).raw)
        inputs = self.processor(
            text=["a photo of a cat", "a photo of a dog"], images=[image, image], return_tensors="pt",
            padding="max_length", max_length=77
        )
        outputs = self.model(**inputs)
        image_embeddings = outputs.image_embeddings  # Batch size X (Number of image patches + 1) x Hidden size => 2 X 197 X 768
        text_embeddings = outputs.text_embeddings  # Batch size X (Text sequence length + 1) X Hidden size => 2 X 77 X 768
        multimodal_embeddings = outputs.multimodal_embeddings  # Batch size X (Number of image patches + Text Sequence Length + 3) X Hidden size => 2 X 275 x 768
        # Multimodal embeddings can be used for multimodal tasks such as VQA

        ## Pass only image
        inputs = self.feature_extractor(images=[image, image], return_tensors="pt")
        outputs = self.model(**inputs)
        image_embeddings = outputs.image_embeddings

        ## Pass only text
        inputs = self.tokenizer(["a photo of a cat", "a photo of a dog"], return_tensors="pt", padding="max_length",
                           max_length=77)
        outputs = self.model(**inputs)
        text_embeddings = outputs.text_embeddings
