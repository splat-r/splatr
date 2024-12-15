import open_clip
from PIL import Image

class CLIPFeatureExtractor():
    def __init__(self, device):
        self.device = device
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
            "ViT-B-16", "laion2b_s34b_b88k"
        )
        self.clip_model = self.clip_model.to(device)
        self.clip_tokenizer = open_clip.get_tokenizer("ViT-B-16")

    def tokenize_text(self, text_prompt):
        """
        text_prompt -> list
        """
        tokenized_text = self.clip_tokenizer(text_prompt).to(self.device)
        text_feat = self.clip_model.encode_text(tokenized_text)
        text_feat /= text_feat.norm(dim=-1, keepdim=True)
        return text_feat.cpu()

    def tokenize_image(self, image):
        image = Image.fromarray(image)
        preprocessed_image = self.clip_preprocess(image).unsqueeze(0).to(self.device)
        img_feat = self.clip_model.encode_image(preprocessed_image).detach()
        img_feat /= img_feat.norm(dim=-1, keepdim=True)
        return img_feat.cpu()