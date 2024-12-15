import torch
import numpy as np
import sys
sys.path.append("dinov2")
from PIL import Image
import torchvision.transforms as transforms
from sklearn.decomposition import PCA

class Dinov2Matcher:
    def __init__(self, repo_name="facebookresearch/dinov2", model_name="dinov2_vitb14", smaller_edge_size=448, half_precision=False, device="cuda"):
        self.repo_name = repo_name
        self.model_name = model_name
        self.smaller_edge_size = smaller_edge_size
        self.half_precision = half_precision
        self.device = device

        if self.half_precision:
          self.model = torch.hub.load(repo_or_dir=repo_name, model=model_name).half().to(self.device)
        else:
          self.model = torch.hub.load(repo_or_dir=repo_name, model=model_name).to(self.device)

        self.model.eval()

        self.transform = transforms.Compose([
          transforms.Resize(size=smaller_edge_size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
          transforms.ToTensor(),
          transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  # imagenet defaults
        ])

    def prepare_image(self, rgb_image_numpy):
        image = Image.fromarray(rgb_image_numpy)
        image_tensor = self.transform(image)
        resize_scale = image.width / image_tensor.shape[2]

        # Crop image to dimensions that are a multiple of the patch size
        height, width = image_tensor.shape[1:]  # C x H x W
        cropped_width, cropped_height = width - width % self.model.patch_size, height - height % self.model.patch_size  # crop a bit from right and bottom parts
        image_tensor = image_tensor[:, :cropped_height, :cropped_width]

        grid_size = (cropped_height // self.model.patch_size, cropped_width // self.model.patch_size)
        return image_tensor, grid_size, resize_scale

    def prepare_mask(self, mask_image_numpy, grid_size, resize_scale):
        cropped_mask_image_numpy = mask_image_numpy[:int(grid_size[0] * self.model.patch_size * resize_scale),
                                   :int(grid_size[1] * self.model.patch_size * resize_scale)]
        image = Image.fromarray(cropped_mask_image_numpy)
        resized_mask = image.resize((grid_size[1], grid_size[0]), resample=Image.Resampling.NEAREST)
        resized_mask = np.asarray(resized_mask).flatten()
        return resized_mask

    def extract_features(self, image_tensor):
        with torch.inference_mode():
            if self.half_precision:
                image_batch = image_tensor.unsqueeze(0).half().to(self.device)
            else:
                image_batch = image_tensor.unsqueeze(0).to(self.device)

            tokens = self.model.get_intermediate_layers(image_batch)[0].squeeze()
        return tokens.cpu().numpy()

    def idx_to_source_position(self, idx, grid_size, resize_scale):
        row = (idx // grid_size[1]) * self.model.patch_size * resize_scale + self.model.patch_size / 2
        col = (idx % grid_size[1]) * self.model.patch_size * resize_scale + self.model.patch_size / 2
        return row, col

    def get_embedding_visualization(self, tokens, grid_size, resized_mask=None):
        pca = PCA(n_components=3)
        if resized_mask is not None:
            tokens = tokens[resized_mask]
        reduced_tokens = pca.fit_transform(tokens.astype(np.float32))
        if resized_mask is not None:
            tmp_tokens = np.zeros((*resized_mask.shape, 3), dtype=reduced_tokens.dtype)
            tmp_tokens[resized_mask] = reduced_tokens
            reduced_tokens = tmp_tokens
        reduced_tokens = reduced_tokens.reshape((*grid_size, -1))
        normalized_tokens = (reduced_tokens - np.min(reduced_tokens)) / (
                    np.max(reduced_tokens) - np.min(reduced_tokens))
        return normalized_tokens

    def get_combined_embedding_visualization(self, tokens1, token2, grid_size1, grid_size2, mask1=None, mask2=None,
                                             random_state=20):
        pca = PCA(n_components=3, random_state=random_state)

        token1_shape = tokens1.shape[0]
        if mask1 is not None:
            tokens1 = tokens1[mask1]
        if mask2 is not None:
            token2 = token2[mask2]
        combinedtokens = np.concatenate((tokens1, token2), axis=0)
        reduced_tokens = pca.fit_transform(combinedtokens.astype(np.float32))

        if mask1 is not None and mask2 is not None:
            resized_mask = np.concatenate((mask1, mask2), axis=0)
            tmp_tokens = np.zeros((*resized_mask.shape, 3), dtype=reduced_tokens.dtype)
            tmp_tokens[resized_mask] = reduced_tokens
            reduced_tokens = tmp_tokens
        elif mask1 is not None and mask2 is None:
            return sys.exit("Either use both masks or none")
        elif mask1 is None and mask2 is not None:
            return sys.exit("Either use both masks or none")

        normalized_tokens = (reduced_tokens - np.min(reduced_tokens)) / (
                    np.max(reduced_tokens) - np.min(reduced_tokens))

        rgbimg1 = normalized_tokens[0:token1_shape, :]
        rgbimg2 = normalized_tokens[token1_shape:, :]

        rgbimg1 = rgbimg1.reshape((*grid_size1, -1))
        rgbimg2 = rgbimg2.reshape((*grid_size2, -1))
        return rgbimg1, rgbimg2, tokens1, token2

"""dense feature matching was inspired from this demo -> 
        https://github.com/antmedellin/dinov2/tree/main"""