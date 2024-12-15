import cv2
import os
import natsort
import glob
import numpy as np
from tqdm import trange
import torch
import torchvision
import torchvision.transforms as transforms
transform = transforms.ToTensor()
name = "FloorPlan324__val__38"
path = f"rearrange/test/runs/{name}/unshuffle"
path_w = f"rearrange/test/runs/{name}/walkthrough"

reason = os.path.join(path, "reason", "*_save.png")
rend = os.path.join(path, "rendered", "*.png")
sim = os.path.join(path, "sim", "*.png")
dino = os.path.join(path_w, "dino_frames", "*.png")

reason_image_paths = natsort.natsorted(glob.glob(reason))
rend_image_paths = natsort.natsorted(glob.glob(rend))
sim_image_paths = natsort.natsorted(glob.glob(sim))
dino_image_paths = natsort.natsorted(glob.glob(dino))

frames_dino = []
for i in trange(len(dino_image_paths)):
    base_name = os.path.basename(rend_image_paths[i])
    base_name = os.path.splitext(base_name)[0]
    base_name = int(base_name)
    rend_image = cv2.imread(rend_image_paths[i])
    rend_image = cv2.cvtColor(rend_image, cv2.COLOR_BGR2RGB)
    sim_image = cv2.imread(sim_image_paths[i])
    sim_image = cv2.cvtColor(sim_image, cv2.COLOR_BGR2RGB)
    dino_image = cv2.imread(dino_image_paths[i])
    dino_image = cv2.cvtColor(dino_image, cv2.COLOR_BGR2RGB)
    image_save = np.zeros((dino_image.shape[0] + 100, dino_image.shape[1]*2 + 250, 3))

    image_save[50:50+sim_image.shape[0], 50:50+sim_image.shape[1], :] = sim_image
    image_save[150+sim_image.shape[0]:150+sim_image.shape[0]*2, 50:50+sim_image.shape[1], :] = rend_image

    image_save[50:50+dino_image.shape[0], 150+sim_image.shape[1]:150+sim_image.shape[1]+int(dino_image.shape[1]/2), :] = dino_image[:, int(dino_image.shape[1]/2):, :]
    image_save[50:50 + dino_image.shape[0],200 + sim_image.shape[1] + int(dino_image.shape[1] / 2):200 + sim_image.shape[1] + dino_image.shape[1], :] = dino_image[:, :int(dino_image.shape[1] / 2), :]

    reason_found = 0
    for j in range(len(reason_image_paths)):
        base_name_reason = os.path.basename(reason_image_paths[j])
        base_name_reason = os.path.splitext(base_name_reason)[0]
        base_name_reason = base_name_reason.split("_")[0]
        base_name_reason = int(base_name_reason)
        if base_name_reason == base_name:
            image_reason = cv2.imread(reason_image_paths[j])
            image_reason = cv2.cvtColor(image_reason, cv2.COLOR_BGR2RGB)

            image_save[50:50+sim_image.shape[0], 300+sim_image.shape[1]+dino_image.shape[1]: 300+2*sim_image.shape[1]+dino_image.shape[1], :] = image_reason[:, :sim_image.shape[1], :]
            image_save[150+sim_image.shape[0]:150+sim_image.shape[0]*2, 300 + sim_image.shape[1] + dino_image.shape[1]: 300 + 2 * sim_image.shape[1] + dino_image.shape[1], :] = image_reason[:, sim_image.shape[1]:, :]

            reason_found = 1
            break
    if not reason_found:
        image_save[50:50 + sim_image.shape[0], 300 + sim_image.shape[1] + dino_image.shape[1]: 300 + 2 * sim_image.shape[1] + dino_image.shape[1], :] = sim_image
        image_save[150 + sim_image.shape[0]:150 + sim_image.shape[0] * 2, 300 + sim_image.shape[1] + dino_image.shape[1]: 300 + 2 * sim_image.shape[1] + dino_image.shape[1], :] = rend_image

    img_tensor = transform(image_save)
    img_tensor = img_tensor.permute(1, 2, 0)
    frames_dino.append(img_tensor)
frames = torch.stack(frames_dino)
filename = os.path.join(f'viz_{name}.mp4')
fps = 3
torchvision.io.write_video(filename, frames, fps)

