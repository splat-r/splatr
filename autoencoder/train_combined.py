import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from autoencoder.dataset import Autoencoder_combined_dataset
from autoencoder.model import Autoencoder
from torch.utils.tensorboard import SummaryWriter
import argparse

torch.autograd.set_detect_anomaly(True)


def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()


def cos_loss(network_output, gt):
    return 1 - F.cosine_similarity(network_output, gt, dim=0).mean()


def train_autoencoder(data_dir_walkthrough_,
                      data_dir_unshuffle_,
                      dataset_name,
                      encoder_dims,
                      decoder_dims,
                      num_epochs=100,
                      lr=0.0001):

    data_dir_walkthrough = f"{data_dir_walkthrough_}/language_features"
    data_dir_unshuffle = f"{data_dir_unshuffle_}/language_features"

    os.makedirs(f'ckpt/{dataset_name}', exist_ok=True)
    train_dataset = Autoencoder_combined_dataset(data_dir_walkthrough, data_dir_unshuffle)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=16,
        drop_last=False
    )

    test_loader = DataLoader(
        dataset=train_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=16,
        drop_last=False
    )

    encoder_hidden_dims = encoder_dims
    decoder_hidden_dims = decoder_dims

    model = Autoencoder(encoder_hidden_dims, decoder_hidden_dims).to("cuda:0")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    logdir = f'ckpt/{dataset_name}'
    tb_writer = SummaryWriter(logdir)

    best_eval_loss = 100.0
    best_epoch = 0
    for epoch in tqdm(range(num_epochs)):
        model.train()
        for idx, feature in enumerate(train_loader):
            data = feature.to("cuda:0")
            outputs_dim3 = model.encode(data)
            outputs = model.decode(outputs_dim3)

            l2loss = l2_loss(outputs, data)
            cosloss = cos_loss(outputs, data)
            loss = l2loss + cosloss * 0.001

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_iter = epoch * len(train_loader) + idx
            tb_writer.add_scalar('train_loss/l2_loss', l2loss.item(), global_iter)
            tb_writer.add_scalar('train_loss/cos_loss', cosloss.item(), global_iter)
            tb_writer.add_scalar('train_loss/total_loss', loss.item(), global_iter)
            tb_writer.add_histogram("feat", outputs, global_iter)

        if epoch > 95:
            eval_loss = 0.0
            model.eval()
            for idx, feature in enumerate(test_loader):
                data = feature.to("cuda:0")
                with torch.no_grad():
                    outputs = model(data)
                loss = l2_loss(outputs, data) + cos_loss(outputs, data)
                eval_loss += loss * len(feature)
            eval_loss = eval_loss / len(train_dataset)
            print("eval_loss:{:.8f}".format(eval_loss))
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                best_epoch = epoch
                torch.save(model.state_dict(), f'ckpt/{dataset_name}/best_ckpt.pth')

            if epoch % 10 == 0:
                torch.save(model.state_dict(), f'ckpt/{dataset_name}/{epoch}_ckpt.pth')

    print(f"best_epoch: {best_epoch}")
    print("best_loss: {:.8f}".format(best_eval_loss))