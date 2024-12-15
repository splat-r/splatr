import torch
from preprocess import PrepareDataset
from autoencoder.train_combined import train_autoencoder
from autoencoder.test_combined import get_combined_embedded_data

def process_langsplat_data(dataset_path_walkthrough,
                           dataset_path_unshuffle,
                           sam_ckpt_path,
                           dataset_name,
                           encoder_dims,
                           decoder_dims
                           ):

    # Prepare dataset
    # pd_walkthrough = PrepareDataset(dataset_path_walkthrough, sam_ckpt_path)
    # del pd_walkthrough
    torch.cuda.empty_cache()
    pd_unshuffle = PrepareDataset(dataset_path_unshuffle, sam_ckpt_path)
    del pd_unshuffle
    torch.cuda.empty_cache()

    # Train a combined autoencoder
    train_autoencoder(dataset_path_walkthrough,
                      dataset_path_unshuffle,
                      dataset_name,
                      encoder_dims=encoder_dims,
                      decoder_dims=decoder_dims,
                      lr=0.0007)
    torch.cuda.empty_cache()

    # Extract reduced feature dim vectors
    get_combined_embedded_data(dataset_path_walkthrough,
                               dataset_path_unshuffle,
                               dataset_name,
                               encoder_dims,
                               decoder_dims)
