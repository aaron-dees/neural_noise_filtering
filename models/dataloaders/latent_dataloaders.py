import torch

def make_latent_dataloaders(train_latents,test_latents,batch_size,num_workers=2):
    train_dataset = torch.utils.data.TensorDataset(train_latents)
    test_dataset = torch.utils.data.TensorDataset(test_latents)
    # shapes are latents = [N,n_grains,z_dim] ; labels = [N]
    print("--- Latent train/test sizes: ",len(train_dataset),len(test_dataset))
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=num_workers)
    return train_dataloader,test_dataloader