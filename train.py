"""Train the pilot DCGAN on precomputed CT patches. Resume trusted checkpoints only."""
import argparse
import json
import random
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader
from models_dcgan import Generator, Discriminator, initialize_weights

parser = argparse.ArgumentParser()
parser.add_argument('--patch-dir', type=Path, required=True)
parser.add_argument('--split', type=Path, default=Path('configs/patient_split_seed42.json'))
parser.add_argument('--out', type=Path, required=True)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--resume', action='store_true')
parser.add_argument('--device', default='cuda')
args = parser.parse_args()
patient_split = json.loads(args.split.read_text())['splits']
sets = [set(patient_split[k]) for k in ('train','validation','test')]
assert all(sets[i].isdisjoint(sets[j]) for i in range(3) for j in range(i+1,3))
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
device = torch.device(args.device)
loader_rng = torch.Generator().manual_seed(seed)
arrays = []
for patient_id in patient_split['train']:
    with np.load(args.patch_dir / f'{patient_id}.npz', allow_pickle=False) as saved:
        assert str(saved['patient_id'].item()) == patient_id
        x = saved['patches']
        assert x.shape[1:] == (1,64,64) and np.isfinite(x).all()
        assert x.min() >= -1 and x.max() <= 1
        arrays.append(x)
train_images = torch.from_numpy(np.concatenate(arrays).astype(np.float32))
del arrays
train_loader = DataLoader(TensorDataset(train_images), batch_size=32, shuffle=True,
                         num_workers=0, pin_memory=device.type=='cuda', generator=loader_rng)
nz = 100
netG = Generator().to(device)
netD = Discriminator().to(device)
netG.apply(initialize_weights)
netD.apply(initialize_weights)
criterion = nn.BCEWithLogitsLoss()
optimizerG = torch.optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5,0.999))
optimizerD = torch.optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5,0.999))
num_epochs = args.epochs
run_dir = args.out
if not args.resume and run_dir.exists() and any(run_dir.iterdir()):
    raise RuntimeError('Output directory is not empty. Use a new directory or --resume.')
run_dir.mkdir(parents=True, exist_ok=True)
fixed_noise = torch.randn(64,nz,1,1,generator=torch.Generator().manual_seed(42)).to(device)
config = {'seed':seed,'epochs':num_epochs,'batch_size':32,'latent_dimension':nz,
          'learning_rate_G':0.0002,'learning_rate_D':0.0002,'adam_betas':[0.5,0.999],
          'loss':'BCEWithLogitsLoss','sampling':'uniform over patches, shuffled',
          'training_patches':len(train_images),'patient_split':patient_split,
          'patch_directory':str(args.patch_dir),'torch_version':str(torch.__version__),
          'cuda_version':torch.version.cuda,'device':str(device)}
history = []
start_epoch = 0
if args.resume:
    # This file contains Python/NumPy RNG objects. Load only your own trusted checkpoint.
    state = torch.load(run_dir / 'checkpoint_latest.pt', map_location='cpu', weights_only=False)
    if state['config']['patient_split'] != patient_split:
        raise RuntimeError('Checkpoint patient split differs.')
    netG.load_state_dict(state['generator'])
    netD.load_state_dict(state['discriminator'])
    optimizerG.load_state_dict(state['optimizerG'])
    optimizerD.load_state_dict(state['optimizerD'])
    fixed_noise = state['fixed_noise'].to(device)
    history = state['history']
    start_epoch = state['epoch']
    config = state['config']
    torch.set_rng_state(state['torch_rng'])
    if device.type == 'cuda': torch.cuda.set_rng_state_all(state['cuda_rng'])
    loader_rng.set_state(state['loader_rng'])
    np.random.set_state(state['numpy_rng'])
    random.setstate(state['python_rng'])
(run_dir / 'config.json').write_text(json.dumps(config, indent=2))
torch.save(fixed_noise.cpu(), run_dir / 'fixed_noise.pt')

import time
import matplotlib.pyplot as plt

# Preserve this cell's code for reproducibility.


for epoch in range(start_epoch, num_epochs):
    started = time.time()
    netG.train()
    netD.train()
    total_G = total_D = 0.0
    seen = 0

    for (images,) in train_loader:
        real = images.to(device, non_blocking=True)
        batch = len(real)

        # Train discriminator.
        optimizerD.zero_grad(set_to_none=True)
        noise = torch.randn(batch, nz, 1, 1, device=device)
        fake = netG(noise)

        lossD = (
            criterion(netD(real), torch.ones(batch, device=device))
            + criterion(netD(fake.detach()), torch.zeros(batch, device=device))
        )
        if not torch.isfinite(lossD):
            raise RuntimeError("Non-finite discriminator loss")
        lossD.backward()
        optimizerD.step()

        # Train generator, keeping discriminator parameters fixed.
        for parameter in netD.parameters():
            parameter.requires_grad_(False)

        optimizerG.zero_grad(set_to_none=True)
        lossG = criterion(netD(fake), torch.ones(batch, device=device))
        if not torch.isfinite(lossG):
            raise RuntimeError("Non-finite generator loss")
        lossG.backward()
        optimizerG.step()

        for parameter in netD.parameters():
            parameter.requires_grad_(True)

        total_G += lossG.item() * batch
        total_D += lossD.item() * batch
        seen += batch

    record = {
        "epoch": epoch + 1,
        "generator_loss": total_G / seen,
        "discriminator_loss": total_D / seen,
        "seconds": time.time() - started,
    }
    history.append(record)

    # Use fixed noise and a fixed display range.
    netG.eval()
    with torch.no_grad():
        samples = netG(fixed_noise).cpu()

    fig, axes = plt.subplots(8, 8, figsize=(8, 8))
    for ax, sample in zip(axes.flat, samples):
        ax.imshow(sample[0], cmap="gray", vmin=-1, vmax=1)
        ax.axis("off")
    fig.tight_layout(pad=0.1)
    fig.savefig(run_dir / f"samples_epoch_{epoch + 1:03d}.png")
    plt.close(fig)
    netG.train()

    checkpoint = {
        "epoch": epoch + 1,
        "generator": netG.state_dict(),
        "discriminator": netD.state_dict(),
        "optimizerG": optimizerG.state_dict(),
        "optimizerD": optimizerD.state_dict(),
        "fixed_noise": fixed_noise.cpu(),
        "history": history,
        "config": config,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
        "loader_rng": loader_rng.get_state(),
        "numpy_rng": np.random.get_state(),
        "python_rng": random.getstate(),
    }

    temporary = run_dir / "checkpoint.partial.pt"
    torch.save(checkpoint, temporary)
    temporary.replace(run_dir / "checkpoint_latest.pt")
    (run_dir / "history.json").write_text(json.dumps(history, indent=2))
    start_epoch = epoch + 1

    print(
        f"Epoch {start_epoch}/{num_epochs} | "
        f"G: {record['generator_loss']:.4f} | "
        f"D: {record['discriminator_loss']:.4f} | "
        f"Time: {record['seconds']:.0f}s"
    )
    print("Checkpoint saved to Google Drive.")