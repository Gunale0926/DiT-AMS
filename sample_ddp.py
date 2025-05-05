import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torchmetrics.image.fid import FrechetInceptionDistance
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import math
from tqdm import tqdm

from models import DiT_models
from download import find_model
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


def preprocess_image(example, image_size):
    # single image transform for FID: [0,1] range
    transform = transforms.Compose([
        transforms.Lambda(lambda img: center_crop_arr(img, image_size)),
        transforms.ToTensor(),  # scales to [0,1]
    ])
    example["image"] = transform(example["image"].convert("RGB"))
    return example


def main(args):
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    assert torch.cuda.is_available(), "Requires at least one GPU"
    torch.set_grad_enabled(False)

    # DDP setup
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    torch.cuda.set_device(device)

    # load model
    latent_size = args.image_size // 8
    model = DiT_models[args.model](input_size=latent_size, num_classes=args.num_classes).to(device)
    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)
    model.eval()

    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    using_cfg = args.cfg_scale > 1.0

    # prepare sample folder
    model_name = args.model.replace("/", "-")
    ckpt_name = os.path.basename(args.ckpt).replace(".pt", "") if args.ckpt else "pretrained"
    sample_folder = f"{args.sample_dir}/{model_name}-{ckpt_name}-size-{args.image_size}-vae-{args.vae}" \
                    f"-cfg-{args.cfg_scale}-seed-{args.global_seed}"
    if rank == 0:
        os.makedirs(sample_folder, exist_ok=True)
    dist.barrier()

    # sampling loop
    total = 0
    per_gpu = args.per_proc_batch_size
    world = dist.get_world_size()
    global_batch = per_gpu * world
    total_samples = int(math.ceil(args.num_fid_samples / global_batch) * global_batch)
    samples_per_gpu = total_samples // world
    iterations = samples_per_gpu // per_gpu

    # Evaluate FID using torchmetrics
    fid_metric = FrechetInceptionDistance().to(device)
    
    for _ in range(iterations):
        z = torch.randn(per_gpu, model.in_channels, latent_size, latent_size, device=device)
        y = torch.randint(0, args.num_classes, (per_gpu,), device=device)
        if using_cfg:
            z = torch.cat([z, z], 0)
            y = torch.cat([y, torch.full_like(y, args.num_classes)], 0)
            sample_fn = model.forward_with_cfg
            model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)
        else:
            sample_fn = model.forward
            model_kwargs = dict(y=y)

        samples = diffusion.p_sample_loop(sample_fn, z.shape, z, clip_denoised=False,
                                          model_kwargs=model_kwargs, progress=False, device=device)
        if using_cfg:
            samples = samples.chunk(2, dim=0)[0]

        samples = vae.decode(samples / 0.18215).sample
        samples = torch.clamp(127.5 * samples + 128.0, 0, 255)
        samples = samples.to(torch.uint8)

        fid_metric.update(samples, real=False)
        
        

    dist.barrier()

    if rank == 0:
        # load validation set
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        def preprocess_image(example):
            example["image"] = [transform(img.convert("RGB")) for img in example["image"]]
            return example
        val_ds = load_dataset(args.data_path, split="validation[:" + str(args.num_fid_samples) + "]").with_transform(preprocess_image)
        val_loader = DataLoader(val_ds["image"], batch_size=args.per_proc_batch_size)
        # update metric in batches
        for batch in val_loader:
            real = batch["image"]               # float32 [0,1]
            real_uint8 = (real * 255.0).round().to(torch.uint8)  # now uint8 [0,255]
            fid_metric.update(real_uint8, real=True)
            
        fid_value = fid_metric.compute()
        print(f"FID-{args.num_fid_samples}: {fid_value:.4f}")

    dist.destroy_process_group()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str)
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--sample-dir", type=str, default="samples")
    parser.add_argument("--per-proc-batch-size", type=int, default=32)
    parser.add_argument("--num-fid-samples", type=int, default=50_000)
    parser.add_argument("--image-size", type=int, choices=[64, 128, 256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=1.5)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ckpt", type=str, default=None)
    args = parser.parse_args()
    main(args)
