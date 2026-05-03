import argparse
import logging
import math
import os
import shutil
from tqdm import tqdm
from pathlib import Path

import accelerate
import datasets
import numpy as np
import torch
import torch.nn.functional as F
import diffusers, transformers
from torchvision.transforms import transforms
from torchvision.transforms.functional import to_pil_image
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed, is_swanlab_available
from datasets import load_dataset
from diffusers import (
    DDPMScheduler,
    StableDiffusionPipeline,
    StableDiffusionAdapterPipeline,
    UNet2DConditionModel,
    AutoencoderKL,
    T2IAdapter,
)
from diffusers.optimization import get_scheduler
from diffusers.utils.torch_utils import is_compiled_module
from diffusers.utils import is_wandb_available, check_min_version
from diffusers.training_utils import free_memory
from omegaconf import OmegaConf, DictConfig
from packaging import version
from torch.utils.data import DataLoader

if is_wandb_available():
    import wandb

if is_swanlab_available():
    import swanlab

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.36.0")
logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train conditional TinyDiT DDPM with accelerate + diffusers."
    )
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)

    return cfg


def make_dataloaders(args):
    data_dir = Path(args.data_dir)
    assert (
        data_dir / "train.parquet"
    ).exists(), f"Missing {(data_dir / 'train.parquet')}"
    assert (
        data_dir / "test.parquet"
    ).exists(), f"Missing {(data_dir / 'test.parquet')}"

    dataset = load_dataset(
        "parquet",
        data_files={
            "train": str(data_dir / "train.parquet"),
            "test": str(data_dir / "test.parquet"),
        },
        cache_dir=args.cache_dir,
    )

    num_train = len(dataset["train"])
    val_size = max(1, int(num_train * args.val_ratio))
    perm = np.random.RandomState(args.seed).permutation(num_train)
    val_idx = perm[:val_size].tolist()
    train_idx = perm[val_size:].tolist()

    ds_train = dataset["train"].select(train_idx)
    ds_val = dataset["train"].select(val_idx)
    if args.max_train_samples is not None:
        ds_train = ds_train.select(range(min(len(ds_train), args.max_train_samples)))

    image_transforms = transforms.Compose(
        [
            transforms.Resize(
                args.resolution, interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.CenterCrop(args.resolution),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )

    conditioning_transforms = transforms.Compose(
        [
            transforms.Resize(
                args.resolution, interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.CenterCrop(args.resolution),
            transforms.ToTensor(),
        ]
    )

    def collate_fn(batch):
        ids = torch.tensor([b["id"] for b in batch], dtype=torch.long)
        src = torch.stack(
            [
                conditioning_transforms(b.pop("gray_image").convert("RGB"))
                for b in batch
            ],
            dim=0,
        )

        data = {"id": ids, "src": src}

        if "target_image" in batch[0] and batch[0]["target_image"] is not None:
            dst = torch.stack(
                [image_transforms(b.pop("target_image").convert("RGB")) for b in batch],
                dim=0,
            )
            data.update({"dst": dst})

        return data

    train_loader = DataLoader(
        ds_train,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.dataloader_num_workers,
    )
    val_loader = DataLoader(
        ds_val,
        batch_size=args.per_device_eval_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.dataloader_num_workers,
    )
    return ds_train, ds_val, train_loader, val_loader


def log_validation(
    eval_dataloader: torch.utils.data.DataLoader,
    t2i_adapter: T2IAdapter,
    unet: UNet2DConditionModel,
    vae: AutoencoderKL,
    args: DictConfig,
    accelerator: Accelerator,
    weight_dtype: torch.dtype,
    step: int,
):
    logger.info("Running validation...")

    pipe = StableDiffusionAdapterPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=vae,
        adapter=t2i_adapter,
        unet=unet,
        safety_checker=None,
        requires_safety_checker=False,
    )

    pipe.to(accelerator.device, weight_dtype)

    pipe.set_progress_bar_config(disable=True)

    generator = torch.Generator(device=accelerator.device)

    if args.seed is not None:
        generator = generator.manual_seed(args.seed)

    image_logs = []
    for batch in eval_dataloader:
        bsz = len(batch["id"])
        images = pipe(
            prompt=[""] * bsz,
            image=batch["src"],
            num_images_per_prompt=1,
            height=args.resolution,
            width=args.resolution,
            generator=generator,
        ).images

        for image, pixel_values, image_id in zip(images, batch["src"], batch["id"]):
            image_logs.append(
                {
                    "image": image,
                    "condition": to_pil_image(pixel_values),
                    "name": image_id,
                }
            )

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            for log in image_logs:
                imagegrid = [np.asarray(log["condition"]), np.asarray(log["image"])]
                imagegrid = np.stack(imagegrid)
                tracker.writer.add_images(
                    log["name"], imagegrid, step, dataformats="NHWC"
                )

        elif tracker.name == "wandb":
            for log in image_logs:
                log_imgs = [
                    wandb.Image(log["condition"], caption="condition"),
                    wandb.Image(log["image"], caption="image"),
                ]
                tracker.log({log["name"]: log_imgs})

        elif tracker.name == "swanlab":
            for log in image_logs:
                log_imgs = [
                    swanlab.Image(log["condition"], caption="condition"),
                    swanlab.Image(log["image"], caption="image"),
                ]
                tracker.log({log["name"]: log_imgs})
        else:
            logger.warning(f"image logging not implemented for {tracker.name}")

    del pipe
    free_memory()

    vae.to(torch.float32)

    return image_logs


def main():
    args = parse_args()
    project_config = ProjectConfiguration(project_dir=args.output_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with=args.report_to,
        project_config=project_config,
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae"
    )
    vae.requires_grad_(False)

    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet"
    )
    unet.requires_grad_(False)

    t2i_adapter = T2IAdapter(
        in_channels=3,
        channels=unet.config.block_out_channels,
        adapter_type="full_adapter",
    )

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae.to(accelerator.device, torch.float32)
    unet.to(accelerator.device, weight_dtype)
    t2i_adapter.to(accelerator.device, weight_dtype)

    optimizer = torch.optim.AdamW(
        t2i_adapter.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
    )

    train_dataset, _, train_dataloader, eval_dataloader = make_dataloaders(args)

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):

        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                for _model in models:
                    unwrap_model(_model).save_pretrained(
                        os.path.join(output_dir, "t2i_adapter")
                    )
                    weights.pop()

        def load_model_hook(models, input_dir):
            for _ in range(len(models)):
                model = models.pop()
                load_model = T2IAdapter.from_pretrained(
                    input_dir, subfolder="t2i_adapter"
                )
                model.register_to_config(**load_model.config)
                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    logger.info(
        f"All models and datamodule loaded successfully. Num trainable parameters = {t2i_adapter.num_parameters()}"
    )
    # Scheduler and math around the number of training steps.
    # Check the PR https://github.com/huggingface/diffusers/pull/8312 for detailed explanation.
    if args.get("max_steps", None) is None:
        len_train_dataloader_after_sharding = math.ceil(
            len(train_dataloader) / accelerator.num_processes
        )
        num_update_steps_per_epoch = math.ceil(
            len_train_dataloader_after_sharding
            / accelerator.gradient_accumulation_steps
        )
        num_training_steps_for_scheduler = (
            args.num_train_epochs
            * num_update_steps_per_epoch
            * accelerator.num_processes
        )
    else:
        num_training_steps_for_scheduler = args.max_steps * accelerator.num_processes

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=num_training_steps_for_scheduler,
    )

    # Prepare everything with our `accelerator`.
    t2i_adapter, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        t2i_adapter, optimizer, train_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / accelerator.gradient_accumulation_steps
    )
    if args.get("max_steps", None) is None:
        args.max_steps = args.num_train_epochs * num_update_steps_per_epoch
        if (
            num_training_steps_for_scheduler
            != args.max_steps * accelerator.num_processes
        ):
            logger.warning(
                f"The length of the 'train_dataloader' after 'accelerator.prepare' ({len(train_dataloader)}) does not match "
                f"the expected length ({len_train_dataloader_after_sharding}) when the learning rate scheduler was created. "
                f"This inconsistency may result in the learning rate scheduler not functioning properly."
            )

    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_args = {}
        for k, v in OmegaConf.to_container(args).items():
            if isinstance(v, (int, float, str, bool, torch.Tensor)):
                tracker_args[k] = v
            elif isinstance(v, dict):
                for kk, vv in v.items():
                    tracker_args[f"{k}.{kk}"] = str(vv)
            else:
                tracker_args[k] = str(v)

        run = os.path.split(__file__)[-1].split(".")[0]
        accelerator.init_trackers(run, config=tracker_args)

    # Train!
    total_batch_size = (
        args.per_device_train_batch_size
        * accelerator.num_processes
        * accelerator.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(
        f"  Instantaneous batch size per device = {args.per_device_train_batch_size}"
    )
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(
        f"  Gradient Accumulation steps = {accelerator.gradient_accumulation_steps}"
    )
    logger.info(f"  Total optimization steps = {args.max_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint is not None:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    # get the fixed prompt embeds
    text_encoding_pipeline = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        unet=None,
        vae=None,
    )
    text_encoding_pipeline.to(accelerator.device, weight_dtype)
    with torch.no_grad():
        prompt_embeds, _ = text_encoding_pipeline.encode_prompt(
            "",
            device=accelerator.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )

    text_encoding_pipeline.to("cpu")
    del text_encoding_pipeline
    free_memory()

    progress_bar = tqdm(
        range(0, args.max_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    for _ in range(first_epoch, args.num_train_epochs):
        train_loss = 0.0
        t2i_adapter.train()
        for batch in train_dataloader:
            with accelerator.accumulate(t2i_adapter):
                # Convert images to latent space
                with torch.no_grad():
                    latents = (
                        vae.encode(batch["dst"].to(accelerator.device, torch.float32))
                        .latent_dist.sample()
                        .to(weight_dtype)
                    )
                    latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]

                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=latents.device,
                )
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_model_input = noise_scheduler.add_noise(latents, noise, timesteps)

                # Predict the noise residual and compute loss
                adapter_input = batch["src"].to(accelerator.device, weight_dtype)
                down_intrablock_additional_residuals = t2i_adapter(adapter_input)

                with torch.autocast(accelerator.device.type, weight_dtype):
                    model_pred = unet(
                        noisy_model_input,
                        timesteps,
                        encoder_hidden_states=prompt_embeds.expand(bsz, -1, -1).to(
                            accelerator.device, weight_dtype
                        ),
                        down_intrablock_additional_residuals=down_intrablock_additional_residuals,
                        return_dict=False,
                    )[0]

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(
                        f"Unknown prediction type {noise_scheduler.config.prediction_type}"
                    )

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(
                    loss.repeat(args.per_device_train_batch_size)
                ).mean()

                train_loss += avg_loss.item() / accelerator.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        t2i_adapter.parameters(), args.max_grad_norm
                    )
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                logs = {"loss": train_loss}

                accelerator.log(logs, step=global_step)
                train_loss = 0.0

                if accelerator.is_main_process:
                    # Checkpointing
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `save_total_limit`
                        if args.save_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [
                                d for d in checkpoints if d.startswith("checkpoint")
                            ]
                            checkpoints = sorted(
                                checkpoints, key=lambda x: int(x.split("-")[1])
                            )

                            # before we save the new checkpoint, we need to have at _most_ `save_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.save_total_limit:
                                num_to_remove = (
                                    len(checkpoints) - args.save_total_limit + 1
                                )
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(
                                    f"removing checkpoints: {', '.join(removing_checkpoints)}"
                                )

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(
                                        args.output_dir, removing_checkpoint
                                    )
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(
                            args.output_dir, f"checkpoint-{global_step}"
                        )
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")
                    # Validation
                    if global_step % args.validation_steps == 0:
                        log_validation(
                            eval_dataloader,
                            unwrap_model(t2i_adapter),
                            unwrap_model(unet),
                            unwrap_model(vae),
                            args,
                            accelerator,
                            weight_dtype,
                            step=global_step,
                        )

            logs = {
                "step_loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
            }
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_steps:
                break

    # Create the pipeline using using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped_model = unwrap_model(t2i_adapter)

        if args.upcast_before_saving:
            unwrapped_model.to(torch.float32)

        unwrapped_model.save_pretrained(args.output_dir)

    accelerator.end_training()


if __name__ == "__main__":
    main()
