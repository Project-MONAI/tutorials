# Data Resampling, Model Training, and Model Inference Processes

This document describes the steps involved in data resampling, model training, and model inference, following the order provided in the Python scripts.

## Table of Contents
1. [Data Resampling](#data-resampling)
2. [Model Training](#model-training)
3. [Model Inference](#model-inference)

---

## Data Resampling

### Initialization and Setup

1. **Set Deterministic Seeds**:
    ```python
    torch.manual_seed(0)
    np.random.seed(0)
    set_determinism(seed=0)
    ```

2. **Define Paths**:
    ```python
    dataroot = "/mnt/drive2/data_128"
    filenames_filepath = "/mnt/drive2/data_128/filenames_image_nii_autopet.txt"
    output_root_embedding = "/mnt/drive2/encoding_128"
    autoencoder_root = "/workspace/monai/generative/from_canz"
    ```

3. **Distributed Initialization**:
    ```python
    dist.init_process_group(backend="nccl", init_method="env://")
    local_rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    ```

### Model Loading

1. **Load Autoencoder Model**:
    ```python
    autoencoder = AutoencoderKLCKModified_TP(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        num_channels=(64, 128, 256),
        latent_channels=4,
        attention_levels=(False, False, False),
        num_res_blocks=(2, 2, 2),
        norm_num_groups=32,
        norm_eps=1e-06,
        with_encoder_nonlocal_attn=False,
        with_decoder_nonlocal_attn=False,
        use_checkpointing=False,
        use_convtranspose=False,
    )
    autoencoder.to(device)
    ```

2. **Load Checkpoints**:
    ```python
    checkpoint_autoencoder = torch.load(
        os.path.join(autoencoder_root, "autoencoder_epoch273.pt"),
        map_location=torch.device(f"cuda:{local_rank}"),
    )
    autoencoder.load_state_dict(checkpoint_autoencoder)
    ```

### Data Transformation

1. **Define Transformations**:
    ```python
    transforms = Compose([
        monai.transforms.LoadImaged(keys="image"),
        monai.transforms.EnsureChannelFirstd(keys="image"),
        monai.transforms.Orientationd(keys="image", axcodes="RAS"),
        monai.transforms.ScaleIntensityRanged(
            keys="image", a_min=-1000, a_max=1000, b_min=0, b_max=1, clip=True
        ),
        monai.transforms.EnsureTyped(keys="image", dtype=torch.float32),
    ])
    ```

2. **Apply Transformations and Save Embeddings**:
    ```python
    for _iter in range(len(filenames_raw)):
        if _iter % world_size != local_rank:
            continue

        filepath = filenames_raw[_iter]
        out_filename_base = os.path.join(output_root_embedding, filepath.replace("_image.nii.gz", ""))
        out_filename = out_filename_base + f"_emb.nii.gz"

        if not os.path.isfile(out_filename):
            test_data = {"image": os.path.join(dataroot, filepath)}
            transformed_data = transforms(test_data)
            nda = transformed_data["image"]

            spacing = nda.meta["pixdim"][1:4]
            nda = nda.numpy().squeeze()

            affine = np.eye(4)
            for _s in range(3):
                affine[_s, _s] = spacing[_s]

            out_path = Path(out_filename)
            out_path.parent.mkdir(parents=True, exist_ok=True)

            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    pt_nda = torch.from_numpy(nda).float().to(device).unsqueeze_(0).unsqueeze_(0)
                    z = autoencoder.encode_stage_2_inputs(pt_nda)
                    out_nda = z.squeeze().cpu().detach().numpy().transpose(1, 2, 3, 0)
                    out_img = nib.Nifti1Image(np.float32(out_nda), affine=affine)
                    nib.save(out_img, out_filename)
    ```

---

## Model Training

### Initialization and Setup

1. **Distributed Initialization**:
    ```python
    dist.init_process_group(backend="nccl", init_method="env://", timeout=timedelta(seconds=7200))
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    ```

2. **Load Dataset**:
    ```python
    with open(data_list, "r") as file:
        lines = file.readlines()
    filenames = [_item.strip() for _item in lines]
    filenames.sort()
    ```

3. **Data Partitioning**:
    ```python
    files = []
    for _i in range(len(filenames)):
        str_img = os.path.join(data_root, filenames[_i])
        str_info = os.path.join(data_root, filenames[_i].replace("_emb.nii.gz", "_image.nii.gz.json"))
        files.append({
            "image": str_img,
            "top_region_index": str_info,
            "bottom_region_index": str_info,
            "spacing": str_info,
        })
    train_files = partition_dataset(data=files, shuffle=True, num_partitions=dist.get_world_size(), even_divisible=True)[local_rank]
    ```

4. **Define Transformations**:
    ```python
    train_transforms = Compose([
        monai.transforms.LoadImaged(keys=["image"]),
        monai.transforms.EnsureChannelFirstd(keys=["image"]),
        monai.transforms.Lambdad(keys="top_region_index", func=lambda x: torch.FloatTensor(json.load(open(x))["top_region_index"])),
        monai.transforms.Lambdad(keys="bottom_region_index", func=lambda x: torch.FloatTensor(json.load(open(x))["bottom_region_index"])),
        monai.transforms.Lambdad(keys="spacing", func=lambda x: torch.FloatTensor(json.load(open(x))["spacing"])),
        monai.transforms.Lambdad(keys="top_region_index", func=lambda x: x * 1e2),
        monai.transforms.Lambdad(keys="bottom_region_index", func=lambda x: x * 1e2),
        monai.transforms.Lambdad(keys="spacing", func=lambda x: x * 1e2),
    ])
    ```

5. **Create Dataset and DataLoader**:
    ```python
    train_ds = monai.data.CacheDataset(data=train_files, transform=train_transforms, cache_rate=0.0, num_workers=2)
    train_loader = ThreadDataLoader(train_ds, num_workers=6, batch_size=1, shuffle=True)
    ```

### Model Definition and Training

1. **Define and Load Model**:
    ```python
    unet = CustomDiffusionModelUNet(
        spatial_dims=3,
        in_channels=4,
        out_channels=4,
        num_channels=(64, 128, 256, 512),
        attention_levels=(False, False, True, True),
        num_head_channels=(0, 0, 32, 32),
        num_res_blocks=2,
        use_flash_attention=True,
        input_top_region_index=True,
        input_bottom_region_index=True,
        input_spacing=True,
    )
    unet.to(device)
    unet = DistributedDataParallel(unet, device_ids=[device], find_unused_parameters=True)
    ```

2. **Define Optimizer and Scheduler**:
    ```python
    optimizer = torch.optim.Adam(params=unet.parameters(), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=total_steps, power=2.0)
    ```

3. **Training Loop**:
    ```python
    for epoch in range(num_epochs):
        unet.train()
        for train_data in train_loader:
            images = train_data["image"].to(device) * scale_factor
            top_region_index_tensor = train_data["top_region_index"].to(device)
            bottom_region_index_tensor = train_data["bottom_region_index"].to(device)
            spacing_tensor = train_data["spacing"].to(device)

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=True):
                noise = torch.randn(num_images_per_batch, 4, images.size(-3), images.size(-2), images.size(-1)).to(device)
                timesteps = torch.randint(0, inferer.scheduler.num_train_timesteps, (images.shape[0],), device=images.device).long()
                noise_pred = inferer(
                    inputs=images,
                    diffusion_model=unet,
                    noise=noise,
                    timesteps=timesteps,
                    top_region_index_tensor=top_region_index_tensor,
                    bottom_region_index_tensor=bottom_region_index_tensor,
                    spacing_tensor=spacing_tensor
                )
                loss = loss_pt(noise_pred.float(), noise.float())

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()
    ```

4. **Save Checkpoints**:
    ```python
    if local_rank == 0:
        unet_state_dict = unet.module.state_dict() if world_size > 1 else unet.state_dict()
        torch.save({"epoch": epoch + 1, "loss": loss_torch_epoch, "num_train_timesteps": num_train_timesteps, "scale_factor": scale_factor, "

scheduler_method": scheduler_method, "output_size": output_size, "unet_state_dict": unet_state_dict}, f"{ckpt_folder}/{ckpt_prefix}_current.pt")
    ```

---

## Model Inference

### Initialization and Setup

1. **Distributed Initialization**:
    ```python
    rank = int(os.environ.get("OMPI_COMM_WORLD_RANK"))
    local_rank = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK"))
    world_size = int(os.environ.get("OMPI_COMM_WORLD_SIZE"))
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    ```

2. **Set Deterministic Seed**:
    ```python
    rand_seed = random_seed + local_rank
    set_determinism(rand_seed)
    ```

3. **Define Model and Load Checkpoints**:
    ```python
    autoencoder = AutoencoderKLCKModified_TP(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        num_channels=(64, 128, 256),
        latent_channels=4,
        attention_levels=(False, False, False),
        num_res_blocks=(2, 2, 2),
        norm_num_groups=32,
        norm_eps=1e-06,
        with_encoder_nonlocal_attn=False,
        with_decoder_nonlocal_attn=False,
        use_checkpointing=False,
        use_convtranspose=False,
    )
    autoencoder.to(device)
    unet = CustomDiffusionModelUNet(
        spatial_dims=3,
        in_channels=4,
        out_channels=4,
        num_channels=(64, 128, 256, 512),
        attention_levels=(False, False, True, True),
        num_head_channels=(0, 0, 32, 32),
        num_res_blocks=2,
        use_flash_attention=True,
        input_top_region_index=True,
        input_bottom_region_index=True,
        input_spacing=True,
    )
    unet.to(device)

    checkpoint = torch.load(f"{ckpt_filepath}", map_location=device)
    unet.load_state_dict(checkpoint["unet_state_dict"])
    autoencoder.load_state_dict(checkpoint_autoencoder)
    ```

### Inference Process

1. **Generate Noise and Define Tensors**:
    ```python
    noise = torch.randn((1, 4, output_size[0] // 4, output_size[1] // 4, output_size[2] // 4)).to(device)
    top_region_index_tensor = torch.from_numpy(np.array([0, 1, 0, 0]).astype(float) * 1e2).float().to(device).unsqueeze(0)
    bottom_region_index_tensor = torch.from_numpy(np.array([0, 0, 1, 0]).astype(float) * 1e2).float().to(device).unsqueeze(0)
    spacing_tensor = torch.from_numpy(np.array(out_spacing).astype(float) * 1e2).float().to(device).unsqueeze(0)
    ```

2. **Inference**:
    ```python
    with torch.cuda.amp.autocast(enabled=amp):
        outputs = DiffusionInferer.sample(
            inferer,
            input_noise=noise,
            diffusion_model=unet,
            scheduler=scheduler,
            save_intermediates=False,
            intermediate_steps=False,
            top_region_index_tensor=top_region_index_tensor,
            bottom_region_index_tensor=bottom_region_index_tensor,
            spacing_tensor=spacing_tensor,
        )
        target_shape = output_size
        recon_pt_nda = torch.zeros((1, 1, target_shape[0], target_shape[1], target_shape[2]), dtype=outputs.dtype).to("cuda")
        _count = torch.zeros((1, 1, target_shape[0], target_shape[1], target_shape[2]), dtype=outputs.dtype).to("cuda")
        z = outputs

        # Reconstruction logic here...

        synthetic_images = recon_pt_nda / _count
        data = synthetic_images.squeeze().cpu().detach().numpy()
        data = (data - b_min) / (b_max - b_min) * (a_max - a_min) + a_min
        data = np.clip(data, a_min, a_max)
        data = np.int16(data)
    ```

3. **Save Output**:
    ```python
    out_affine = np.eye(4)
    for _k in range(3):
        out_affine[_k, _k] = out_spacing[_k]
    new_image = nib.Nifti1Image(data, affine=out_affine)
    nib.save(new_image, f'./predictions/{output_prefix}_seed{rand_seed}_size{output_size[0]:d}x{output_size[1]:d}x{output_size[2]:d}_spacing{out_spacing[0]:.2f}x{out_spacing[1]:.2f}x{out_spacing[2]:.2f}_{timestamp}.nii.gz')
    ```

This document provides an overview of the data resampling, model training, and model inference processes, following the structure and order of the provided Python scripts.
