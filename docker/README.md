# DigiForests Docker Environment

This README provides instructions for setting up and using the Docker environment for the DigiForests project, which includes GPU support and a multi-stage build process.

> Note: Host machine should support CUDA 11.8!

## Prerequisites

1. Install Docker on your system.
2. Install NVIDIA Container Runtime:
   Follow the instructions in the [NVIDIA Documentation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

3. Verify GPU support in Docker:
   ```bash
   docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
   ```
   This should display your GPU information if everything is set up correctly.

## Building the DigiForests Docker Image

The Dockerfile uses a 3-stage build process:

1. Base CUDA and PyTorch setup
2. MinkowskiEngine compilation
3. DigiForests package installation

**Note**: The build process can take 30+ minutes, especially the MinkowskiEngine compilation.

### Important Considerations

- Ensure your host machine supports CUDA 11.8, as this is the base image which we build upon.
- In the Dockerfile, adjust the `TORCH_CUDA_ARCH_LIST` environment variable to match your GPU's compute capability.

### Build Command

```bash
docker build -t digiforests_devkit -f docker/Dockerfile .
```

## Running the Container

### Using Docker Run

```bash
docker run -it --rm --gpus all digiforests_devkit
```

### Using Docker Compose

1. Ensure your user ID and group ID are set in the environment:

   ```bash
   export UID=$(id -u)
   export GID=$(id -g)
   ```

2. Run the container:
   ```bash
   docker compose -f docker/compose.yaml run devkit
   ```

## Customization

- The `compose.yaml` file includes volume mounts for development. Adjust these as needed.
- Uncomment the data volume mount in `compose.yaml` to access external data.
