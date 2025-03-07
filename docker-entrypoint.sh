#!/bin/bash
set -e  # Exit on error

# Check if GPU is available
if command -v nvidia-smi &> /dev/null; then
    echo "GPU detected."
    # Check driver version
    DRIVER_CUDA_VERSION=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+')
    echo "nvidia-smi reports CUDA version: $DRIVER_CUDA_VERSION"

    # Check installed CUDA version
    if command -v nvcc &> /dev/null; then
        INSTALLED_CUDA_VERSION=$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+')
    else
        INSTALLED_CUDA_VERSION="Not Installed"
    fi
    echo "Installed CUDA version: $INSTALLED_CUDA_VERSION"

    if [[ "$INSTALLED_CUDA_VERSION" != "Not Installed" && "$DRIVER_CUDA_VERSION" != "$INSTALLED_CUDA_VERSION" ]]; then
        echo "CUDA version mismatch detected. Attempting to create symbolic links..."

        ln -sf /usr/local/cuda-$INSTALLED_CUDA_VERSION/lib64/libcublasLt.so.11.4.1.1043 /usr/local/cuda-$INSTALLED_CUDA_VERSION/lib64/libcublasLt.so.${DRIVER_CUDA_VERSION%.*}
        ln -sf /usr/local/cuda-$INSTALLED_CUDA_VERSION/lib64/libcublas.so.11.4.1.1043 /usr/local/cuda-$INSTALLED_CUDA_VERSION/lib64/libcublas.so.${DRIVER_CUDA_VERSION%.*}

        echo "Symbolic links created."
    else
        echo "No mismatch detected. Proceeding..."
    fi
else
    echo "No GPU detected. Running without GPU."
fi

echo "Starting deepaas..."
exec deepaas-run --listen-ip 0.0.0.0 --listen-port 5000
