FROM nvcr.io/nvidia/pytorch:25.08-py3

ARG USERNAME
ARG USER_UID
ARG USER_GID
ARG DEBIAN_FRONTEND=noninteractive

ENV TZ=Europe/Paris
# ENV TF_USE_LEGACY_KERAS=1

# Remove existing group/user if they occupy the ID (e.g. ubuntu user)
RUN groupdel -f $(getent group $USER_GID | cut -d: -f1) || true 
RUN userdel -f $(getent passwd $USER_UID | cut -d: -f1) || true 

# Create our fresh user
RUN groupadd --gid $USER_GID $USERNAME 
RUN useradd --uid $USER_UID --gid $USER_GID -m $USERNAME 

# Install libgl1
RUN apt-get update && apt-get install libgl1 -y 

# Configure the environement
USER $USERNAME
WORKDIR /home/$USERNAME
COPY donkeycar donkeycar
COPY setup_uv.sh .
COPY pyproject.toml .

# Settung up the virtual environment
RUN ./setup_uv.sh --platform ngc --extras torch_spark

# Upgrade ml_dtypes for ONNX export (conflicts with tensorflow but works at runtime)
RUN .venv/bin/pip install "ml_dtypes>=0.5.0"

# Activate the virtual environment and add alias to check if there is GPU
RUN echo "source .venv/bin/activate" >> /home/$USERNAME/.bashrc
# RUN echo 'alias hasgpu="python -c '\''import tensorflow as tf; print(tf.config.list_physical_devices(\"GPU\"))'\''"' >> /home/$USERNAME/.bashrc
RUN echo 'alias hasgpu="python -c '\''import torch; print(torch.cuda.is_available())'\''"' >> /home/$USERNAME/.bashrc