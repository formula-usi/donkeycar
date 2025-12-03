FROM nvcr.io/nvidia/tensorflow:24.05-tf2-py3

ARG USERNAME
ARG USER_UID
ARG USER_GID
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Paris
# Create the user
RUN groupadd --gid $USER_GID $USERNAME 
RUN useradd --uid $USER_UID --gid $USER_GID -m $USERNAME 

RUN apt-get update && apt-get install libgl1 -y 

USER $USERNAME
WORKDIR /home/$USERNAME
COPY donkeycar donkeycar
COPY setup_uv.sh .
COPY pyproject.toml .
RUN ./setup_uv.sh --platform ngc

RUN echo "source .venv/bin/activate" >> /home/$USERNAME/.bashrc
RUN echo 'alias hasgpu="python -c '\''import tensorflow as tf; print(tf.config.list_physical_devices(\"GPU\"))'\''"' \
    >> /home/$USERNAME/.bashrc