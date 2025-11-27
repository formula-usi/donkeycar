FROM nvcr.io/nvidia/tensorflow:25.01-tf2-py3

ARG USERNAME
ARG USER_UID
ARG USER_GID

# Create the user
RUN groupadd --gid $USER_GID $USERNAME 
RUN useradd --uid $USER_UID --gid $USER_GID -m $USERNAME 

# Add sudo support. Omit if you don't need to install software after connecting.
RUN apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

USER $USERNAME
WORKDIR /home/$USERNAME

COPY donkeycar donkeycar
COPY setup_uv.sh .

RUN ./setup_uv.sh --platform ngc

RUN echo "source .venv/bin/activate" >> /home/$USERNAME/.bashrc
RUN echo 'alias hasgpu="python -c '\''import tensorflow as tf; print(tf.config.list_physical_devices(\"GPU\"))'\''"' \
    >> /home/$USERNAME/.bashrc