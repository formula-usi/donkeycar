FROM nvcr.io/nvidia/tensorflow:24.05-tf2-py3

ARG USERNAME
ARG USER_UID
ARG USER_GID
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Paris
# Create the user
RUN groupadd --gid $USER_GID $USERNAME 
RUN useradd --uid $USER_UID --gid $USER_GID -m $USERNAME 

# # Remove NVIDIA internal sources that cause apt to hang
# RUN sed -i '/nvidia\.com/d' /etc/apt/sources.list && \
#     sed -i '/nvidia/d' /etc/apt/sources.list.d/*.list || true

# # Add official Ubuntu mirrors
# RUN echo "deb http://archive.ubuntu.com/ubuntu jammy main restricted universe multiverse" > /etc/apt/sources.list && \
#     echo "deb http://archive.ubuntu.com/ubuntu jammy-updates main restricted universe multiverse" >> /etc/apt/sources.list && \
#     echo 'deb http://security.ubuntu.com/ubuntu jammy-security main restricted universe multiverse' >> /etc/apt/sources.list


# # Add sudo support. Omit if you don't need to install software after connecting.
# RUN apt-get update \
#     && apt-get install -y sudo software-properties-common lsb-release \
#     && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
#     && chmod 0440 /etc/sudoers.d/$USERNAME

# # Add Deadsnakes PPA and install Python 3.11 as root (do this before switching to the unprivileged user)
# RUN add-apt-repository ppa:deadsnakes/ppa \
#     && apt-get update \
#     && apt-get install -y python3.11 \
#     && rm -rf /var/lib/apt/lists/*

# RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 2
# RUN update-alternatives --set python3 /usr/bin/python3.11

USER $USERNAME
WORKDIR /home/$USERNAME
COPY donkeycar donkeycar
COPY setup_uv.sh .
COPY pyproject.toml .
RUN ./setup_uv.sh --platform ngc

RUN echo "source .venv/bin/activate" >> /home/$USERNAME/.bashrc
RUN echo 'alias hasgpu="python -c '\''import tensorflow as tf; print(tf.config.list_physical_devices(\"GPU\"))'\''"' \
    >> /home/$USERNAME/.bashrc