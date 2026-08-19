FROM nvidia/cuda:11.1.1-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        python-is-python3 \
        python3 \
        python3-dev \
        python3-pip \
    \
 && rm -rf /var/lib/apt/lists/* \
 && python -m pip install --upgrade \
        pip \
    \
 && python -m pip install \
        cython \
    \
 && python -m pip install \
        numpy==1.23.5 \
    \
 && python -m pip install \
        torch==1.9.0+cu111 \
        torchvision==0.10.0+cu111 \
    -f https://download.pytorch.org/whl/torch_stable.html \
 && python -m pip install \
        "git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI&egg=pycocotools" \
        "git+https://github.com/cocodataset/panopticapi.git#egg=panopticapi" \
    \
 && python -m pip install \
        addict \
        albumentations==1.3.0 \
        ipdb \
        ipykernel \
        ipywidgets \
        matplotlib \
        opencv-python-headless \
        pillow \
        pyyaml \
        scipy==1.10.1 \
        seaborn \
        submitit \
        termcolor==2.4.0 \
        timm==0.6.13 \
        tqdm \
        yapf==0.32.0 \
    ;

CMD ["bash"]
