
FROM nvidia/cuda:11.6.2-base-ubuntu20.04

WORKDIR /app 

COPY . . 

# Install system dependencies
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
        git \
        python3-pip \
        python3-dev \
        python3-opencv \
        libglib2.0-0
  
# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Install any needed packages specified in requirements.txt
RUN python3 -m pip install -r requirements.txt


# CMD ["python", "tools/trainer.py"] 