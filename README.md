# Diffusion Module

This repo contains an implementation of denoising diffusion modules with the PyTorch Lightning framework with Hydra for organizing config files.

Once in the directory containing the contents of the repository, run
```
pip install -r requirements.txt
```
if you do not have all the necessary packages listed. Or, if you are using conda, you use the commands

```
conda env create --file environment.yml
conda activate diffllm
```

More on conda with GPUS:
 - https://fmorenovr.medium.com/set-up-conda-environment-pytorch-1-7-cuda-11-1-96a8e93014cc

## Instructions 

To train the model, run the following command in the terminal:
```
python tools/trainer.py
```

and to test the model, 
```
python tools/predictor.py
```

To observe tensorboard logs if enabled, use the following command
```
tensorboard --logdir ./lightning_logs/{current version}
```

Some useful links
 - https://github.com/karpathy/nanoGPT
 - https://github.com/jon-tow/text-sed
 - https://github.com/Infatoshi/fcc-intro-to-llms
 - https://towardsdatascience.com/train-and-deploy-fine-tuned-gpt-2-model-using-pytorch-on-amazon-sagemaker-to-classify-news-articles-612f9957c7b
 - https://medium.com/@jacobparnell/tune-transformers-using-pytorch-lightning-and-huggingface-f056373ff0e3
 - https://github.com/huggingface/transformers/blob/298bed16a841fae3608d334441ccae4d9043611f/src/transformers/modeling_gpt2.py#L146
 - https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
 


## Docker

Run a docker container from a docker image built from the Dockerfile  

```
docker build -t diffllm .
```

and then run a container using this command

```
docker run --name diffllm --gpus all -it --rm diffllm
```

Used these resources to help make dockerfile
 - https://saturncloud.io/blog/how-to-install-pytorch-on-the-gpu-with-docker/
 - https://stackoverflow.com/questions/65492490/how-to-conda-install-cuda-enabled-pytorch-in-a-docker-container

