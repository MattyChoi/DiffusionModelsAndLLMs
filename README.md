# Diffusion Module

This repo contains an implementation of denoising diffusion modules with the PyTorch Lightning framework with Hydra for organizing config files.

Once in the directory containing the contents of the repository, run
```
pip install -r requirements.txt
```
if you do not have all the necessary packages listed.

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
