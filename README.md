# Diffusion Module

This repo contains an implementation of denoising diffusion modules with the PyTorch Lightning framework with Hydra for organizing config files.

Once in the directory containing the contents of the repository, run
```
pip install -r requirements.txt
```
if you do not have all the necessary packages listed.

## Train 

To train the model, run the following command in the terminal:
```
python tools/trainer.py
```

and to test the model, 
```
python tools/predictor.py
```
