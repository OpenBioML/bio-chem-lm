# Training an Electra-style Model on PubChem Selfies
Here we will detail the training of an Electra-style model on the PubChem dataset with SELFIES representations.

## Coordinate Checking
We have plotted coordinate checks for varying model widths to verify that the model is using µP correctly. The results are shown in `coord_check/`

To create additional coordinate checks, run the following command:
```bash
python coord_check.py
python coord_check.py --mup
```

## Training
We are leveraging [HuggingFace Accelerate](https://huggingface.co/docs/accelerate/index) to handle distributed training. To train the model, run the following command:
```bash
accelerate launch train.py <training args found in options.py>
```

## Hyperparameter Search
To launch our hyperparameter search, we are using a combination of `accelerate` and `wandb`. To launch a sweep, run the following command:
```bash
wandb sweep sweep.yaml
wandb agent <sweep_id>
```
