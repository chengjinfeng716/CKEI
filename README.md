## Empathetic Response Generation via Commonsense Knowledge and Emotional Intensity
This is the official implementation for paper Empathetic Response Generation via Commonsense Knowledge and Emotional Intensity

## Setup
Install the required libraries (Python 3.8.10 | CUDA 11.3)

```console
pip install -r requirements.txt
```

## Dataset
The preprocessed dataset is already provided in the project data directory `/data`

## Training
```sh
python train.py [--woIntent] [--woNeed] [--woWant] [--woEffect] [--woReact] [--woEmotionalIntensity]
```
The extra flags is used for ablation studies.

## Evaluation
```sh
python evaluate.py --best_mode_file result/model/xxxxxxx.model [--woIntent] [--woNeed] [--woWant] [--woEffect] [--woReact] [--woEmotionalIntensity]
```
Where best_mode_file is the trained Model after training process.



For reproducibility, download the trained [checkpoint](https://drive.google.com/file/d/1Yca30b1spNSNrZDwgh-QwV0sMKQR6afG/view?usp=sharing),  move it into the folder named  `result/model` and run the following:

```sh
python evaluate.py --best_mode_file result/model/best_model-36000-25.2731.model
```
