# B2TPT
This is the implementation of " Black-box Test-Time Prompt Tuning for Vision-Language Models ".

## Framework
![image](https://github.com/MFAaaaaaa/B2TPT/blob/main/model/A-Fram-B2TPT.png)
## Installation
* Clone this repository.
```bash
https://github.com/MFAaaaaaa/B2TPT.git
```
## Datasets
* Please manually download the datasets from the official websites, and modify the path of images.
## Training
```
python B2TPT.py --task_name caltech101 --opt B2TPT_config
```
## Acknowledgements
```
Our code is based on [CoOp](https://github.com/KaiyangZhou/CoOp), [MAPLE](https://github.com/muzairkhattak/multimodal-prompt-learning), [BPT-VLM](https://github.com/BruthYU/BPT-VLM). We thank the authors for releasing their code. If you use our model and code, please consider citing these works as well.
```
