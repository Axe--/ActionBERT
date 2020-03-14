# ActionBERT
### Is Attention All That We Need?

Investigating Transformers for Action Recognition (Video classification)


>The aim of this work is to understand the sequence modelling capabilities 
of transformer models (BERT-like) for continuous input spaces such as video frames, unlike
language where the inputs are discrete (vocabulary).

> PyTorch Implementation

---


## Table of Contents

> The project comprises of the following sections.
- [Dataset](#dataset)
- [Architecture](#architecture)
- [Training](#training)
- [Inference](#inference)

---

## Dataset

Given the path to [UCF-101's](https://www.crcv.ucf.edu/data/UCF101.php) 
raw dataset folder, prepare the dataset in a standardized format as follows:

```bash
$ python3 prepare_ucf101.py \
-v /home/axe/Datasets/UCF_101/raw/videos \
-o /home/axe/Datasets/UCF_101 \
-s 0.8 -fps 1
```

Produces json files for training & validation sets in the following format:

```json5
[
  {
    "video_name": "str",
    "label_idx": "int",
  }
]
```

Also creates a new directory within the output dir `-o`
for storing video frames, organized as follows:
```
├── out_dir
    │
    └── frames
        │
        ├── video_1
        │    ├── frame_1.jpg
        │    │   ...
        │    └── frame_n.jpg
        │
        └── video_k
             ├── frame_1.jpg
             │   ...
             └── frame_m.jpg
```
<br>
---

Given the above frames dir `-f` & split set json `-j`, 
produces the final json & embeddings file (npy) in the 
following format:

```bash
$ python3 prepare_data.py -s train \
-f /home/axe/Datasets/UCF_101/frames_1_fps \
-j /home/axe/Datasets/UCF_101/train_ucf101.json \
-o /home/axe/Datasets/UCF_101/data_res18_fps_1 \
-m resnet18 -bs 1024 -nw 4
```

The files are stored in output dir `-o`. <br>

<b>Processed dataset</b>
```json5
{
  "data": [
            {
                "video_idx": "int",
                "video_name": "str",
                "video_length": "int",
                "label_idx": "int"
            }
         ],
  "memmap_size": "tuple(total_videos, max_video_len, emb_dim)",    
  "split": "str"
}
```
The `video_idx` refers to the 0<sup>th</sup> axis of the embeddings array.

<b>Embeddings</b>
```
np.array(shape=[total_videos, max_video_len, emb_dim])
```



---
## Architecture


### BiLSTM


- Pre-Trained Conv + LSTM

- End-to-End Conv + LSTM

<br>


### Transformer


- Pre-Trained Conv + Transformer

- End-to-End Conv + Transformer?


<br>

---

## Training

Run the following script for training:

```bash
$ python3 main.py --mode train --expt_dir ./results_log  \
--expt_name BERT --model bert \
--data_dir ~/Datasets/UCF_101/processed_fps_1_res18 \
--run_name res18_1fps_lyr_1_bs_256_lr_1e4 \
--num_layers 1 --batch_size 256 --epochs 300 \
--gpu_id 1 --opt_lvl 1  --num_workers 4 --lr 1e-4
```
Specify `--model_ckpt` (filename.pth) to load model checkpoint from disk <i>(resume training/inference)</i> <br>

Select the architecture by using `--model` ('bilstm', 'bert', 'roberta'). <br>

For pre-trained transformers see this 
<a href="https://huggingface.co/transformers/pretrained_models.html"> link. </a> <br>

> *Note*: ...


### Inference 

- **....**


---

<br>

UCF-101 Dataset            | `  
:-------------------------:|:-------------------------:
![Crawling](assets/crawling.gif)  |  ![Penalty](assets/penalty.gif)

<br>
---

> *TODO*: ....


- [x] With Pre-Trained (Train+Val)
- [ ] End-to-End (Train+Val)
- [ ] ..