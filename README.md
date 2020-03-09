# ActionBERT
### Is Attention All That We Need?

Investigating Transformers for Action Recognition


>The aim of this work is to understand the sequence modelling capabilities 
of transformer models (BERT-like) for continuous inputs such as videos, unlike
language where we have discrete vocabulary.

> PyTorch implementation

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
raw dataset folder, generates a json file & embeddings file in the following format:

<b>Processed dataset</b>
```json
{
  "data": [
            {
                "video_idx": "int",
                "video_name": "str",
                "video_length": "int",
                "label_idx": "int"
            }
         ],
  "memmap_size": "(total_videos, max_video_len, emb_dim)" ,    
  "split": "str"
}
```
The `video_idx` refers to the 0<sup>th</sup> axis of the embeddings array.

<b>Embeddings</b>
```
np.array(shape=[total_videos, max_video_len, emb_dim])
```

Generate the train & validation set json files.

```bash
$ python3 prepare_ucf101.py \
-v /home/axe/Datasets/UCF_101/videos \
-o /home/axe/Datasets/UCF_101 \
-s 0.8 -fps 1
```

The outputs are the ... (temp csv & frame images)

```bash
$ python3 prepare_data.py -s train \
-f /home/axe/Datasets/UCF_101/frames_1_fps \
-c /home/axe/Datasets/UCF_101/train_temp.csv \
-o /home/axe/Datasets/UCF_101/processed_fps_1_res18 \
-m resnet18 -bs 1024 -nw 4
```

Stores the dataset file (json) & embeddings file (npy) in the 
 output directory `-o`, for the given split set `-c`<br>

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