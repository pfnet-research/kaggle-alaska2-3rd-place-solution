3rd place solution of ALASKA2 Image Steganalysis
===

Code for 3rd place solution of [ALASKA2 Image Steganalysis](https://www.kaggle.com/c/alaska2-image-steganalysis/overview) on Kaggle.

A description of the method can be found in the following paper and [this post](https://www.kaggle.com/c/alaska2-image-steganalysis/discussion/168870) in the kaggle discussion.
> [**An Ensemble Model using CNNs on Different Domains for ALASKA2 Image Steganalysis**](https://ieeexplore.ieee.org/document/9360892)  
> Kaizaburo Chubachi

Run Instructions
---

### Environments

Run it on the docker image shown in `env/Dockerfile`.

(Optional) We mounted `tmpfs` in `/tmp/ram` because we used there to pass images read from the zip file to `jpegio` during training.

### Data Preparation

```bash
# download data
mkdir -p data/input/
cd data/input/
kaggle competitions download alaska2-image-steganalysis
unzip alaska2-image-steganalysis.zip
cd ../../

# clone codes for some architectures
git clone https://github.com/facebookresearch/pycls
git clone https://github.com/iduta/pyconv

# extract metadata such as qualities of JPEG files.
python scripts/make_assets/make_quality_df.py
python scripts/make_assets/make_payload_stats.py
```

### Reproduce the final submission

Run `reproduce_final_submission.sh`.  We primarily used four NVIDIA Tesla V100 16GB to perform the training, so the script is written to use four GPUs.

### Train 3-channel color image models

Make a configuration file like `scripts/rgb_naive/rgb-4class-eb5-cutmix-p100.yaml`. Put the file as `scripts/rgb_naive/${CONFIG_NAME}.yaml`. Then run following commands.

```bash
python scripts/rgb_naive/train.py \
    --config scripts/rgb_naive/${CONFIG_NAME}.yaml \
    --task download_model
python -m torch.distributed.launch --nproc_per_node=4 \
    scripts/rgb_naive/train.py \
    --resume \
    --config scripts/rgb_naive/${CONFIG_NAME}.yaml \
    --task train
```

### Train a DCT coefficient model

Make a configuration file like `scripts/dct_d8/dct-d8-stride-1-2-l16-q-cos.yaml`. Put the file as `scripts/dct_d8/${CONFIG_NAME}.yaml`. Then run following commands.

```bash
python scripts/dct_d8/train.py \
    --config scripts/dct_d8/${CONFIG_NAME}.yaml \
    --task download_model
python -m torch.distributed.launch --nproc_per_node=4 \
    scripts/dct_d8/train.py \
    --resume \
    --config scripts/dct_d8/${CONFIG_NAME}.yaml \
    --task train
```

### Train a MLP

Before training, dump feature maps of trained CNNs.

```bash
python -m torch.distributed.launch --nproc_per_node=4 \
    scripts/rgb_naive/train.py \
    --config scripts/rgb_naive/${CONFIG_NAME}.yaml \
    --task predict_feature
```

Make a configuration file like `scripts/cnn_feature_concat/dct-d8-stride-1-2-l16-q-cos.yaml`, and put the file as `scripts/dct_d8/${CONFIG_NAME}.yaml`. Then run following commands.

```bash
python scripts/cnn_feature_concat/train.py \
    --config scripts/cnn_feature_concat/${CONFIG_NAME}.yaml \
    --task train
```

Citation
---

```
@inproceedings{chubachi2020alaska,
  author={Kaizaburo Chubachi},
  booktitle={2020 IEEE International Workshop on Information Forensics and Security (WIFS)}, 
  title={An Ensemble Model using CNNs on Different Domains for ALASKA2 Image Steganalysis}, 
  year={2020},
  pages={1-6},
  doi={10.1109/WIFS49906.2020.9360892}
}
```
