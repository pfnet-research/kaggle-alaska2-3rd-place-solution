
master_port=29500


# Train 3-channel color image models for 150 epochs
for suffix in "" "-2nd" "-3rd"; do
    for color in "rgb" "yuv" "lab"; do
        python -m torch.distributed.launch --nproc_per_node=4 --master_port=${master_port} \
            scripts/rgb_naive/train.py \
            --resume \
            --config scripts/rgb_naive/${color}-4class-eb5-cutmix-p100${suffix}.yaml \
            --task train
        master_port=$(expr $master_port + 1)
    done
done

# Train a DCT-coefficient model
python -m torch.distributed.launch --nproc_per_node=4 --master_port=${master_port} \
    scripts/dct_d8/train.py \
    --resume \
    --config scripts/dct_d8/dct-d8-stride-1-2-l16-q-cos.yaml \
    --task train
master_port=$(expr $master_port + 1)

# Train an MLP to create pseudo labels for the last 50 epochs of training
for color in "rgb" "yuv" "lab"; do
    python -m torch.distributed.launch --nproc_per_node=4 --master_port=${master_port} \
        scripts/rgb_naive/train.py \
        --config scripts/rgb_naive/${color}-4class-eb5-cutmix-p100-3rd.yaml \
        --task predict_feature
    master_port=$(expr $master_port + 1)
done

python -m torch.distributed.launch --nproc_per_node=4 --master_port=${master_port} \
    scripts/dct_d8/train.py \
    --config scripts/dct_d8/dct-d8-stride-1-2-l16-q-cos.yaml \
    --task predict_feature
master_port=$(expr $master_port + 1)

python scripts/cnn_feature_concat/train.py \
    --config scripts/cnn_feature_concat/eb2-1st-dct-eb5-3rd-rgb-yuv-lab.yaml \
    --task train

python scripts/cnn_feature_concat/train.py \
    --config scripts/cnn_feature_concat/eb2-1st-dct-eb5-3rd-rgb-yuv-lab.yaml \
    --task predict

# Train 3-channel color image models for the last 50 epochs with pseudo labels
for color in "rgb" "yuv" "lab"; do
    python -m torch.distributed.launch --nproc_per_node=4 --master_port=${master_port} \
        scripts/rgb_naive/train.py \
        --resume \
        --config scripts/rgb_naive/${color}-4class-eb5-cutmix-p100-4th.yaml \
        --task train
    master_port=$(expr $master_port + 1)
done

# Train an MLP
for color in "rgb" "yuv" "lab"; do
    python -m torch.distributed.launch --nproc_per_node=4 --master_port=${master_port} \
        scripts/rgb_naive/train.py \
        --config scripts/rgb_naive/${color}-4class-eb5-cutmix-p100-4th.yaml \
        --task predict_feature
    master_port=$(expr $master_port + 1)
done

python scripts/cnn_feature_concat/train.py \
    --config scripts/cnn_feature_concat/eb2-1st-dct-eb5-4th-rgb-yuv-lab-softmax-pseudo.yaml \
    --task train

python scripts/cnn_feature_concat/train.py \
    --config scripts/cnn_feature_concat/eb2-1st-dct-eb5-4th-rgb-yuv-lab-softmax-pseudo.yaml \
    --task predict

# Train a holdout stacking model
python scripts/stacking/train_with_augmented_holdout.py \
    --config holdout-eb5-with-tsne.yaml

# the final submission (0.931 in the private LB) is
#     data/working/stacking-holdout-eb5-with-tsne/submission.csv
