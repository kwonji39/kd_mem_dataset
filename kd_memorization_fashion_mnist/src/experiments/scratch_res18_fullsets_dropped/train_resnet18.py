import sys
sys.path.append("../../")

import pandas as pd
import torch
import math
import pickle
import os
import numpy as np
import idx2numpy
import models.resnet as ResNet

from utils.data_mappers import LabeledPickleDatasetMapper
from utils.preprocessing import Preprocessor
from utils.constants import (
    DATASET_ROOT,
    DATASET_TRAIN_DATA_FILE,
    DATASET_TEST_DATA_FILE,
)
from utils.constants import SCORE_FUNCTIONS_CLASSIFICATION

from pipelines.dropping_classification_pipeline import Pipeline

from temp.eval_model import model_t
from temp.eval_model import model_s
import argparse
# from torchsummary import summary

if __name__ == "__main__":
    PATH_PREFIX = "../../../"

    parser = argparse.ArgumentParser(prog="train_resnet18", description="Training ResNet18 from scratch")
    parser.add_argument("model_name", type=str, help='Model Name (without spaces)')
    # parser.add_argument("subset_csv_path", type=str, help='Path to subset csv')
    parser.add_argument("save_models", type=str, help='Path to directory to save models')
    parser.add_argument("--cuda-num", dest="cuda_num", type=int, help="Device number for cuda")
    parser.add_argument("--num-workers", dest="num_workers", type=int, help="Number of workers for dataloader")
    parser.add_argument("--reduce", dest="dim_scale_factor", type=int,help="Scale down student model dimenstions byt factor")
    parser.add_argument("--drop", dest="drop_factor", type=float, help="Drop training samples with high gradients")
    parser.add_argument("--drop_epoch", dest="drop_epoch", type=int, help="Start dropping at given epoch")


    args = parser.parse_args()

    model_name = args.model_name
    # subset_file = args.subset_csv_path
    save_dir = args.save_models
    cuda_num = args.cuda_num
    num_workers = args.num_workers
    dim_scale_factor = args.dim_scale_factor
    drop_factor = args.drop_factor
    drop_epoch = args.drop_epoch

    # model_name = "teacher_resnet50_0-0.1_sub_set3_6_conn_sig"
    print("Name: {}".format(model_name))
    # print("Subset File: {}".format(subset_file))
    print("Save Dir: {}".format(save_dir))
    print("Cuda Num: {}".format(cuda_num), flush=True)
    print("Dimension Reducing: {}".format(dim_scale_factor), flush=True)
    print("Drop Factor: {}".format(drop_factor))
    print("Drop Epoch: {}".format(drop_epoch))

    dataset_base_path = "/scratch/gilbreth/kwon165/kd_memorization/src/dataset"
    train_images_path = os.path.join(dataset_base_path, "train-images-idx3-ubyte")
    train_labels_path = os.path.join(dataset_base_path, "train-labels-idx1-ubyte")
    test_images_path = os.path.join(dataset_base_path, "t10k-images-idx3-ubyte")
    test_labels_path = os.path.join(dataset_base_path, "t10k-labels-idx1-ubyte")

    preprocessor = Preprocessor((28, 28))

    model = ResNet.resnet18(num_classes=10, include_top=True)

    train_images = idx2numpy.convert_from_file(train_images_path)
    train_labels = idx2numpy.convert_from_file(train_labels_path)
    test_images = idx2numpy.convert_from_file(test_images_path)
    test_labels = idx2numpy.convert_from_file(test_labels_path)

    # (N, H, W) to (N, C, H, W)
    train_images = np.expand_dims(train_images, axis=1)
    test_images = np.expand_dims(test_images, axis=1)

    # Initialize the dataset mapper
    train_set = LabeledPickleDatasetMapper(
        data=train_images,
        labels=train_labels,
        image_data_file=None,
        preprocessor=preprocessor,
        augment=True
    )

    test_set = LabeledPickleDatasetMapper(
        data=test_images,
        labels=test_labels,
        image_data_file=None,
        preprocessor=preprocessor,
        augment=False
    )

    # full_train_set = LabeledPickleDatasetMapper(
    #     cifar_train[b'data'].copy(),
    #     cifar_train[b'fine_labels'].copy(),
    #     None,
    #     preprocessor,
    #     augment=False
    # )

    score_sets = [
        # {"name": "full_train_set", "dataset": full_train_set}, 
        # *score_sets_low_mem,
        # *score_sets_high_mem,
        {"name": "test_set", "dataset": test_set},
    ]

    trainer = Pipeline(
        name=model_name,
        model=model_s,
        batch_size=256,
        workers=8,
        train_set=train_set,
        test_sets=score_sets,
        preprocessor=preprocessor,
        log_files_path="logs/fit/",
        # teacher_model= torch.nn.Module,
        teacher_model=model_t,
        cuda_num=cuda_num
    )

    num_epochs = 200
    lr = 0.4
    # step_size_func = lambda e: 1 / math.sqrt(1 + e)
    step_size_func = lambda e: ((e - num_epochs*0.15)/(num_epochs*0.15) + 1) if e <= num_epochs*0.15 else (num_epochs - e)/(num_epochs*0.85)

    loss_func_with_grad = torch.nn.CrossEntropyLoss()
    loss_func = torch.nn.functional.cross_entropy

    training_log, validation_log, grad_log1, grad_log2, _ = trainer.train(
        num_epochs=num_epochs,
        lr=lr,
        step_size_func=step_size_func,
        loss_func_with_grad=loss_func_with_grad,
        loss_func=loss_func,
        score_functions=SCORE_FUNCTIONS_CLASSIFICATION,
        validation_score_epoch=5,
        drop_percentage=drop_factor,
        drop_epoch=drop_epoch
    )

    save_path = (trainer.save(save_dir, num_epochs))

    with open("{}_save_path.txt".format(model_name), 'w') as f:
        f.write(save_path)

    os.makedirs(os.path.join("logs/logfiles", trainer.name))
    with open(
        os.path.join("logs/logfiles", trainer.name, "training_log.pkl"), "wb"
    ) as f:
        pickle.dump(training_log, f)

    with open(
        os.path.join("logs/logfiles", trainer.name, "validation_log.pkl"), "wb"
    ) as f:
        pickle.dump(validation_log, f)

    with open(
        os.path.join("logs/logfiles", trainer.name, "grad_log1.pkl"), "wb"
    ) as f:
        pickle.dump(grad_log1, f)

    with open(
        os.path.join("logs/logfiles", trainer.name, "grad_log2.pkl"), "wb"
    ) as f:
        pickle.dump(grad_log2, f)
