import sys
sys.path.append("../../")

import pandas as pd
import torch
import math
import pickle
import os

import models.resnet as ResNet

from utils.data_mappers import LabeledPickleDatasetMapper
from utils.preprocessing import Preprocessor
from utils.constants import (
    DATASET_ROOT,
    DATASET_TRAIN_DATA_FILE,
    DATASET_TEST_DATA_FILE,
)
from utils.constants import SCORE_FUNCTIONS_CLASSIFICATION

from pipelines.classification_pipeline import Pipeline

import argparse

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

if __name__ == "__main__":
    PATH_PREFIX = "../../"

    parser = argparse.ArgumentParser(prog="train_resnet50", description="Training ResNet50 from scratch")
    parser.add_argument("model_name", type=str, help='Model Name (without spaces)')
    parser.add_argument("subset_csv_path", type=str, help='Path to subset csv')
    parser.add_argument("save_models", type=str, help='Path to directory to save models')
    parser.add_argument("--cuda-num", dest="cuda_num", type=int, help="Device number for cuda")

    args = parser.parse_args()

    model_name = args.model_name
    subset_file = args.subset_csv_path
    save_dir = args.save_models
    cuda_num = args.cuda_num

    # model_name = "teacher_resnet50_0-0.1_sub_set3_6_conn_sig"
    print("Name: {}".format(model_name))
    print("Subset File: {}".format(subset_file))
    print("Save Dir: {}".format(save_dir))
    print("Cuda Num: {}".format(cuda_num))

    preprocessor = Preprocessor((32,32))

    model = ResNet.resnet50(num_classes=10, include_top=True)

    cifar_train = unpickle("../../dataset/cifar-10-python/train/train_batch")
    cifar_test = unpickle("../../dataset/cifar-10-python/test/test_batch")

    
    print(cifar_test.keys())
    
    train_set = LabeledPickleDatasetMapper(
        cifar_train['data'].copy(),
        cifar_train['labels'].copy(),
        subset_file,
        preprocessor,
        augment=True,
        return_idx=True
    )

    full_train_set = LabeledPickleDatasetMapper(
        cifar_train['data'].copy(),
        cifar_train['labels'].copy(),
        None,
        preprocessor,
        augment=False,
        return_idx=False
    )

    test_set = LabeledPickleDatasetMapper(
        cifar_test[b'data'].copy(),
        cifar_test[b'labels'].copy(),
        None,
        preprocessor,
        augment=False,
        return_idx=False
    )

    trainer = Pipeline(
        name=model_name,
        model=model,
        batch_size=256,
        workers=8,
        train_set=train_set,
        test_sets=[{"name": "full_train_set", "dataset": full_train_set}, {"name": "test_set", "dataset": test_set}],
        preprocessor=preprocessor,
        log_files_path="logs/fit/",
        cuda_num=cuda_num
    )

    num_epochs = 200
    lr = 0.4
    # step_size_func = lambda e: 1 / math.sqrt(1 + e)
    step_size_func = lambda e: ((e - num_epochs*0.15)/(num_epochs*0.15) + 1) if e <= num_epochs*0.15 else (num_epochs - e)/(num_epochs*0.85)

    loss_func_with_grad = torch.nn.CrossEntropyLoss()
    loss_func = torch.nn.functional.cross_entropy

    training_log, validation_log, _, _ = trainer.train(
        num_epochs=num_epochs,
        lr=lr,
        step_size_func=step_size_func,
        loss_func_with_grad=loss_func_with_grad,
        loss_func=loss_func,
        score_functions=SCORE_FUNCTIONS_CLASSIFICATION,
        # save_checkpoints_epoch=25,
        # save_checkpoints_path="../../saved_models/",
    )

    trainer.save(save_dir, num_epochs)

    os.makedirs(os.path.join("logs/logfiles", trainer.name))
    with open(
        os.path.join("logs/logfiles", trainer.name, "training_log.pkl"), "wb"
    ) as f:
        pickle.dump(training_log, f)

    with open(
        os.path.join("logs/logfiles", trainer.name, "validation_log.pkl"), "wb"
    ) as f:
        pickle.dump(validation_log, f)
