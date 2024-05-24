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

# from pipelines.dropping_classification_pipeline import Pipeline as PipelineS1
# from pipelines.classification_pipeline import Pipeline as PipelineS2
from pipelines.classification_pipeline import Pipeline

import argparse
# from torchsummary import summary
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

if __name__ == "__main__":
    PATH_PREFIX = "../../"

    parser = argparse.ArgumentParser(prog="train_resnet18", description="Training ResNet18 from scratch")
    parser.add_argument("model_name", type=str, help='Model Name (without spaces)')
    parser.add_argument("subset_csv_path", type=str, help='Path to subset csv')
    parser.add_argument("save_models", type=str, help='Path to directory to save models')
    parser.add_argument("--cuda-num", dest="cuda_num", type=int, help="Device number for cuda")
    parser.add_argument("--num-workers", dest="num_workers", type=int, help="Number of workers for dataloader")
    parser.add_argument("--reduce", dest="dim_scale_factor", type=int,help="Scale down student model dimenstions byt factor")

    args = parser.parse_args()

    model_name = args.model_name
    subset_file = args.subset_csv_path
    save_dir = args.save_models
    cuda_num = args.cuda_num
    num_workers = args.num_workers
    dim_scale_factor = args.dim_scale_factor
    
    # model_name = "teacher_resnet50_0-0.1_sub_set3_6_conn_sig"
    print("Name: {}".format(model_name))
    print("Subset File: {}".format(subset_file))
    print("Save Dir: {}".format(save_dir))
    print("Cuda Num: {}".format(cuda_num), flush=True)
    print("Dimension Reducing: {}".format(dim_scale_factor), flush=True)


    preprocessor = Preprocessor((32,32))

    model = ResNet.resnet18(num_classes=10, include_top=True, inplanes=64//dim_scale_factor)
    # print(model)
    # summary(model, (3, 32, 32), device='cpu')


    cifar_train = unpickle("../../dataset/cifar-10-python/train/train_batch")
    cifar_test = unpickle("../../dataset/cifar-10-python/test/test_batch")

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

    low_mem_test_names = [
        # "train_set_0-{}".format(max_mem) for max_mem in [0.1, 0.2, 0.4, 0.6, 0.8]
        "train_set_0-{}".format(max_mem) for max_mem in [0.1]
    ]
    print(["../../dataset/scratch_fullsets/subset_0-{}.csv".format(max_mem) for max_mem in [0.1]])
    low_mem_part_train_sets = [
        LabeledPickleDatasetMapper(
            cifar_train['data'].copy(),
            cifar_train['labels'].copy(),
            "../../dataset/scratch_fullsets/subset_0-{}.csv".format(max_mem),
            preprocessor,
            augment=False
        ) for max_mem in [0.1]
    ]
    
    score_sets_low_mem = [
        {"name": low_mem_test_names[i], "dataset": low_mem_part_train_sets[i]} for i in range(len(low_mem_test_names))
    ]

    high_mem_test_names = [
        "train_set_{}-1".format(min_mem) for min_mem in [0.1]
    ]
    print(["../../dataset/high_mem/subset_{}-1.csv".format(min_mem) for min_mem in [0.1]])
    high_mem_part_train_sets = [
        LabeledPickleDatasetMapper(
            cifar_train['data'].copy(),
            cifar_train['labels'].copy(),
            "../../dataset/high_mem/subset_{}-1.csv".format(min_mem),
            preprocessor,
            augment=False
        ) for min_mem in [0.1]
    ]

    score_sets_high_mem = [
        {"name": high_mem_test_names[i], "dataset": high_mem_part_train_sets[i]} for i in range(len(high_mem_test_names))
    ]

    test_set = LabeledPickleDatasetMapper(
        cifar_test[b'data'].copy(),
        cifar_test[b'labels'].copy(),
        None,
        preprocessor,
        augment=False,
        return_idx=False
    )

    score_sets = [
        {"name": "full_train_set", "dataset": full_train_set}, 
        *score_sets_low_mem,
        *score_sets_high_mem,
        {"name": "test_set", "dataset": test_set},
    ]

    print(score_sets)

    trainer = Pipeline(
        name=model_name + "_s1",
        model=model,
        batch_size=256,
        workers=num_workers,
        train_set=train_set,
        test_sets=score_sets,
        preprocessor=preprocessor,
        log_files_path="logs/fit/",
        cuda_num=cuda_num
    )

    num_epochs = 2
    lr = 0.4
    # step_size_func = lambda e: 1 / math.sqrt(1 + e)
    step_size_func = lambda e: ((e - num_epochs*0.15)/(num_epochs*0.15) + 1) if e <= num_epochs*0.15 else (num_epochs - e)/(num_epochs*0.85)

    loss_func_with_grad = torch.nn.CrossEntropyLoss()
    loss_func = torch.nn.functional.cross_entropy

    training_log, validation_log, grad_log1, grad_log2 = trainer.train(
        num_epochs=num_epochs,
        lr=lr,
        step_size_func=step_size_func,
        loss_func_with_grad=loss_func_with_grad,
        loss_func=loss_func,
        score_functions=SCORE_FUNCTIONS_CLASSIFICATION,
        # save_checkpoints_epoch=25,
        # save_checkpoints_path="../../saved_models/",
        validation_score_epoch=5,
    )

    save_path = trainer.save(save_dir, num_epochs)

    with open("{}_save_path.txt".format(model_name), 'w') as f:
        f.write("{}\n".format(save_path))

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
