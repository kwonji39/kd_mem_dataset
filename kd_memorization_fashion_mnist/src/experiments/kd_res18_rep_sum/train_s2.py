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

from pipelines.two_students_rep_sum_classification_pipeline import PipelineS1, PipelineS2
#from temp.eval_model import model_t
#from temp.eval_model import model_s
import argparse
from functools import partial
# from torchsummary import summary

if __name__ == "__main__":
    PATH_PREFIX = "../../../"

    parser = argparse.ArgumentParser(
        prog="train_resnet18", description="Training ResNet18 from scratch"
    )
    parser.add_argument("model_name", type=str, help="Model Name (without spaces)")
    # parser.add_argument("subset_csv_path", type=str, help="Path to subset csv")
    parser.add_argument(
        "save_models", type=str, help="Path to directory to save models"
    )
    parser.add_argument("teacher_path", type=str, help="Path to saved teacher model")
    parser.add_argument("s1_path", type=str, help="Path to saved S1 model")

    parser.add_argument(
        "--soft_temp",
        dest="soft_temp",
        default=1,
        type=float,
        help="Smoothening parameter for Softmax",
    )
    parser.add_argument(
        "--cuda-num", dest="cuda_num", type=int, help="Device number for cuda"
    )
    parser.add_argument(
        "--num-workers",
        dest="num_workers",
        type=int,
        help="Number of workers for dataloader",
    )
    parser.add_argument(
        "--reduce_s1",
        dest="dim_scale_factor_s1",
        type=int,
        help="Scale down model s1 dimenstions byt factor",
    )
    parser.add_argument(
        "--reduce_s2",
        dest="dim_scale_factor_s2",
        type=int,
        help="Scale down student model dimenstions byt factor",
    )
    parser.add_argument("--bitvector_path", dest="bitvector_path", default=None, type=str, help="Path to saved S1 model's samples bitvector")

    
    args = parser.parse_args()

    model_name = args.model_name
    # subset_file = args.subset_csv_path
    save_dir = args.save_models
    teacher_path = args.teacher_path
    soft_temp = args.soft_temp
    cuda_num = args.cuda_num
    num_workers = args.num_workers
    dim_scale_factor_s1 = args.dim_scale_factor_s1
    dim_scale_factor_s2 = args.dim_scale_factor_s2
    s1_path = args.s1_path
    bitvector_path = args.bitvector_path
    
    # model_name = "teacher_resnet50_0-0.1_sub_set3_6_conn_sig"
    print("Name: {}".format(model_name))
    # print("Subset File: {}".format(subset_file))
    print("Save Dir: {}".format(save_dir))
    print("Teacher Model: {}".format(teacher_path))
    print("S1 Model: {}".format(s1_path))
    print("Softmax Temperature: {}".format(soft_temp), flush=True)
    print("Cuda Num: {}".format(cuda_num), flush=True)
    print("Dimension Reducing (S1): {}".format(dim_scale_factor_s1), flush=True)
    print("Dimension Reducing (Student): {}".format(dim_scale_factor_s2), flush=True)


    preprocessor = Preprocessor((28, 28))

    
    teacher_model = ResNet.resnet50(
        num_classes=10,
        include_top=True,
        final_activation=True,
        inplanes=64,
        temperature=soft_temp,
    )
    # print(teacher_model)
    # summary(teacher_model, (3, 32, 32), device="cpu")

    teacher_model.load_state_dict(torch.load(teacher_path)["model_state_dict"])

    for params in teacher_model.parameters():
        params.requires_grad = False

    teacher_model.eval()

    
    model_s1 = ResNet.resnet18(
        num_classes=10,
        include_top=True,
        final_activation=True,
        inplanes=64 // dim_scale_factor_s1,
        temperature=soft_temp,
    )
    # print(model_s1)
    # summary(model_s1, (3, 32, 32), device="cpu")

    print("model S1 loaded from {}".format(s1_path))
    # model_s1.conv1 = torch.nn.Conv2d(1, 256, kernel_size=7, stride=2, padding=3, bias=False)

    model_s1.load_state_dict(torch.load(s1_path)["model_state_dict"])

    for params in model_s1.parameters():
        params.requires_grad = False

    model_s1.eval()

    
    model_s2 = ResNet.resnet18(
        num_classes=10,
        include_top=True,
        final_activation=True,
        inplanes=64 // dim_scale_factor_s2,
    )
    
    # print(model_s2)
    # summary(model_s2, (3, 32, 32), device="cpu")

    dataset_base_path = "/scratch/gilbreth/kwon165/kd_memorization/src/dataset"
    train_images_path = os.path.join(dataset_base_path, "train-images-idx3-ubyte")
    train_labels_path = os.path.join(dataset_base_path, "train-labels-idx1-ubyte")
    test_images_path = os.path.join(dataset_base_path, "t10k-images-idx3-ubyte")
    test_labels_path = os.path.join(dataset_base_path, "t10k-labels-idx1-ubyte")

    train_images = idx2numpy.convert_from_file(train_images_path)
    train_labels = idx2numpy.convert_from_file(train_labels_path)
    test_images = idx2numpy.convert_from_file(test_images_path)
    test_labels = idx2numpy.convert_from_file(test_labels_path)

    # (N, H, W) to (N, C, H, W)
    train_images = np.expand_dims(train_images, axis=1)
    test_images = np.expand_dims(test_images, axis=1)

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

    """
    S2
    """
    # if s1_path is not None:
    #     model_s1.load_state_dict(torch.load(s1_path)["model_state_dict"])
    #     print("model S1 loaded from {}".format(s1_path))

    # if bitvector_path is not None:
    #     with open(bitvector_path, "rb") as f:
    #         samples_bitvector = pickle.load(f)
    #         print("Bitvector loaded from {}".format(bitvector_path))

    # for params in model_s1.parameters():
    #     params.requires_grad = False

    # model_s1.eval()


    trainer_s2 = PipelineS2(
        name=model_name,
        model_s1=model_s1,
        model_s2=model_s2,
        batch_size=256,
        workers=num_workers,
        train_set=train_set,
        test_sets=score_sets,
        preprocessor=preprocessor,
        log_files_path="logs/fit/",
        teacher_model=teacher_model,
        cuda_num=cuda_num,
    )

    num_epochs = 200
    lr = 0.4
    # step_size_func = lambda e: 1 / math.sqrt(1 + e)
    step_size_func = (
        lambda e: ((e - num_epochs * 0.15) / (num_epochs * 0.15) + 1)
        if e <= num_epochs * 0.15
        else (num_epochs - e) / (num_epochs * 0.85)
    )

    loss_func_with_grad = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)
    loss_func = partial(
        torch.nn.functional.kl_div, reduction="batchmean", log_target=True
    )

    # samples_bitvector[:] = samples_bitvector == 0

    (
        training_log,
        validation_log,
        grad_log1,
        grad_log2,
        # samples_bitvector_s2,
        _
    ) = trainer_s2.train(
        num_epochs=num_epochs,
        teacher_weightage=1,
        score_on_gnd_truth=True,
        lr=lr,
        score_functions=SCORE_FUNCTIONS_CLASSIFICATION,
        step_size_func=step_size_func,
        loss_func_with_grad=loss_func_with_grad,
        loss_func=loss_func,
        # save_checkpoints_epoch=25,
        # save_checkpoints_path="../../saved_models/",
        validation_score_epoch=5
        # samples_bitvector=samples_bitvector,
    )

    save_path = trainer_s2.save(save_dir, num_epochs)

    with open("{}_save_path.txt".format(model_name), "w") as f:
        f.write(save_path)

    os.makedirs(os.path.join("logs/logfiles", trainer_s2.name))
    with open(
        os.path.join("logs/logfiles", trainer_s2.name, "training_log.pkl"), "wb"
    ) as f:
        pickle.dump(training_log, f)

    with open(
        os.path.join("logs/logfiles", trainer_s2.name, "validation_log.pkl"), "wb"
    ) as f:
        pickle.dump(validation_log, f)

    with open(
        os.path.join("logs/logfiles", trainer_s2.name, "grad_log1.pkl"), "wb"
    ) as f:
        pickle.dump(grad_log1, f)

    with open(
        os.path.join("logs/logfiles", trainer_s2.name, "grad_log2.pkl"), "wb"
    ) as f:
        pickle.dump(grad_log2, f)