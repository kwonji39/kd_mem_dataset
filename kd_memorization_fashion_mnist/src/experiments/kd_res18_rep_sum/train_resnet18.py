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
from temp.eval_model import model_t
from temp.eval_model import model_s
# from pipelines.dropping_classification_pipeline import Pipeline as PipelineS1
# from pipelines.classification_pipeline import Pipeline as PipelineS2
from pipelines.classification_pipeline import Pipeline

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

    args = parser.parse_args()

    model_name = args.model_name
    # subset_file = args.subset_csv_path
    save_dir = args.save_models
    cuda_num = args.cuda_num
    num_workers = args.num_workers
    dim_scale_factor = args.dim_scale_factor
    
    # model_name = "teacher_resnet50_0-0.1_sub_set3_6_conn_sig"
    print("Name: {}".format(model_name))
    # print("Subset File: {}".format(subset_file))
    print("Save Dir: {}".format(save_dir))
    print("Cuda Num: {}".format(cuda_num), flush=True)
    print("Dimension Reducing: {}".format(dim_scale_factor), flush=True)


    # data = pd.read_csv(os.path.join(PATH_PREFIX, DATASET_TRAIN_DATA_FILE))
    # subset_file = "../../../dataset/scratch_res50_fullsets/subset_0-0.1.csv"
    # subset_file = pd.read_csv()

    # preprocessor = Preprocessor((32,32))

    # model = ResNet.resnet18(num_classes=100, include_top=True, inplanes=64//dim_scale_factor)
    # print(model)
    # summary(model, (3, 32, 32), device='cpu')

    # # # Creating dataset mapper instances
    # # train_set = LabeledDatasetMapper(
    # #     os.path.join(PATH_PREFIX, DATASET_ROOT),
    # #     os.path.join(PATH_PREFIX, DATASET_TRAIN_DATA_FILE),
    # #     preprocessor,
    # #     augment=True,
    # # )
    # # test_set = LabeledDatasetMapper(
    # #     os.path.join(PATH_PREFIX, DATASET_ROOT),
    # #     os.path.join(PATH_PREFIX, DATASET_TEST_DATA_FILE),
    # #     preprocessor,
    # #     augment=False,
    # # )

    # with open("../../../dataset/cifar-100-python/train", 'rb') as fo:
    #     cifar_train = pickle.load(fo, encoding='bytes')

    # with open("../../../dataset/cifar-100-python/test", 'rb') as fo:
    #     cifar_test = pickle.load(fo, encoding='bytes')

    dataset_base_path = "/scratch/gilbreth/kwon165/kd_memorization/src/dataset"
    train_images_path = os.path.join(dataset_base_path, "train-images-idx3-ubyte")
    train_labels_path = os.path.join(dataset_base_path, "train-labels-idx1-ubyte")
    test_images_path = os.path.join(dataset_base_path, "t10k-images-idx3-ubyte")
    test_labels_path = os.path.join(dataset_base_path, "t10k-labels-idx1-ubyte")

    preprocessor = Preprocessor((28, 28))

    model = ResNet.resnet50(num_classes=10, include_top=True)

    train_images = idx2numpy.convert_from_file(train_images_path)
    train_labels = idx2numpy.convert_from_file(train_labels_path)
    test_images = idx2numpy.convert_from_file(test_images_path)
    test_labels = idx2numpy.convert_from_file(test_labels_path)

    # (N, H, W) to (N, C, H, W)
    train_images = np.expand_dims(train_images, axis=1)
    test_images = np.expand_dims(test_images, axis=1)

    # train_set = LabeledPickleDatasetMapper(
    #     cifar_train[b'data'].copy(),
    #     cifar_train[b'fine_labels'].copy(),
    #     subset_file,
    #     preprocessor,
    #     augment=True,
    #     return_idx=True
    # )

    # full_train_set = LabeledPickleDatasetMapper(
    #     cifar_train[b'data'].copy(),
    #     cifar_train[b'fine_labels'].copy(),
    #     None,
    #     preprocessor,
    #     augment=False
    # )

    # low_mem_test_names = [
    #     "train_set_0-{}".format(max_mem) for max_mem in [0.1, 0.2, 0.4, 0.6, 0.8]
    # ]
    # print(["../../../dataset/scratch_fullsets/subset_0-{}.csv".format(max_mem) for max_mem in [0.1, 0.2, 0.4, 0.6, 0.8]])
    # low_mem_part_train_sets = [
    #     LabeledPickleDatasetMapper(
    #         cifar_train[b'data'].copy(),
    #         cifar_train[b'fine_labels'].copy(),
    #         "../../../dataset/scratch_fullsets/subset_0-{}.csv".format(max_mem),
    #         preprocessor,
    #         augment=False
    #     ) for max_mem in [0.1, 0.2, 0.4, 0.6, 0.8]
    # ]
    
    # score_sets_low_mem = [
    #     {"name": low_mem_test_names[i], "dataset": low_mem_part_train_sets[i]} for i in range(len(low_mem_test_names))
    # ]

    # high_mem_test_names = [
    #     "train_set_{}-1".format(min_mem) for min_mem in [0.1, 0.2, 0.4, 0.6, 0.8]
    # ]
    # print(["../../../dataset/high_mem/subset_{}-1.csv".format(min_mem) for min_mem in [0.1, 0.2, 0.4, 0.6, 0.8]])
    # high_mem_part_train_sets = [
    #     LabeledPickleDatasetMapper(
    #         cifar_train[b'data'].copy(),
    #         cifar_train[b'fine_labels'].copy(),
    #         "../../../dataset/high_mem/subset_{}-1.csv".format(min_mem),
    #         preprocessor,
    #         augment=False
    #     ) for min_mem in [0.1, 0.2, 0.4, 0.6, 0.8]
    # ]

    # score_sets_high_mem = [
    #     {"name": high_mem_test_names[i], "dataset": high_mem_part_train_sets[i]} for i in range(len(high_mem_test_names))
    # ]

    # test_set = LabeledPickleDatasetMapper(
    #     cifar_test[b'data'].copy(),
    #     cifar_test[b'fine_labels'].copy(),
    #     None,
    #     preprocessor,
    #     augment=False
    # )

    # score_sets = [
    #     {"name": "full_train_set", "dataset": full_train_set}, 
    #     *score_sets_low_mem,
    #     *score_sets_high_mem,
    #     {"name": "test_set", "dataset": test_set},
    # ]

    # print(score_sets)

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

    # trainer = Pipeline(
    #     name=model_name + "_s1",
    #     model=model,
    #     batch_size=256,
    #     workers=num_workers,
    #     train_set=train_set,
    #     test_sets=score_sets,
    #     preprocessor=preprocessor,
    #     log_files_path="logs/fit/",
    #     cuda_num=cuda_num
    # )

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
        score_functions=SCORE_FUNCTIONS_CLASSIFICATION
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