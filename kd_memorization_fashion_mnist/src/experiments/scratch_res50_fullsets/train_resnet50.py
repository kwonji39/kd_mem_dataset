import sys
sys.path.append("../../")


import numpy as np
import torch
import os
import argparse
import idx2numpy
import pickle

import models.resnet as ResNet

from utils.data_mappers import LabeledPickleDatasetMapper
from utils.preprocessing import Preprocessor
from pipelines.classification_pipeline import Pipeline

from utils.constants import SCORE_FUNCTIONS_CLASSIFICATION

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="train_resnet50", description="Training ResNet50 from scratch on Fashion-MNIST")
    parser.add_argument("model_name", type=str, help='Model Name (without spaces)')
    # parser.add_argument("subset_csv_path", type=str, help='Path to subset csv (set to None if not applicable)')
    parser.add_argument("save_models", type=str, help='Path to directory to save models')
    parser.add_argument("--cuda-num", dest="cuda_num", type=int, default=None, help="Device number for CUDA, if applicable")

    args = parser.parse_args()

    model_name = args.model_name
    # subset_file = args.subset_csv_path if args.subset_csv_path != 'None' else None
    save_dir = args.save_models
    cuda_num = args.cuda_num

    print("Name: {}".format(model_name))
    # print("Subset File: {}".format(subset_file))
    print("Save Dir: {}".format(save_dir))
    print("Cuda Num: {}".format(cuda_num))

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

    trainer = Pipeline(
        name=model_name,
        model=model,
        batch_size=256,
        workers=8,
        train_set=train_set,
        test_sets=[{"name": "test_set", "dataset": test_set}],
        preprocessor=preprocessor,
        log_files_path="logs/fit/",
        cuda_num=cuda_num
    )

    num_epochs = 200
    lr = 0.01
    step_size_func = lambda e: 0.95 ** e

    loss_func_with_grad = torch.nn.CrossEntropyLoss()
    loss_func = torch.nn.functional.cross_entropy

    training_log, validation_log, _, _= trainer.train(
        num_epochs=num_epochs,
        lr=lr,
        step_size_func=step_size_func,
        loss_func_with_grad=loss_func_with_grad,
        loss_func=loss_func,
        score_functions=SCORE_FUNCTIONS_CLASSIFICATION
    )

    trainer.save(save_dir, num_epochs)

    os.makedirs(os.path.join("logs/logfiles", trainer.name), exist_ok=True)
    with open(os.path.join("logs/logfiles", trainer.name, "training_log.pkl"), "wb") as f:
        pickle.dump(training_log, f)

    with open(os.path.join("logs/logfiles", trainer.name, "validation_log.pkl"), "wb") as f:
        pickle.dump(validation_log, f)
