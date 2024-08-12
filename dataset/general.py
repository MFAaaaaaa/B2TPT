import torch
import torchvision
import os
import numpy as np
from torchvision.datasets import CIFAR100
from torch.utils.data import Dataset, DataLoader
import argparse
from dataset.utils import Util
import pickle
from PIL import Image
from torchvision import transforms, datasets
from randaugment import RandAugmentMC


class TrainDataset(Dataset):
    def __init__(self, args):
        self.prepocess = args["preprocess"]
        self.root = args["root"]
        self.dataset_dir = args["dataset_dir"]
        self.seed = args["seed"]
        self.dataset_dir = os.path.join(self.root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "image_data")
        self.split_path = os.path.join(self.dataset_dir, "split.json")
        self.all_data = Util.read_split(self.split_path)
        self.all_train = self.all_data["train"]

        # -------------------------------------------------------------------------------------------
        self.train_data_dir = os.path.join(self.dataset_dir, "train_data")
        Util.mkdir_if_missing(self.train_data_dir)
        preprocessed = os.path.join(self.train_data_dir, f"train.pkl")
        if os.path.exists(preprocessed):
            print(f"Loading preprocessed train data from {preprocessed}")
            with open(preprocessed, "rb") as file:
                content = pickle.load(file)
                self.all_train = content["all_train"]
                self.classes = content["classes"]
        else:
            for tmp in self.all_train:
                image_path = os.path.join(self.image_dir, tmp[0])
                tmp[0] = self.prepocess(Image.open(image_path))
            self.classes = self.construct_data()
            print(f"Saving preprocessed train data to {preprocessed}")
            content = {"all_train": self.all_train, "classes": self.classes}
            with open(preprocessed, "wb") as file:
                pickle.dump(content, file, protocol=pickle.HIGHEST_PROTOCOL)

    # -----------------------------------------------------------------------------------------------

    def __len__(self):
        return len(self.all_train)

    def __getitem__(self, idx):
        return {"image": self.all_train[idx][0], "label": self.all_train[idx][1]}

    def construct_data(self):
        train_shot_count = {}
        classes_dict = {}
        all_indices = [_ for _ in range(len(self.all_train))]
        np.random.seed(self.seed)
        np.random.shuffle(all_indices)

        for index in all_indices:
            label, classname = self.all_train[index][1], self.all_train[index][2]

            if label not in train_shot_count:
                train_shot_count[label] = 0
                classes_dict[label] = classname
                train_shot_count[label] += 1
        classes = [classes_dict[i] for i in range(len(classes_dict))]
        return classes


def load_train(batch_size=1, seed=42, preprocess=None, root=None, dataset_dir=None):
    args = {"preprocess": preprocess, "root": root, "dataset_dir": dataset_dir, "seed": seed}
    train_data = TrainDataset(args)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
    return train_data, train_loader


class TestDataset(Dataset):
    def __init__(self, args):
        self.prepocess = args["preprocess"]
        self.root = args["root"]
        self.dataset_dir = args["dataset_dir"]
        self.dataset_dir = os.path.join(self.root, self.dataset_dir)
        self.seed = args["seed"]
        self.image_dir = os.path.join(self.dataset_dir, "image_data")
        self.split_path = os.path.join(self.dataset_dir, "split.json")
        self.all_data = Util.read_split(self.split_path)
        self.all_test = self.all_data["test"]
        # -------------------------------------------------------------------------------------------
        self.test_data_dir = os.path.join(self.dataset_dir, "test_data")
        Util.mkdir_if_missing(self.test_data_dir)
        preprocessed = os.path.join(self.test_data_dir, f"test.pkl")
        if os.path.exists(preprocessed):
            print(f"Loading preprocessed test data from {preprocessed}")
            with open(preprocessed, "rb") as file:
                content = pickle.load(file)
                self.all_test = content["all_test"]
                self.classes = content["classes"]
        else:
            for tmp in self.all_test:
                image_path = os.path.join(self.image_dir, tmp[0])
                tmp[0] = self.prepocess(Image.open(image_path))
            self.classes = self.construct_data()
            print(f"Saving preprocessed test data to {preprocessed}")
            # content = self.all_test
            content = {"all_test": self.all_test, "classes": self.classes}
            with open(preprocessed, "wb") as file:
                pickle.dump(content, file, protocol=pickle.HIGHEST_PROTOCOL)

    def construct_data(self):
        train_shot_count = {}
        classes_dict = {}
        all_indices = [_ for _ in range(len(self.all_test))]
        np.random.seed(self.seed)
        np.random.shuffle(all_indices)

        for index in all_indices:
            label, classname = self.all_test[index][1], self.all_test[index][2]

            if label not in train_shot_count:
                train_shot_count[label] = 0
                classes_dict[label] = classname
                train_shot_count[label] += 1
        classes = [classes_dict[i] for i in range(len(classes_dict))]
        return classes

    def __len__(self):
        return len(self.all_test)

    def __getitem__(self, idx):

        return {"image": self.all_test[idx][0], "label": self.all_test[idx][1]}


def load_test(batch_size=1, seed=42, preprocess=None, root=None, dataset_dir=None):
    args = {"preprocess": preprocess, "root": root, "dataset_dir": dataset_dir, "seed": seed}
    test_data = TestDataset(args)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=4)
    return test_data, test_loader



class FixDataset(Dataset):
    def __init__(self, args, transform):
        self.transform = transform
        self.prepocess = args["preprocess"]
        self.root = args["root"]
        self.dataset_dir = args["dataset_dir"]
        self.dataset_dir = os.path.join(self.root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "image_data")
        self.split_path = os.path.join(self.dataset_dir, "split.json")
        self.all_data = Util.read_split(self.split_path)
        self.all_train = self.all_data["train"]

        # -------------------------------------------------------------------------------------------
        self.train_data_dir = os.path.join(self.dataset_dir, "fix_data")
        Util.mkdir_if_missing(self.train_data_dir)
        preprocessed = os.path.join(self.train_data_dir, f"fix.pkl")
        if os.path.exists(preprocessed):
            print(f"Loading preprocessed train data from {preprocessed}")
            with open(preprocessed, "rb") as file:
                content = pickle.load(file)
                self.all_train = content
        else:
            for tmp in self.all_train:
                image_path = os.path.join(self.image_dir, tmp[0])
                # print(Image.open(image_path).size)
                # print(Image.open(image_path).mode)

                tmp[0] = self.transform(Image.open(image_path).convert("RGB"))
                # tmp[0] = self.transform(Image.open(image_path))

            print(f"Saving preprocessed train data to {preprocessed}")
            content = self.all_train
            with open(preprocessed, "wb") as file:
                pickle.dump(content, file, protocol=pickle.HIGHEST_PROTOCOL)

    # -----------------------------------------------------------------------------------------------

    def __len__(self):
        return len(self.all_train)

    def __getitem__(self, idx):
        return {"image": self.all_train[idx][0], "label": self.all_train[idx][1]}


def load_fix(batch_size=1, preprocess=None, root=None, dataset_dir=None):
    args = {"preprocess": preprocess, "root": root, "dataset_dir": dataset_dir}
    fixmatch_transform = TransformFixMatch()
    fix_data = FixDataset(args, transform=fixmatch_transform)

    fix_data_loader = DataLoader(fix_data, batch_size=batch_size, shuffle=True, num_workers=4)
    return fix_data_loader


class TransformFixMatch:
    def __init__(self):
        self.weak = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(size=224),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.125)])
        self.strong = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(size=224),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.125),
            RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)

        return self.normalize(weak), self.normalize(strong)

