import os
import time
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
from tqdm import tqdm

os.environ[
    "CUDA_LAUNCH_BLOCKING"] = "1"  # Print out CUDA error trackback details

train_transformer = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
])

validation_transformer = transforms.Compose([
    transforms.Resize(size=(224, 224)),
])


# Custom Lungs Dataset that can be constructed into Train, Test, Validation dataset respectively, and select a Dataset (Normal-Infected or Covid-NonCovid) to use, based on the binary classifier implementation.
class LungDataset3C(Dataset):
    def __init__(self, group):

        self.img_size = (150, 150)

        self.class_names = ['normal', 'covid', 'non-covid']
        self.classes = {
            0: 'normal',
            1: 'infected_covid',
            2: 'infected_non_covid'
        }

        self.groups = [group]

        # Number of images in each part of the dataset
        self.dataset_numbers = {'train_normal': 1341,\
                                'train_infected_covid': 1345,\
                                'train_infected_non_covid': 2530,\
                                'val_normal': 8,\
                                'val_infected_covid': 9,\
                                'val_infected_non_covid': 8,\
                                'test_normal': 234,\
                                'test_infected_covid': 139,\
                                'test_infected_non_covid': 242}

    def get_dataset_path(self, _class):
        sub_path = None
        group = self.groups[0]
        if _class == self.classes[1]:
            sub_path = os.path.join("infected", "covid")
        elif _class == self.classes[2]:
            sub_path = os.path.join("infected", "non-covid")
        else:
            sub_path = "normal"
        return os.path.join("dataset", group, sub_path)

    def filter_dataset_numbers(self):
        filtered_dataset_numbers_map = dict()
        for key, value in self.dataset_numbers.items():
            if self.groups[0] in key:
                filtered_dataset_numbers_map[key] = value
        return filtered_dataset_numbers_map

    def describe(self):
        filtered_dataset_numbers_map = self.filter_dataset_numbers()
        # Generate description
        msg = "This is the Lung {} Dataset used for the Small Project Demo in the 50.039 Deep Learning class".format(
            self.groups[0].upper())
        msg += " in Feb-March 2021. \n"
        msg += "It contains a total of {} images, ".format(len(self))
        msg += "of size {} by {}.\n".format(self.img_size[0], self.img_size[1])
        msg += "Images have been split in three groups: training, testing and validation sets.\n"
        msg += "The images are stored in the following locations "
        msg += "and each one contains the following number of images:\n"
        for group in self.groups:
            for _class in self.classes.values():
                label = "{}_{}".format(group, _class)
                path = self.get_dataset_path(_class)
                msg += " - {}, in folder {}: {} images.\n".format(
                    label, path, filtered_dataset_numbers_map[label])
        print(msg)

    def open_img(self, _class, index):
        group = self.groups[0]
        if _class not in self.classes.values():
            raise ValueError(
                "Input class not found! Please input: {}. Got: {}".format(
                    list(self.classes.values()), _class))
        max_val = self.dataset_numbers['{}_{}'.format(group, _class)]
        if index < 0 or index >= max_val:
            raise ValueError(
                "Index out of range! Should be (0 ~ {}) but got {}".format(
                    max_val - 1, index))
        path_to_file = os.path.join(self.get_dataset_path(_class),
                                    "{}.jpg".format(index))
        with open(path_to_file, 'rb') as f:
            img = Image.open(f)
            if self.groups[0] == "train":
                img = train_transformer(img)
            else:
                img = validation_transformer(img)
        img = np.asarray(img) / 255  # Normalize
        f.close()
        return img

    def show_img(self, _class, index):
        # Open image
        im = self.open_img(_class, index)

        # Display
        plt.imshow(im)

    def __len__(self):
        length = 0
        for key, item in self.dataset_numbers.items():
            if self.groups[0] in key:
                length += item
        return length

    def __getitem__(self, index):
        filtered_dataset_numbers_map = self.filter_dataset_numbers()
        first_val = int(list(filtered_dataset_numbers_map.values())[0])
        second_val = int(list(filtered_dataset_numbers_map.values())[1])
        if index < first_val:
            _class = 'normal'
            label = 0
        elif first_val <= index < first_val + second_val:
            _class = 'infected_covid'
            index = index - first_val
            label = 1
        else:
            _class = 'infected_non_covid'
            index = index - first_val - second_val
            label = 2
        im = self.open_img(_class, index)
        im = transforms.functional.to_tensor(np.array(im)).float()
        return im, label


trainset_normal_infected = LungDataset3C(group="train")
testset_nomral_infected = LungDataset3C(group="test")
valset_normal_infected = LungDataset3C(group="val")

train_loader = DataLoader(trainset_normal_infected, batch_size=8, shuffle=True)
test_loader = DataLoader(testset_nomral_infected, batch_size=8, shuffle=True)
val_loader = DataLoader(valset_normal_infected, batch_size=8, shuffle=True)

print(len(train_loader.dataset), len(test_loader.dataset),
      len(val_loader.dataset))

train_loader.dataset.describe()
test_loader.dataset.describe()
val_loader.dataset.describe()

class_names = train_loader.dataset.class_names


def show_images(images, labels, preds):
    plt.figure(figsize=(16, 8))
    for i, image in enumerate(images):
        plt.subplot(1, 8, i + 1, xticks=[], yticks=[])
        image = image.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = image * std + mean
        image = np.clip(image, 0., 1.)
        plt.imshow(image)
        col = 'green'
        if preds[i] != labels[i]:
            col = 'red'
        plt.rc('axes', labelsize=14)
        plt.xlabel(f'{class_names[int(labels[i].numpy())]}')
        plt.ylabel(f'{class_names[int(preds[i].numpy())]}', color=col)
    plt.tight_layout()
    plt.show()


def show_preds(model):
    model.eval()
    images, labels = next(iter(test_loader))
    images = images.to("cuda")
    labels = labels.to("cuda")
    outputs = model(images)
    _, preds = torch.max(outputs, dim=1)
    show_images(images.cpu(), labels.cpu(), preds.cpu())


class Model(nn.Module):
    def __init__(self, dropout=0.7):
        super().__init__()

        self.dropout = dropout

        self.conv2d_1 = nn.Conv2d(in_channels=1,
                                  out_channels=64,
                                  kernel_size=3,
                                  stride=1)
        self.conv2d_2 = nn.Conv2d(in_channels=64,
                                  out_channels=64,
                                  kernel_size=3,
                                  stride=1)
        self.conv2d_3 = nn.Conv2d(in_channels=64,
                                  out_channels=128,
                                  kernel_size=3,
                                  stride=1)
        self.conv2d_4 = nn.Conv2d(in_channels=128,
                                  out_channels=128,
                                  kernel_size=3,
                                  stride=1)
        self.conv2d_5 = nn.Conv2d(in_channels=128,
                                  out_channels=256,
                                  kernel_size=3,
                                  stride=1)
        self.conv2d_6 = nn.Conv2d(in_channels=256,
                                  out_channels=256,
                                  kernel_size=3,
                                  stride=1)
        self.conv2d_7 = nn.Conv2d(in_channels=256,
                                  out_channels=512,
                                  kernel_size=3,
                                  stride=1)
        self.conv2d_8 = nn.Conv2d(in_channels=512,
                                  out_channels=512,
                                  kernel_size=3,
                                  stride=1)
        self.maxPool2d = nn.MaxPool2d(kernel_size=2, stride=2, dilation=1)
        self.linear_1 = nn.Linear(512 * 7 * 7, 4096)
        self.linear_2 = nn.Linear(4096, 4096)
        self.linear_3 = nn.Linear(4096, 1000)
        self.linear_4 = nn.Linear(1000, 10)
        self.linear_5 = nn.Linear(4096, 3)  # output layer
        self.dropout = nn.Dropout(self.dropout)
        self.adaptiveAvgPool = nn.AdaptiveAvgPool2d(7)

    def forward(self, x: torch.Tensor):
        # 1st Conv
        x = F.relu(self.conv2d_1(x))
        # x = F.relu(self.conv2d_2(x))
        x = self.maxPool2d(x)

        # 2nd Conv
        x = F.relu(self.conv2d_3(x))
        # x = F.relu(self.conv2d_4(x))
        x = self.maxPool2d(x)

        # 3rd Conv
        x = F.relu(self.conv2d_5(x))
        # x = F.relu(self.conv2d_6(x))
        # x = F.relu(self.conv2d_6(x))
        x = self.maxPool2d(x)

        # 4th Conv
        x = F.relu(self.conv2d_7(x))
        # x = F.relu(self.conv2d_8(x))
        # x = F.relu(self.conv2d_8(x))
        x = self.maxPool2d(x)

        # 5th Conv
        x = F.relu(self.conv2d_8(x))
        # x = F.relu(self.conv2d_8(x))
        # x = F.relu(self.conv2d_8(x))
        x = self.maxPool2d(x)

        x = self.adaptiveAvgPool(x)

        x = x.view(x.size(0), -1)

        # Classifier
        x = F.relu(self.linear_1(x))
        x = self.dropout(x)
        x = F.relu(self.linear_2(x))
        x = self.dropout(x)
        # x = F.relu(self.linear_3(x))
        # x = self.dropout(x)
        # x = F.relu(self.linear_4(x))
        # x = self.dropout(x)
        x = self.linear_5(x)
        return x


model = Model()

# model.to("cuda")
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-5, weight_decay=1e-5)


# To save the model
def save_model(model):
    save_file = 'model/small_project_model.pth'
    torch.save(model.state_dict(), save_file)


# Train and validate
def train(epochs):

    n_epochs = epochs

    start = time.time()

    train_loss_list = []
    validation_loss_list = []
    accuracy_list = []

    best_accuracy = 0

    for epoch in range(1, n_epochs + 1):

        train_loss = 0
        valid_loss = 0
        steps = 0

        # Training
        model.train()
        for data, target in train_loader:

            # data, target = data.to("cuda"), target.to("cuda")
            optimizer.zero_grad()
            output = model.forward(data)

            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            if steps % 20 == 0:
                accuracy = 0

                # Evaluation
                model.eval()
                for val_step, (data, target) in enumerate(test_loader):

                    # data, target = data.to("cuda"), target.to("cuda")
                    val_output = model.forward(data)
                    loss = criterion(val_output, target)
                    valid_loss += loss.item()

                    _, preds = torch.max(val_output, 1)
                    accuracy += sum((preds == target).cpu().numpy())

                valid_loss /= (val_step + 1)
                accuracy = accuracy / len(test_loader.dataset)
                print(
                    "Epoch: {:3}/{:3} Steps: {:3}/{:3} Validation Loss: {:.6f} Accuracy: {:.4f}"
                    .format(epoch, n_epochs, steps, len(train_loader),
                            valid_loss, accuracy))

                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    save_model(model)

                # show predictions plots
                # show_preds(model)

                accuracy_list.append(accuracy)
                validation_loss_list.append(valid_loss)

                model.train()

                if accuracy >= 0.98:
                    print('Performance condition satisfied, stopping..')
                    save_model(model)
                    print("Run time: {:.3f} min".format(
                        (time.time() - start) / 60))
                    return train_loss_list, validation_loss_list, accuracy_list

            train_loss /= (steps + 1)
            train_loss_list.append(train_loss)
            steps += 1

    save_model(model)
    print("Run time: {:.3f} min".format((time.time() - start) / 60))
    return train_loss_list, validation_loss_list, accuracy_list


train_loss_list, validation_loss_list, accuracy_list = train(20)