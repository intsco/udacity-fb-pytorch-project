from collections import Counter
import numpy as np

import torch
import torchvision
import torchvision as torchvision
from torchvision.datasets.folder import find_classes, make_dataset, IMG_EXTENSIONS, default_loader


class StratifiedImageFolder(torchvision.datasets.ImageFolder):
    def __init__(self, root, transform=None, p=0.1):
        super().__init__(root, transform)
        self.p = p

        sample_classes = np.asarray([c for _, c in self.imgs])
        class_freq = Counter(sample_classes).most_common()
        class_freq = np.asarray(sorted(class_freq))[:, 1]

        self.indices = []
        for c in set(sample_classes):
            class_strat_inds = np.arange(len(sample_classes))[sample_classes == c]
            np.random.shuffle(class_strat_inds)
            n = int(np.ceil(class_freq[c] * p))
            class_strat_inds = class_strat_inds[:n]
            self.indices.append(class_strat_inds)
        self.indices = np.concatenate(self.indices)

    def __getitem__(self, idx):
        return super().__getitem__(self.indices[idx])

    def __len__(self):
        return len(self.indices)


class FlowerDataset(torch.utils.data.Dataset):

    def __init__(self, root, transform):
        self.classes, self.class_to_idx = find_classes(root)
        self.samples = make_dataset(root, self.class_to_idx, extensions=IMG_EXTENSIONS)  # path, target
        self.loader = default_loader
        self.data = []
        self.targets = []
        self.transform = transform

        for path, target in self.samples:
            img = self.loader(path)
            self.data.append(img)
            self.targets.append(target)

    def __getitem__(self, index):
        img = self.transform(self.data[index])
        return img, self.targets[index]

    def __len__(self):
        return len(self.data)


class TrainValidListDataset(torch.utils.data.Dataset):

    def __init__(self, root_path, train_dir, valid_dir):
        self.classes, self.class_to_idx = find_classes(root_path / 'train')
        train_samples = make_dataset(root_path / 'train', self.class_to_idx, extensions=IMG_EXTENSIONS)
        valid_samples = make_dataset(root_path / 'valid', self.class_to_idx, extensions=IMG_EXTENSIONS)
        self.samples = train_samples + valid_samples

    def __getitem__(self, index):
        return self.samples[index]

    def __len__(self):
        return len(self.samples)


class SubsetDataset(object):

    def __init__(self, dataset, inds, transform):
        self.dataset = dataset
        self.inds = inds
        self.transform = transform
        self.loader = default_loader
        self.classes = self.dataset.classes
        self.class_to_idx = self.dataset.class_to_idx

    def __getitem__(self, index):
        path, target = self.dataset[self.inds[index]]
        sample = self.loader(path)
        sample = self.transform(sample)
        return sample, target

    def __len__(self):
        return len(self.inds)


# def load_data(batch_size=64, subset_p=None, tta=True):
#     image_transforms = {
#         'train': train_transform
#     }
#     if tta:
#         image_transforms['valid'] = valid_tranform_tta
#     else:
#         image_transforms['valid'] = valid_tranform
#
#     image_datasets = {}
#     dataloaders = {}
#
#     # train set
#     if subset_p:
#         image_datasets['train'] = StratifiedImageFolder(root=str(data_path / 'train'),
#                                                         transform=image_transforms['train'], p=subset_p)
#         dataloaders['train'] = torch.utils.data.DataLoader(image_datasets['train'], batch_size=batch_size,
#                                                            shuffle=True, num_workers=4)
#     else:
#         image_datasets['train'] = torchvision.datasets.ImageFolder(root=str(data_path / 'train'),
#                                                                    transform=image_transforms['train'])
#         dataloaders['train'] = torch.utils.data.DataLoader(image_datasets['train'], batch_size=batch_size,
#                                                            shuffle=True, num_workers=4)
#     # valid set
#     image_datasets['valid'] = torchvision.datasets.ImageFolder(root=str(data_path / 'valid'),
#                                                                transform=image_transforms['valid'])
#     dataloaders['valid'] = torch.utils.data.DataLoader(image_datasets['valid'], batch_size=batch_size,
#                                                        shuffle=False, num_workers=4)
#
#     # train-valid set
#     image_datasets['train-valid'] = torch.utils.data.ConcatDataset([image_datasets['train'],
#                                                                     image_datasets['valid']])
#     dataloaders['train-valid'] = torch.utils.data.DataLoader(image_datasets['train-valid'],
#                                                              batch_size=batch_size,
#                                                              shuffle=True, num_workers=4)
#
#     # test set
#     if (data_path / 'test').exists():
#         image_datasets['test'] = datasets.ImageFolder(root=str(data_path / 'test'),
#                                                       transform=image_transforms['valid'])
#         dataloaders['test'] = torch.utils.data.DataLoader(image_datasets['test'], batch_size=batch_size,
#                                                           shuffle=False, num_workers=4)
#     return dataloaders

