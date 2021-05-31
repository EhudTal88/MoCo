from torchvision.datasets.utils import download_url
import os
import tarfile
import hashlib
import torchvision
import torch
from torchvision.transforms import transforms
# This is a fix to overcome OS permissions for downloading models:
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Main data module:
def DataModule(batch_size,ks,imagenet_stats):

    # https://github.com/fastai/imagenette
    # Define parameters for dataset construction:
    dataset_url = 'https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz'
    dataset_filename = dataset_url.split('/')[-1]
    dataset_foldername = dataset_filename.split('.')[0]
    data_path = './data'
    dataset_filepath = os.path.join(data_path, dataset_filename)
    dataset_folderpath = os.path.join(data_path, dataset_foldername)

    os.makedirs(data_path, exist_ok=True) # Create a data folder (@ project folder)

    # If data does not exist, download it from specified URL:
    download = False
    if not os.path.exists(dataset_filepath):
        download = True
    else:
        md5_hash = hashlib.md5()

        file = open(dataset_filepath, "rb")

        content = file.read()

        md5_hash.update(content)

        digest = md5_hash.hexdigest()
        if digest != 'fe2fc210e6bb7c5664d602c3cd71e612':
            download = True
    if download:
        download_url(dataset_url, data_path)
    # Extract tar file containing dataset examples
    with tarfile.open(dataset_filepath, 'r:gz') as tar:
        tar.extractall(path=data_path)

    # Define model-input transforms for data augmentation:
    train_transform = TwoCropsTransform(transforms.Compose([transforms.RandomResizedCrop(scale=(0.2, 1), size=224),
                                                            transforms.RandomHorizontalFlip(),
                                                            transforms.RandomApply(
                                                                [transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
                                                            transforms.RandomGrayscale(p=0.2),
                                                            # transforms.GaussianBlur(kernel_size=ks),
                                                            transforms.ToTensor(),
                                                            transforms.Normalize(**imagenet_stats)]))

    # Define train and val dataset wrappers:
    dataset_train = torchvision.datasets.ImageFolder(os.path.join(dataset_folderpath, 'train'), train_transform)
    dataset_validation = torchvision.datasets.ImageFolder(os.path.join(dataset_folderpath, 'val'), train_transform)
    # Define train and val dataloaders:
    train_dataloader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=batch_size,
        num_workers=8,
        drop_last=True,
        shuffle=True,
    )
    validation_dataloader = torch.utils.data.DataLoader(
        dataset_validation,
        batch_size=batch_size,
        num_workers=8,
        drop_last=True,
        shuffle=True,
    )
    return train_dataloader,validation_dataloader,transforms

"""For MoCo scheme: Create a positive example out of a given query example"""
class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]

    def __repr__(self):
        format_string = self.__class__.__name__ + '(\n\t'
        format_string += self.base_transform.__repr__().replace('\n', '\n\t')
        format_string += '\n)'
        return format_string