# PyTorch
import torch
from torch.utils.data import IterableDataset, DataLoader
from donkeycar.utils import train_test_split
from donkeycar.parts.tub_v2 import Tub
from torchvision import transforms
from typing import List, Any
from donkeycar.pipeline.types import TubRecord, TubDataset
from donkeycar.pipeline.sequence import TubSequence
import pytorch_lightning as pl
import albumentations as A

from PIL import Image
import os
import numpy as np

def get_default_transform(for_video=False, for_inference=False, resize=True, crop=True, enhance_contrast=False):
    """
    Creates a default transform to work with torchvision models

    Video transform:
    All pre-trained models expect input images normalized in the same way, 
    i.e. mini-batches of 3-channel RGB videos of shape (3 x T x H x W), 
    where H and W are expected to be 112, and T is a number of video frames 
    in a clip. The images have to be loaded in to a range of [0, 1] and 
    then normalized using mean = [0.43216, 0.394666, 0.37645] and 
    std = [0.22803, 0.22145, 0.216989].

    If crop crops 35 pixel from the top of the image and keep the rest the same
    
    Args:
        enhance_contrast: Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
                         Helps with varying lighting conditions at inference time
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    input_size = (224, 224)

    if for_video:
        mean = [0.43216, 0.394666, 0.37645]
        std = [0.22803, 0.22145, 0.216989]
        input_size = (112, 112)

    transform_items = [
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ]
    
    # Add a wrapper to handle numpy arrays - converts to PIL if needed
    def numpy_to_pil_wrapper(img):
        if isinstance(img, np.ndarray):
            return Image.fromarray(img)
        return img
    
    # Handle different crop/resize combinations
    if crop and not resize:
        # Crop then resize back to original size
        def crop_and_restore(img):
            original_size = img.size  # (width, height)
            cropped = img.crop((0, 35, img.width, img.height))
            return cropped.resize(original_size, Image.BILINEAR)
        transform_items.insert(0, transforms.Lambda(crop_and_restore))
    elif crop and resize:
        # Crop then resize to target size
        transform_items.insert(0, transforms.Resize(input_size))
        transform_items.insert(0, transforms.Lambda(lambda img: img.crop((0, 30, img.width, img.height))))
    elif resize:
        # Just resize to target size
        transform_items.insert(0, transforms.Resize(input_size))
    
    # Albumentations wrapper for processing
    class AlbumentationsTransform:
        def __init__(self, transform):
            self.transform = transform
        
        def __call__(self, img):
            # Convert PIL Image to numpy array
            img_np = np.array(img)
            # Apply albumentations
            augmented = self.transform(image=img_np)
            # Convert back to PIL Image
            return Image.fromarray(augmented['image'])

    # Add contrast enhancement for inference (helps with varying lighting)
    if enhance_contrast:
        from PIL import ImageOps
        
        # Combine CLAHE with PIL's autocontrast
        clahe = A.Compose([
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
        ])
        transform_items.insert(0, transforms.Lambda(lambda img: ImageOps.autocontrast(img)))
        # transform_items.insert(0, AlbumentationsTransform(clahe))

    if not for_inference:
        # Add data augmentation for training
        augmentation = A.Compose([
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
            # A.RandomGamma(gamma_limit=(80, 120), p=0.2),
            A.Blur(blur_limit=3, p=0.2),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.3),
            A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=0.1),
            A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), p=0.2),
            A.GaussianBlur(blur_limit=(3, 5), p=0.1),
        ])
        
        # Add augmentation before ToTensor and Normalize
        transform_items.insert(0, AlbumentationsTransform(augmentation))
    
    # Insert numpy-to-PIL converter at the very beginning
    transform_items.insert(0, transforms.Lambda(numpy_to_pil_wrapper))

    return transforms.Compose(transform_items)


class TorchTubDataset(IterableDataset):
    '''
    Loads the dataset, and creates a train/test split.
    '''

    def __init__(self, config, records: List[TubRecord], transform=None):
        """Create a PyTorch Tub Dataset

        Args:
            config (object): the configuration information
            records (List[TubRecord]): a list of tub records
            transform (function, optional): a transform to apply to the data
        """
        self.config = config

        # Handle the transforms
        if transform:
            self.transform = transform
        else:
            self.transform = get_default_transform()

        self.sequence = TubSequence(records)
        self.pipeline = self._create_pipeline()
        self.len = len(records)

    def _create_pipeline(self):
        """ This can be overridden if more complicated pipelines are
            required """

        def y_transform(record: TubRecord):
            angle: float = record.underlying['user/angle']
            throttle: float = record.underlying['user/throttle']
            predictions = torch.tensor([angle, throttle], dtype=torch.float)

            # Normalize to be between [0, 1]
            # angle and throttle are originally between [-1, 1]
            predictions = (predictions + 1) / 2
            return predictions

        def x_transform(record: TubRecord):
            # Loads the result of Image.open()
            img_arr = record.image(as_nparray=False)

            #save the augmented image to a file for debugging

            folder = 'test_images'
            #get the highest index in the folder
            os.makedirs(folder, exist_ok=True)
            #get the highest index of the images in the folder
            existing_images = [f for f in os.listdir(folder) if f.endswith('.jpg')]
            if existing_images:
                latest_index = max(int(f.split('.')[0]) for f in existing_images)
            else:
                latest_index = -1
            next_index = latest_index + 1

            if next_index < 50:
                transformed = self.transform(img_arr)
                # Denormalize and convert tensor back to PIL Image
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                denormalized = transformed * std + mean
                # Clamp to [0, 1] and convert to [0, 255]
                denormalized = torch.clamp(denormalized, 0, 1)
                # Convert from CHW to HWC and to numpy
                img_numpy = (denormalized.permute(1, 2, 0).numpy() * 255).astype('uint8')
                Image.fromarray(img_numpy).save(os.path.join(folder, f'{next_index}.jpg'))
                return transformed
            return self.transform(img_arr)

        # Build pipeline using the transformations
        pipeline = self.sequence.build_pipeline(x_transform=x_transform,
                                                y_transform=y_transform)
        return pipeline

    def __len__(self):
        return len(self.sequence)

    def __iter__(self):
        return iter(self.pipeline)


class TorchTubDatasetWithSurface(IterableDataset):
    '''
    TorchTubDataset that includes surface_id for multi-weather models.
    '''

    def __init__(self, config, records: List[TubRecord], transform=None):
        """Create a PyTorch Tub Dataset with surface_id support

        Args:
            config (object): the configuration information
            records (List[TubRecord]): a list of tub records
            transform (function, optional): a transform to apply to the data
        """
        self.config = config

        # Handle the transforms
        if transform:
            self.transform = transform
        else:
            self.transform = get_default_transform()

        self.sequence = TubSequence(records)
        self.pipeline = self._create_pipeline()
        self.len = len(records)

    def _create_pipeline(self):
        """ Pipeline that includes surface_id """

        def y_transform(record: TubRecord):
            angle: float = record.underlying['user/angle']
            throttle: float = record.underlying['user/throttle']
            surface_id: int = record.underlying.get('surface_id', 0)
            
            predictions = torch.tensor([angle, throttle], dtype=torch.float)
            # Normalize to be between [0, 1]
            # angle and throttle are originally between [-1, 1]
            predictions = (predictions + 1) / 2
            
            # Return predictions and surface_id as separate items
            return predictions, torch.tensor(surface_id, dtype=torch.long)

        def x_transform(record: TubRecord):
            # Loads the result of Image.open()
            img_arr = record.image(as_nparray=False)
            return self.transform(img_arr)

        # Build pipeline using the transformations
        pipeline = self.sequence.build_pipeline(x_transform=x_transform,
                                                y_transform=y_transform)
        return pipeline

    def __len__(self):
        return len(self.sequence)

    def __iter__(self):
        # Yield (image, (predictions, surface_id)) tuples
        for img, (predictions, surface_id) in self.pipeline:
            yield (img, surface_id), predictions


class TorchTubDataModule(pl.LightningDataModule):

    def __init__(self, config: Any, tub_paths: List[str], transform=None):
        """Create a PyTorch Lightning Data Module to contain all data loading logic

        Args:
            config (object): the configuration information
            tub_paths (List[str]): a list of paths to the tubs to use (minimum size of 1).
                                   Each tub path corresponds to another training run.
            transform (function, optional): a transform to apply to the data
        """
        super().__init__()

        self.config = config
        self.tub_paths = tub_paths

        # Handle the transforms
        if transform:
            self.transform = transform
        else:
            self.transform = get_default_transform()

        self.tubs: List[Tub] = [Tub(tub_path, read_only=True)
                                for tub_path in self.tub_paths]
        self.records: List[TubRecord] = []

    def setup(self, stage=None):
        """Load all the tub data and set up the datasets.

        Args:
            stage ([string], optional): setup expects a string arg stage. 
                                        It is used to separate setup logic for trainer.fit 
                                        and trainer.test. Defaults to None.
        """
        # Loop through all the different tubs and load all the records for each of them
        for tub in self.tubs:
            for underlying in tub:
                record = TubRecord(self.config, tub.base_path,
                                   underlying=underlying)
                self.records.append(record)

        train_records, val_records = train_test_split(
            self.records, test_size=(1. - self.config.TRAIN_TEST_SPLIT))

        assert len(val_records) > 0, "Not enough validation data. Add more data"

        self.train_dataset = TorchTubDataset(
            self.config, train_records, transform=self.transform)
        self.val_dataset = TorchTubDataset(
            self.config, val_records, transform=self.transform)

    def train_dataloader(self):
        # The number of workers are set to 0 to avoid errors on Macs and Windows
        # See: https://github.com/rusty1s/pytorch_geometric/issues/366#issuecomment-498022534
        return DataLoader(self.train_dataset, batch_size=self.config.BATCH_SIZE, num_workers=0)

    def val_dataloader(self):
        # The number of workers are set to 0 to avoid errors on Macs and Windows
        # See: https://github.com/rusty1s/pytorch_geometric/issues/366#issuecomment-498022534
        return DataLoader(self.val_dataset, batch_size=self.config.BATCH_SIZE, num_workers=0)
