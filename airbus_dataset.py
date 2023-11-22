import os
import gc
import cv2
import numpy as np
import pandas as pd
import albumentations as A
from tensorflow.keras.utils import Sequence
from albumentations.core.composition import OneOf


class DataGenerator(Sequence):
    """ Data generator that loads images and masks in batches """
    def __init__(self, dataframe, image_folder, batch_size, augment=False):
        self.dataframe = dataframe
        self.image_folder = image_folder
        self.batch_size = batch_size
        self.augment = augment
        self.on_epoch_end()
        self.resized_width, self.resized_height = 512, 512

    def __len__(self):
        return int(np.ceil(len(self.dataframe) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_samples = self.dataframe.iloc[idx * self.batch_size:(idx + 1) * self.batch_size]

        images = []
        masks = []

        augmentation = self._get_augmentation() if self.augment else None

        for _, row in batch_samples.iterrows():
            image, mask = self._load_image_and_mask(row)
            if augmentation:
                augmented = augmentation(image=image, mask=mask)
                image, mask = augmented['image'], augmented['mask']
            images.append(image)
            masks.append(mask)

        # Freeing up resources and collecting garbage
        gc.collect()

        return np.array(images), np.array(masks)

    def _load_image_and_mask(self, row):
        image_path = os.path.join(self.image_folder, row['ImageId'])
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Unable to load image at path {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.resized_width, self.resized_height))
        image = image.astype(np.float32) / 255.0
        mask_rle = row['EncodedPixels']
        mask = rle_decode(mask_rle) if not pd.isna(mask_rle) else np.zeros((768, 768), dtype=np.float32)
        mask = cv2.resize(mask, (self.resized_width, self.resized_height))[:, :, np.newaxis]
        mask = mask.astype(np.float32)

        return image, mask

    @staticmethod
    def _get_augmentation():
        return A.Compose([
            OneOf([  # rotate
                A.Rotate(limit=0, p=0.25),
                A.Rotate(limit=90, p=0.25),
                A.Rotate(limit=180, p=0.25),
                A.Rotate(limit=270, p=0.25),
            ]),
            A.Flip(p=0.5),
            OneOf([  # brightness or contrast
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            ]),
            OneOf([  # blur or sharpen
                A.GaussianBlur(blur_limit=(3, 5), p=0.5),
                A.Sharpen(alpha=(0, 0.1), p=0.5),
            ]),
        ], additional_targets={'mask': 'image'})

    def on_epoch_end(self):
        self.dataframe = self.dataframe.sample(frac=1).reset_index(drop=True)


def rle_decode(mask_rle, shape=(768, 768)):
    """ Function for decoding RLE masks """
    if mask_rle is np.nan:
        return np.zeros(shape, dtype=np.uint8)
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T
