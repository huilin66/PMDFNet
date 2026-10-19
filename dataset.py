import os

import numpy as np
import tifffile
import torch
import torch.utils.data as data

from augmentations import MarsAugmentor

MARS_MEAN_TRAIN = [
    124.91780822097614,
    12.013606200679655,
    899.3700297694053,
    125.89562030094926,
    116.2159690282678,
    90.23042963499664,
    87.73630174206149,
]
MARS_STD_TRAIN = [
    38.79083746829768,
    10.732307525464076,
    3125.315890758196,
    20.716423385646948,
    43.95551091249073,
    50.56910494962908,
    55.89415768189323,
]

MARS_MEAN_VAL = [
    124.77450469045928,
    12.014302065878203,
    1264.9298021721117,
    128.31604558771306,
    119.37995771928267,
    94.09248213334517,
    91.95520851828836,
]
MARS_STD_VAL = [
    39.202188536118584,
    10.603291256055343,
    2815.767831935175,
    22.445230182955232,
    45.16733832826379,
    54.052509327647144,
    60.48449485818394,
]

MARS_MEAN_TEST = [
    124.27780954461349,
    11.63265690991753,
    1313.9822162828948,
    124.69709307448308,
    117.63051014376762,
    90.58424641315202,
    87.48808345938087,
]
MARS_STD_TEST = [
    39.243043949611845,
    10.682431370797609,
    2875.401435989946,
    23.621196943774855,
    42.39756831034838,
    49.5390059646989,
    55.06808465601418,
]

MARS_MEAN_TESTB = [
    124.66898268547611,
    9.474722544352213,
    6068.357886715212,
    128.1149415831635,
    142.42926445560178,
    112.68067202360734,
    109.11560633562613,
]
MARS_STD_TESTB = [
    35.23305609995754,
    9.931602501891595,
    1537.105845288447,
    16.972382818134335,
    37.44691286869666,
    54.85370221136559,
    63.81935657085674,
]


class MarsSegDataset(data.Dataset):
    def __init__(self, root_dir, split="train", size=128, val_ana=False):
        """
        root_dir: Dataset root directory containing train/ val/ test/ subdirectories
        split: 'train', 'val', or 'test'
        """
        self.root_dir = root_dir
        self.split = split
        self.size = size

        self.augmentor = MarsAugmentor(prob=0.5) if split == "train" else None

        if split == "train":
            mars_mean = MARS_MEAN_TRAIN
            mars_std = MARS_STD_TRAIN
        elif split == "val":
            mars_mean = MARS_MEAN_VAL
            mars_std = MARS_STD_VAL
        else:
            mars_mean = MARS_MEAN_TEST
            mars_std = MARS_STD_TEST

        self.mean = np.array(mars_mean, dtype=np.float32).reshape(7, 1, 1)
        self.std = np.array(mars_std, dtype=np.float32).reshape(7, 1, 1)

        self.images_dir = os.path.join(root_dir, split, "images")
        self.masks_dir = os.path.join(root_dir, split, "masks")

        if not os.path.exists(self.images_dir):
            if split != "test":
                print(f"Warning: Directory not found: {self.images_dir}")
            self.images = []
        else:
            self.images = sorted(
                [f for f in os.listdir(self.images_dir) if f.endswith(".tif")]
            )

        if val_ana:
            self.images_dir = os.path.join(root_dir, "val", "images")
            self.masks_dir = os.path.join(root_dir, "val", "masks")
            self.images = sorted(
                [f for f in os.listdir(self.images_dir) if f.endswith(".tif")]
            )

        if split != "test":
            if not os.path.exists(self.masks_dir):
                self.masks = []
            else:
                self.masks = sorted(
                    [f for f in os.listdir(self.masks_dir) if f.endswith(".tif")]
                )

            if (
                len(self.images) > 0
                and len(self.masks) > 0
                and len(self.images) != len(self.masks)
            ):
                print(
                    f"Warning: Number of images ({len(self.images)}) and masks ({len(self.masks)}) do not match in {split} set."
                )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_name = self.images[index]
        image_path = os.path.join(self.images_dir, image_name)
        try:
            image = tifffile.imread(image_path).astype(np.float32)
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            return torch.zeros(7, self.size, self.size), torch.zeros(
                self.size, self.size
            ).long()
        if image.ndim == 3 and image.shape[2] == 7:
            image = image.transpose(2, 0, 1)
        image[image < -100000] = 0.0
        if image.shape[0] == 7:
            for c in range(7):
                image[c] = (image[c] - self.mean[c, 0, 0]) / (self.std[c, 0, 0] + 1e-8)

        if self.split == "test":
            return torch.from_numpy(image).float(), image_name
        if index < len(self.masks):
            mask_name = self.masks[index]
            mask_path = os.path.join(self.masks_dir, mask_name)
            mask = tifffile.imread(mask_path).astype(np.float32)
        else:
            mask = np.zeros((self.size, self.size))

        image_t = torch.from_numpy(image).float()
        mask_t = torch.from_numpy(mask).long()

        if self.augmentor is not None:
            image_t, mask_t = self.augmentor(image_t, mask_t)
        return image_t, mask_t


class MMLSV2Dataset(MarsSegDataset):
    """Dataset for mmlsv2 released data which is already normalized to [0, 1].

    Inherits MarsSegDataset but removes the mean/std normalization step.
    """

    def __init__(self, root_dir, split="train", size=128, val_ana=False):
        super().__init__(root_dir, split, size, val_ana)
        # Disable normalization: mmlsv2 data is pre-normalized to [0, 1]
        self.mean = None
        self.std = None

    def __getitem__(self, index):
        image_name = self.images[index]
        image_path = os.path.join(self.images_dir, image_name)
        try:
            image = tifffile.imread(image_path).astype(np.float32)
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            return torch.zeros(7, self.size, self.size), torch.zeros(
                self.size, self.size
            ).long()
        if image.ndim == 3 and image.shape[2] == 7:
            image = image.transpose(2, 0, 1)
        image[image < -100000] = 0.0
        # No mean/std normalization for mmlsv2 data

        if self.split == "test":
            return torch.from_numpy(image).float(), image_name
        if index < len(self.masks):
            mask_name = self.masks[index]
            mask_path = os.path.join(self.masks_dir, mask_name)
            mask = tifffile.imread(mask_path).astype(np.float32)
        else:
            mask = np.zeros((self.size, self.size))

        image_t = torch.from_numpy(image).float()
        mask_t = torch.from_numpy(mask).long()

        if self.augmentor is not None:
            image_t, mask_t = self.augmentor(image_t, mask_t)
        return image_t, mask_t


class MarsSegDatasetInferV0(data.Dataset):
    def __init__(self, root_dir, split="train", size=128, val_ana=False):
        """
        root_dir: Dataset root directory containing train/ val/ test/ subdirectories
        split: 'train', 'val', or 'test'
        """
        self.root_dir = root_dir
        self.split = split
        self.size = size

        self.augmentor = MarsAugmentor(prob=0.5) if split == "train" else None

        if split == "train":
            mars_mean = MARS_MEAN_TRAIN
            mars_std = MARS_STD_TRAIN
        elif split == "val":
            mars_mean = MARS_MEAN_VAL
            mars_std = MARS_STD_VAL
        else:
            mars_mean = MARS_MEAN_VAL
            mars_std = MARS_STD_VAL

        self.mean = np.array(mars_mean, dtype=np.float32).reshape(7, 1, 1)
        self.std = np.array(mars_std, dtype=np.float32).reshape(7, 1, 1)

        self.images_dir = os.path.join(root_dir, "val", "images")
        self.masks_dir = os.path.join(root_dir, "val", "masks")

        if not os.path.exists(self.images_dir):
            if split != "test":
                print(f"Warning: Directory not found: {self.images_dir}")
            self.images = []
        else:
            self.images = sorted(
                [f for f in os.listdir(self.images_dir) if f.endswith(".tif")]
            )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_name = self.images[index]
        image_path = os.path.join(self.images_dir, image_name)

        try:
            image = tifffile.imread(image_path).astype(np.float32)
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            return torch.zeros(7, self.size, self.size), torch.zeros(
                self.size, self.size
            ).long()
        if image.ndim == 3 and image.shape[2] == 7:
            image = image.transpose(2, 0, 1)
        image[image < -100000] = 0.0

        if image.shape[0] == 7:
            for c in range(7):
                image[c] = (image[c] - self.mean[c, 0, 0]) / (self.std[c, 0, 0] + 1e-8)

        if self.split == "test":
            return torch.from_numpy(image).float(), image_name

        if index < len(self.masks):
            mask_name = self.masks[index]
            mask_path = os.path.join(self.masks_dir, mask_name)
            mask = tifffile.imread(mask_path).astype(np.float32)
        else:
            mask = np.zeros((self.size, self.size))

        image_t = torch.from_numpy(image).float()
        mask_t = torch.from_numpy(mask).long()
        if self.augmentor is not None:
            image_t, mask_t = self.augmentor(image_t, mask_t)

        return image_t, mask_t


class MarsSegDatasetInferV1(data.Dataset):
    def __init__(self, root_dir, split="train", size=128, val_ana=False):
        """
        root_dir: Dataset root directory containing train/ val/ test/ subdirectories
        split: 'train', 'val', or 'test'
        """
        self.root_dir = root_dir
        self.split = split
        self.size = size

        self.augmentor = MarsAugmentor(prob=0.5) if split == "train" else None

        if split == "train":
            mars_mean = MARS_MEAN_TRAIN
            mars_std = MARS_STD_TRAIN
        elif split == "val":
            mars_mean = MARS_MEAN_VAL
            mars_std = MARS_STD_VAL
        else:  # test
            mars_mean = MARS_MEAN_TEST
            mars_std = MARS_STD_TEST

        self.mean = np.array(mars_mean, dtype=np.float32).reshape(7, 1, 1)
        self.std = np.array(mars_std, dtype=np.float32).reshape(7, 1, 1)

        self.images_dir = os.path.join(root_dir, split, "images")
        self.masks_dir = os.path.join(root_dir, split, "masks")

        if not os.path.exists(self.images_dir):
            if split != "test":
                print(f"Warning: Directory not found: {self.images_dir}")
            self.images = []
        else:
            self.images = sorted(
                [f for f in os.listdir(self.images_dir) if f.endswith(".tif")]
            )

        if val_ana:
            self.images_dir = os.path.join(root_dir, "val", "images")
            self.masks_dir = os.path.join(root_dir, "val", "masks")
            self.images = sorted(
                [f for f in os.listdir(self.images_dir) if f.endswith(".tif")]
            )

        if split != "test":
            if not os.path.exists(self.masks_dir):
                self.masks = []
            else:
                self.masks = sorted(
                    [f for f in os.listdir(self.masks_dir) if f.endswith(".tif")]
                )

            if (
                len(self.images) > 0
                and len(self.masks) > 0
                and len(self.images) != len(self.masks)
            ):
                print(
                    f"Warning: Number of images ({len(self.images)}) and masks ({len(self.masks)}) do not match in {split} set."
                )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_name = self.images[index]
        image_path = os.path.join(self.images_dir, image_name)

        try:
            image = tifffile.imread(image_path).astype(np.float32)
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            return torch.zeros(7, self.size, self.size), torch.zeros(
                self.size, self.size
            ).long()
        if image.ndim == 3 and image.shape[2] == 7:
            image = image.transpose(2, 0, 1)
        image[image < -100000] = 0.0

        if image.shape[0] == 7:
            for c in range(7):
                image[c] = (image[c] - self.mean[c, 0, 0]) / (self.std[c, 0, 0] + 1e-8)

        if self.split == "test":
            return torch.from_numpy(image).float(), image_name

        if index < len(self.masks):
            mask_name = self.masks[index]
            mask_path = os.path.join(self.masks_dir, mask_name)
            mask = tifffile.imread(mask_path).astype(np.float32)
        else:
            mask = np.zeros((self.size, self.size))

        image_t = torch.from_numpy(image).float()
        mask_t = torch.from_numpy(mask).long()
        if self.augmentor is not None:
            image_t, mask_t = self.augmentor(image_t, mask_t)

        return image_t, mask_t


class MarsSegDatasetInferV2(data.Dataset):
    def __init__(self, root_dir, split="train", size=128, val_ana=False):
        """
        root_dir: Dataset root directory containing train/ val/ test/ subdirectories
        split: 'train', 'val', or 'test'
        """
        self.root_dir = root_dir
        self.split = split
        self.size = size
        self.augmentor = MarsAugmentor(prob=0.5) if split == "train" else None

        self.train_mean = np.array(MARS_MEAN_TRAIN, dtype=np.float32).reshape(7, 1, 1)
        self.train_std = np.array(MARS_STD_TRAIN, dtype=np.float32).reshape(7, 1, 1)
        self.test_mean = np.array(MARS_MEAN_TESTB, dtype=np.float32).reshape(7, 1, 1)
        self.test_std = np.array(MARS_STD_TESTB, dtype=np.float32).reshape(7, 1, 1)
        if split == "train":
            mars_mean = MARS_MEAN_TRAIN
            mars_std = MARS_STD_TRAIN
        elif split == "val":
            mars_mean = MARS_MEAN_VAL
            mars_std = MARS_STD_VAL
        else:
            mars_mean = MARS_MEAN_TEST
            mars_std = MARS_STD_TEST

        self.mean = np.array(mars_mean, dtype=np.float32).reshape(7, 1, 1)
        self.std = np.array(mars_std, dtype=np.float32).reshape(7, 1, 1)

        self.images_dir = os.path.join(root_dir, split, "images")
        self.masks_dir = os.path.join(root_dir, split, "masks")

        if not os.path.exists(self.images_dir):
            if split != "test":
                print(f"Warning: Directory not found: {self.images_dir}")
            self.images = []
        else:
            self.images = sorted(
                [f for f in os.listdir(self.images_dir) if f.endswith(".tif")]
            )

        if val_ana:
            self.images_dir = os.path.join(root_dir, "val", "images")
            self.masks_dir = os.path.join(root_dir, "val", "masks")
            self.images = sorted(
                [f for f in os.listdir(self.images_dir) if f.endswith(".tif")]
            )

        if split != "test":
            if not os.path.exists(self.masks_dir):
                self.masks = []
            else:
                self.masks = sorted(
                    [f for f in os.listdir(self.masks_dir) if f.endswith(".tif")]
                )

            if (
                len(self.images) > 0
                and len(self.masks) > 0
                and len(self.images) != len(self.masks)
            ):
                print(
                    f"Warning: Number of images ({len(self.images)}) and masks ({len(self.masks)}) do not match in {split} set."
                )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_name = self.images[index]
        image_path = os.path.join(self.images_dir, image_name)
        try:
            image = tifffile.imread(image_path).astype(np.float32)
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            return torch.zeros(7, self.size, self.size), torch.zeros(
                self.size, self.size
            ).long()

        if image.ndim == 3 and image.shape[2] == 7:
            image = image.transpose(2, 0, 1)

        image[image < -100000] = 0.0

        if image.shape[0] == 7:
            for c in range(7):
                if c in [2]:
                    x_base = (image[c] - self.mean[c, 0, 0]) / (
                        self.std[c, 0, 0] + 1e-8
                    )
                    x_aligned = (image[c] - self.test_mean[c, 0, 0]) / (
                        self.test_std[c, 0, 0] + 1e-8
                    ) * self.train_std[c, 0, 0] + self.train_mean[c, 0, 0]

                    x_aligned_norm = (x_aligned - self.train_mean[c, 0, 0]) / (
                        self.train_std[c, 0, 0] + 1e-8
                    )
                    a = 0.33
                    image[c] = (1 - a) * x_base + a * x_aligned_norm
                else:
                    image[c] = (image[c] - self.mean[c, 0, 0]) / (
                        self.std[c, 0, 0] + 1e-8
                    )

        if self.split == "test":
            return torch.from_numpy(image).float(), image_name

        if index < len(self.masks):
            mask_name = self.masks[index]
            mask_path = os.path.join(self.masks_dir, mask_name)
            mask = tifffile.imread(mask_path).astype(np.float32)
        else:
            mask = np.zeros((self.size, self.size))

        image_t = torch.from_numpy(image).float()
        mask_t = torch.from_numpy(mask).long()

        if self.augmentor is not None:
            image_t, mask_t = self.augmentor(image_t, mask_t)
        return image_t, mask_t


class MosaicCastDataset(data.Dataset):
    def __init__(self, dataset, prob=0.5, size=128):
        self.dataset = dataset
        self.prob = prob
        self.size = size

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if torch.rand(1).item() > self.prob:
            img, mask = self.dataset[idx]
            return img, mask

        dataset_len = len(self.dataset)
        random_indices = torch.randint(0, dataset_len, (3,)).tolist()
        indices = [idx] + random_indices

        images, masks = [], []
        for i in indices:
            img, mask = self.dataset[i]
            images.append(img)
            masks.append(mask)

        top_img = torch.cat([images[0], images[1]], dim=2)
        bot_img = torch.cat([images[2], images[3]], dim=2)
        mosaic_img = torch.cat([top_img, bot_img], dim=1)

        top_mask = torch.cat([masks[0], masks[1]], dim=1)
        bot_mask = torch.cat([masks[2], masks[3]], dim=1)
        mosaic_mask = torch.cat([top_mask, bot_mask], dim=0)

        max_i = mosaic_img.shape[1] - self.size
        max_j = mosaic_img.shape[2] - self.size

        i = torch.randint(0, max_i + 1, (1,)).item()
        j = torch.randint(0, max_j + 1, (1,)).item()

        final_img = mosaic_img[:, i : i + self.size, j : j + self.size]
        final_mask = mosaic_mask[i : i + self.size, j : j + self.size]

        return final_img, final_mask


if __name__ == "__main__":
    pass
    for i in range(7):
        print(
            f"channel[{i}]: {MARS_MEAN_VAL[i]:.02f}({MARS_MEAN_VAL[i] - MARS_MEAN_TRAIN[i]:.02f}, {(MARS_MEAN_VAL[i] - MARS_MEAN_TRAIN[i]) / MARS_MEAN_TRAIN[i] * 100:.02f}%);{MARS_MEAN_TEST[i]:.02f}({MARS_MEAN_TEST[i] - MARS_MEAN_TRAIN[i]:.02f}, {(MARS_MEAN_TEST[i] - MARS_MEAN_TRAIN[i]) / MARS_MEAN_TRAIN[i] * 100:.02f}%);{MARS_MEAN_TESTB[i]:.02f}({MARS_MEAN_TESTB[i] - MARS_MEAN_TRAIN[i]:.02f}, {(MARS_MEAN_TESTB[i] - MARS_MEAN_TRAIN[i]) / MARS_MEAN_TRAIN[i] * 100:.02f}%)"
        )
    for i in range(7):
        print(
            f"channel[{i}]: {MARS_STD_VAL[i]:.02f}({MARS_STD_VAL[i] - MARS_STD_TRAIN[i]:.02f}, {(MARS_STD_VAL[i] - MARS_STD_TRAIN[i]) / MARS_STD_TRAIN[i] * 100:.02f}%);{MARS_STD_TEST[i]:.02f}({MARS_STD_TEST[i] - MARS_STD_TRAIN[i]:.02f}, {(MARS_STD_TEST[i] - MARS_STD_TRAIN[i]) / MARS_STD_TRAIN[i] * 100:.02f}%);{MARS_MEAN_TESTB[i]:.02f}({MARS_STD_TESTB[i] - MARS_STD_TRAIN[i]:.02f}, {(MARS_STD_TESTB[i] - MARS_STD_TRAIN[i]) / MARS_STD_TRAIN[i] * 100:.02f}%))"
        )
