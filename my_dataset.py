from torchvision.datasets import CIFAR100
import numpy as np
import torch
from PIL import Image

class MyDataset(CIFAR100):
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        ########## modified
        img = (img.data*255).byte()
        img.numpy()

        ch, x, y = img.shape
        out_shape = (8, x, y)

        out1 = np.unpackbits(img[0].reshape(x * y, 1), axis=1).T.reshape(out_shape)
        out2 = np.unpackbits(img[1].reshape(x * y, 1), axis=1).T.reshape(out_shape)
        out3 = np.unpackbits(img[2].reshape(x * y, 1), axis=1).T.reshape(out_shape)
        per_input = np.concatenate([out1, out2, out3])
        img = torch.from_numpy(per_input)
        #########

        return img, target

