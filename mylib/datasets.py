import os
from PIL import Image
from torch.utils.data import Dataset


class DatasetFromCSV(Dataset):

    def __init__(self, data, classes, root_dir, ext, transform=None):
        """
        This class loads image data from a csv file. Check the
        https://www.kaggle.com/c/dog-breed-identification/data
        data format.

        :param data: list of (id, class name) pairs. id is assumed to be the filename without the extension.
        :param classes: list of class names (order is important, should match desired output of NN)
        :param root_dir: directory where all images reside
        :param ext: file extension of the images (will be appended to the id, which is the filename)
        :param transform: transforms.Compose() object
        """
        self.data = data
        self.classes = classes
        self.root_dir = root_dir
        self.ext = ext
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        :param idx: index of the item to fetch
        :return: (PILImage, index of class) tupple
        """

        path_to_image = os.path.join(self.root_dir, self.data[idx][0] + '.' + self.ext)
        image = Image.open(path_to_image).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        breed_name = self.data[idx][1]

        return image, self.classes.index(breed_name)