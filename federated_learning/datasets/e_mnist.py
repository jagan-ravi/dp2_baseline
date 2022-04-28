from .dataset import Dataset
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader

class EMNISTDataset(Dataset):

    def __init__(self, args):
        super(EMNISTDataset, self).__init__(args)

    def load_train_dataset(self):
        self.get_args().get_logger().debug("Loading MNIST train data")

        train_dataset = datasets.EMNIST(self.get_args().get_data_path(),split="mnist",train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
        train_loader = DataLoader(train_dataset, batch_size=len(train_dataset))

        train_data = self.get_tuple_from_data_loader(train_loader)

        self.get_args().get_logger().debug("Finished loading Fashion MNIST train data")

        return train_data

    def load_test_dataset(self):
        self.get_args().get_logger().debug("Loading Fashion MNIST test data")

        test_dataset = datasets.EMNIST(self.get_args().get_data_path(),split="mnist", train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))
        test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))

        test_data = self.get_tuple_from_data_loader(test_loader)

        self.get_args().get_logger().debug("Finished loading Fashion MNIST test data")

        return test_data
