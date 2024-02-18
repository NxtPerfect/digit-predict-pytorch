from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import Resize, ToTensor

test_batch_size = 4096
training_batch_size = 4096

resize_transform = Resize((28, 28))

training_data = datasets.EMNIST(
        root="data",
        split="digits",
        train=True,
        download=True,
        # transform=ToTensor()
        transform=transforms.Compose([ToTensor(), resize_transform])
        )

test_data_emnist = datasets.EMNIST(
        root="data",
        split="digits",
        download=True,
        # transform=ToTensor()
        transform=transforms.Compose([ToTensor(), resize_transform])
        )

test_data_semeion = datasets.SEMEION(
        root="data",
        download=True,
        transform=transforms.Compose([ToTensor(), resize_transform])
    )

numbers_map = {
        0: "zero",
        1: "one",
        2: "two",
        3: "three",
        4: "four",
        5: "five",
        6: "six",
        7: "seven",
        8: "eight",
        9: "nine"
        }

train_dataloader = DataLoader(training_data, batch_size=training_batch_size, shuffle=True, num_workers=12)
test_dataloader_emnist = DataLoader(test_data_emnist, batch_size=test_batch_size, shuffle=True, num_workers=12)
test_dataloader_semeion = DataLoader(test_data_semeion, batch_size=test_batch_size, shuffle=True, num_workers=12)
