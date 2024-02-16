from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

training_data = datasets.EMNIST(
        root="data",
        split="digits",
        train=True,
        download=True,
        transform=ToTensor()
        )

test_data = datasets.SEMEION(
        root="data",
        download=True,
        transform=ToTensor()
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

train_dataloader = DataLoader(training_data, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=32, shuffle=True)
