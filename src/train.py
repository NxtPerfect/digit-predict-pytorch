from torch import nn
from torch import optim
from torchvision import torch
from model import NeuralNetwork
from dataset import train_dataloader, test_dataloader_emnist, test_dataloader_semeion


if __name__ == "__main__":
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")
    model = NeuralNetwork().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.026)
    # learning rate EMNIST accuracy + loss / SEMEION accuracy + loss optimizer
    # 0.04 92.22+1.59/8.78+1.6 EMNIST/SEMEION with SDM
    # 0.06 97.22+1.51/10.55+1.5 ADAM
    # 0.03 98.29+1.48/15.13+1.48 ADAM
    # 0.026 98.45+1.48/12.55+1.48 ADAM

    # Define a learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1) # step_size 15 gamma -1.1 25/0.1 98.45/13.56

    # Number of iterations over dataset
    num_epochs = 20
    # How many epochs to wait until early stop training with no change in patience
    patience = 3

    while num_epochs > 0 and patience > 0:
        for inputs, labels in train_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch {20-num_epochs+1}/{20}, Loss: {loss.item():.6f}')
        num_epochs -= 1

    model.eval()

    with torch.no_grad():
        validation_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in test_dataloader_emnist:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            validation_loss += loss.item()


    scheduler.step()

    accuracy = correct / total
    avg_validation_loss = validation_loss / len(test_dataloader_emnist)

    print(f"\n{'-' * 20}\nEMNIST:\n\nTest accuracy: {accuracy * 100:.2f}%, with average validation loss: {avg_validation_loss:.6f}.")

    with torch.no_grad():
        validation_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in test_dataloader_semeion:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            validation_loss += loss.item()

    scheduler.step()

    accuracy = correct / total
    avg_validation_loss = validation_loss / len(test_dataloader_semeion)

    print(f"SEMEION:\n\nTest accuracy: {accuracy * 100:.2f}%, with average validation loss: {avg_validation_loss:.6f}.")
