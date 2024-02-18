from torch import nn
from torch import optim
from torchvision import torch
from model import NeuralNetwork
from dataset import training_data, test_data, train_dataloader, test_dataloader


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
    optimizer = optim.SGD(model.parameters(), lr=0.04) # 0.003

    # Define a learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=-1.1) # step_size 15 gamma -1.1

    # Number of iterations over dataset
    num_epochs = 20
    # Best validation loss and how many epochs to wait until early stop training
    best_validation_loss = float('inf')
    patience = 3

    while num_epochs > 0:
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
    correct = 0
    total = 0

    with torch.no_grad():
        validation_loss = 0.0
        for inputs, labels in test_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            # # Reshape inputs to match the expected input size (assuming it's 28x28)
            # inputs = inputs.view(inputs.size(0), -1, 1)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            validation_loss += loss.item()

    scheduler.step()

    accuracy = correct / total
    avg_validation_loss = validation_loss / len(test_dataloader)
    print(f'Test accuracy: {accuracy * 100:.2f}%, with average validation loss: {avg_validation_loss:.6f}.')
