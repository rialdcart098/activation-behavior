import torch
import torch.nn as nn

def train(model, train_loader, test_loader, device, epochs=15):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    train_losses, test_losses, test_accuracies = [], [], []
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)
        test_loss, accuracy = evaluate(model, test_loader, device)
        test_losses.append(test_loss)
        test_accuracies.append(accuracy)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | Acc: {accuracy:.4f}")
    return train_losses, test_losses, test_accuracies

def evaluate(model, data_loader, device):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            batch_size = labels.size(0)
            total_loss += loss_fn(outputs, labels).item() * batch_size
            preds = torch.argmax(outputs, dim=1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy