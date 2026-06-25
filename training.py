import torch
import torch.nn as nn
import matplotlib.pyplot as plt


def train(model, train_loader, test_loader, device, epochs=15):
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()
    train_losses, test_losses, test_accuracies, grad_flows = [], [], [], {}

    for name, param in model.named_parameters():
        if 'weight' in name:
            grad_flows[name] = []

    sparsities = []
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for step_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()

            for name, param in model.named_parameters():
                if 'weight' in name and param.grad is not None:
                    grad_flows[name].append(param.grad.norm().item())
            optimizer.step()
            running_loss += loss.item()
        sparsities.append(sparsity(model))
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)
        test_loss, accuracy = evaluate(model, test_loader, device)
        test_losses.append(test_loss)
        test_accuracies.append(accuracy)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | Acc: {accuracy:.4f}")
    return sparsities, test_losses, test_accuracies, grad_flows

# trials x epochs x layers (avg over trials)
def sparsity(model):
    activation = model.activation.__class__.__name__
    sparsities = [(out.detach().cpu() < 0.01).float().mean().item() for out in model.saved.values()] if activation == 'ReLU' else [(out.detach().cpu().abs() > 0.97).float().mean().item() for out in model.saved.values()]
    return sparsities

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