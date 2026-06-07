import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.v2 as v2
import numpy as np
import random
import matplotlib.pyplot as plt
from model import CNN
from training import train

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

def gradient_heatmap(grad_flows, activation, layers):
    matrix = np.mean(grad_flows, axis=0)

    plt.figure(figsize=(12, 6))
    im = plt.imshow(matrix, cmap='viridis', aspect='auto')
    plt.colorbar(im, label='Gradient L2 Norm')
    plt.yticks(ticks=range(len(layers)), labels=layers)
    plt.xlabel('Epoch')
    plt.ylabel('Layer')
    plt.title(f'Gradient Flow Heatmap for {activation}')

    plt.show()

def main():

    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=False, num_workers=2)

    
    activations = {
        'ReLU': nn.ReLU(),
        'LeakyReLU': nn.LeakyReLU(),
        'GELU': nn.GELU(),
        'Tanh': nn.Tanh()
    }
    final_test_loss = {activation: [] for activation in activations}
    final_accuracy = {activation: [] for activation in activations}
    avg_accuracy_curve = {activation: [] for activation in activations}
    avg_loss_curve = {activation: [] for activation in activations}
    avg_grad_flows = {activation: [] for activation in activations}
    
    for seed in range(1, 6):
        print(f'########## TRIAL {seed} ##########')
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        for name, activation in activations.items():
            print(f'--- Training {name} model ---')
            model = CNN(activation)
            test_loss, acc, grad_flows = train(model, train_loader, test_loader, device)
            final_test_loss[name].append(test_loss[-1])
            final_accuracy[name].append(acc[-1])
            avg_loss_curve[name].append(test_loss)
            avg_accuracy_curve[name].append(acc)
            avg_grad_flows[name].append(grad_flows)

    print("########################## Final Results ##########################\n\n")
    for name in activations:
        print(name)
        print(f'{name}: Test Loss: {final_test_loss[name]} | Accuracy: {final_accuracy[name]}\n')
    accuracy_means = [np.mean(a) for a in final_accuracy.values()]
    accuracy_stds = [np.std(a) for a in final_accuracy.values()]
    test_loss_means = [np.mean(a) for a in final_test_loss.values()]
    test_loss_stds = [np.std(a) for a in final_test_loss.values()]

    plt.style.use("seaborn-v0_8-darkgrid")
    plt.figure(figsize=(8, 5))
    plt.bar(list(activations.keys()), accuracy_means, yerr=accuracy_stds, capsize=5)
    plt.ylabel("Mean Accuracy")
    plt.title("Mean Accuracy with Standard Deviations (5 Trials)")
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    plt.style.use("seaborn-v0_8-darkgrid")
    plt.figure(figsize=(8, 5))
    plt.bar(list(activations.keys()), test_loss_means, yerr=test_loss_stds, capsize=5)
    plt.ylabel("Mean Test Loss")
    plt.title("Mean Test Loss with Standard Deviations (5 Trials)")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5))
    for name in activations:
        grad_trial = []
        for trial in avg_grad_flows[name]:
            layers = list(trial.keys())
            grad_flows = np.array([trial[layer] for layer in layers])
            grad_trial.append(grad_flows)
        avg_grad_flows[name] = np.array(grad_trial)
        gradient_heatmap(avg_grad_flows[name], name, layers)

    for name in activations:
        arr = np.array(avg_accuracy_curve[name])
        mean_curve = np.mean(arr, axis=0)
        std_curve = np.std(arr, axis=0)

        plt.plot(range(1,16), mean_curve, label=name)
        plt.fill_between(
            range(1,16),
            mean_curve - std_curve,
            mean_curve + std_curve,
            alpha=0.2
        )
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curve (Mean and Std over 5 Trials)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5))
    for name in activations.keys():
        arr = np.array(avg_loss_curve[name])

        mean_curve = np.mean(arr, axis=0)
        std_curve = np.std(arr, axis=0)

        plt.plot(range(1,16), mean_curve, label=name)
        plt.fill_between(
            range(1,16),
            mean_curve - std_curve,
            mean_curve + std_curve,
            alpha=0.2
        )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve (Mean and Std over 5 Trials)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__': main()