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

def plot_gradient_steps(ax, grad_flows, activation_name, layers):
    num_trials = grad_flows.shape[0]
    total_steps = grad_flows.shape[2]
    steps_range = range(1, total_steps + 1)
    for layer_idx, layer_name in enumerate(layers):
        clean_name = layer_name.replace('.weight', '')
        
        for trial_idx in range(num_trials):
            layer_data = grad_flows[trial_idx, layer_idx, :]  
            layer_data_safe = layer_data + 1e-8
            label_text = clean_name if trial_idx == 0 else None
            ax.plot(steps_range, layer_data_safe, 
                    label=label_text, 
                    alpha=0.35, 
                    linewidth=0.7)
        
    ax.set_title(f"{activation_name} Raw Step Gradients (All Trials)")
    ax.set_xlabel("Global Training Step (Batch)")
    ax.set_ylabel("Gradient RMS Magnitude")
    ax.grid(alpha=0.3)
    ax.set_yscale('log')  
    ax.legend(fontsize=8, loc='upper right')

def plot_curves(activations, avg_accuracy_curve, avg_loss_curve):
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(avg_accuracy_curve[list(activations.keys())[0]]) + 1)
    for name in activations:
        arr = np.array(avg_accuracy_curve[name])
        mean_curve = np.mean(arr, axis=0)
        std_curve = np.std(arr, axis=0)

        ax[0].plot(epochs, mean_curve, label=name)
        ax[0].fill_between(
            epochs,
            mean_curve - std_curve,
            mean_curve + std_curve,
            alpha=0.2
        )
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Accuracy")
    ax[0].set_title("Accuracy Curve (Mean and Std over 5 Trials)")
    ax[0].legend()
    ax[0].grid(alpha=0.3)

    for name in activations.keys():
        arr = np.array(avg_loss_curve[name])

        mean_curve = np.mean(arr, axis=0)
        std_curve = np.std(arr, axis=0)

        ax[1].plot(epochs, mean_curve, label=name)
        ax[1].fill_between(
            epochs,
            mean_curve - std_curve,
            mean_curve + std_curve,
            alpha=0.2
        )
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Loss")
    ax[1].set_title("Loss Curve (Mean and Std over 5 Trials)")
    ax[1].legend()
    ax[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    fig.savefig('accuracy_loss_curves.pdf', bbox_inches='tight')

def main():
    experiment_mode = False
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=False, num_workers=2)

    
    activations = {
        'Tanh': nn.Tanh(),
        'ReLU': nn.ReLU(),
        'GELU': nn.GELU()

    }
    final_test_loss = {activation: [] for activation in activations}
    final_accuracy = {activation: [] for activation in activations}
    avg_accuracy_curve = {activation: [] for activation in activations}
    avg_loss_curve = {activation: [] for activation in activations}
    avg_grad_flows = {activation: [] for activation in activations}
    
    epochs = 2 if experiment_mode else 15
    seeds = [1, 2] if experiment_mode else range(1, 6)
    for seed in seeds:
        print(f'########## TRIAL {seed} ##########')
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        for name, activation in activations.items():
            print(f'--- Training {name} model ---')
            model = CNN(activation)
            test_loss, acc, grad_flows = train(model, train_loader, test_loader, device, epochs=epochs)
            final_test_loss[name].append(test_loss[-1])
            final_accuracy[name].append(acc[-1])
            avg_loss_curve[name].append(test_loss)
            avg_accuracy_curve[name].append(acc)
            avg_grad_flows[name].append(grad_flows)

    print("########################## Final Results ##########################\n\n")
    for name in activations:
        print(name)
        print(f'{name}: Test Loss: {final_test_loss[name]} | Accuracy: {final_accuracy[name]}\n')

    fig, ax = plt.subplots(1, 3, figsize=(21, 6))
    for i, name in enumerate(activations):
        grad_trial = []
        for trial in avg_grad_flows[name]:
            layers = list(trial.keys())
            grad_flows = np.array([trial[layer] for layer in layers])
            grad_trial.append(grad_flows)
        avg_grad_flows[name] = np.array(grad_trial)
        plot_gradient_steps(ax[i], avg_grad_flows[name], name, layers)
    fig.tight_layout()
    fig.savefig('gradient_curves.pdf', bbox_inches='tight')
    plt.show()
    plot_curves(activations, avg_accuracy_curve, avg_loss_curve)

if __name__ == '__main__': main()