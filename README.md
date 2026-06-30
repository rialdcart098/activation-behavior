# On the Persistence of Classic Activation Limitations in CNNs

This project investigates if classic activation limitations, such as saturation fron Tanh layers, and the Dying ReLU problem still meaningfully appear in modern CNNs. These problems are well-known in textbooks and in theory, but it is unclear how strongly they affect modern architecture.

### Research Question:
- To what extent do classic activation limitations (Dying ReLU, Tanh saturation) persist in modern CNNs?
---
### Methodology:
  -  Used two small CNNs with **Kaiming** initialization, 3 convolutional layers, Max Pooling, and 2 fully connected layers. One with ReLU activations and one with Tanh activations.
  - SGD optimizer with a learning rate of 0.01 and momentum of 0.9, trained on CIFAR-10 over 15 epochs
  - Ran 5 times with different random seeds to account for variability in training.
### Metrics:
  -  Activation and Gradient distributions after training
  - ReLU sparsity (percentage of zero activations) and Tanh saturation (percentage of activations derivatives near 0) over epochs
  - Gradient RMS across layers over steps (batch size * epochs)
  - Loss & Accuracy Curves over epochs
### Discussion:
- ReLU's sparsity still persists, with over deeper layers creeping up to 90% sparsity rate
- ReLU overfits much quicker than Tanh under identical conditions
- Tanh saturation is less of a problem, with loss decreasing steadily, and activation distributions remaining well spread
- With 80-90% of ReLU activations being zero in deeper layers, the activation distribution heavily spiking at 0, and overfitting on epoch 3 as opposed to normal training behavior in tanh, it is clear that the Dying ReLU problem is still a significant issue in modern CNNs.