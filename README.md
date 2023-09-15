# Federated Learning Framework

This framework provides a modular and extensible implementation of federated learning, allowing multiple clients to collaboratively train a shared global model without exchanging their local data.

## Features

- **Federated Averaging (FedAvg)**: Implementation of the FedAvg algorithm for model aggregation.
- **Client-Server Architecture**: Simulation of distributed clients and a central server.
- **Privacy-Preserving**: Designed to keep local data on client devices.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Federated Training Example

```python
import tensorflow as tf
from federated_model import FederatedClient, FederatedServer, create_cnn_model

# Load and preprocess data (e.g., MNIST)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1).astype(\'float32\') / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype(\'float32\') / 255.0

# Simulate multiple clients with data partitions
num_clients = 3
client_data_size = len(x_train) // num_clients
clients = []
for i in range(num_clients):
    start_idx = i * client_data_size
    end_idx = (i + 1) * client_data_size
    client_X = x_train[start_idx:end_idx]
    client_y = y_train[start_idx:end_idx]
    clients.append(FederatedClient(client_id=i, data_X=client_X, data_y=client_y, model_builder=create_cnn_model))

# Initialize and run Federated Server
server = FederatedServer(model_builder=create_cnn_model, clients=clients)

print("Initial global model evaluation:")
loss, accuracy = server.evaluate_global_model(x_test, y_test)
print(f"Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

num_federated_rounds = 3
for round_num in range(num_federated_rounds):
    print(f"\nFederated Round {round_num + 1}/{num_federated_rounds}")
    server.federated_round(epochs_per_client=1)
    loss, accuracy = server.evaluate_global_model(x_test, y_test)
    print(f"Global Model - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
