import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

class FederatedClient:
    def __init__(self, client_id, data_X, data_y, model_builder):
        self.client_id = client_id
        self.data_X = data_X
        self.data_y = data_y
        self.model = model_builder()

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, weights):
        self.model.set_weights(weights)

    def train_epoch(self, global_weights, epochs=1, batch_size=32):
        self.set_weights(global_weights)
        self.model.compile(optimizer=\'adam\', loss=\'sparse_categorical_crossentropy\', metrics=[\'accuracy\'])
        self.model.fit(self.data_X, self.data_y, epochs=epochs, batch_size=batch_size, verbose=0)
        return self.get_weights(), len(self.data_X)

class FederatedServer:
    def __init__(self, model_builder, clients):
        self.global_model = model_builder()
        self.clients = clients

    def aggregate_weights(self, client_weights_list, client_data_sizes):
        # Federated Averaging (FedAvg)
        new_weights = [np.zeros_like(w) for w in self.global_model.get_weights()]
        total_data_size = sum(client_data_sizes)

        for client_weights, data_size in zip(client_weights_list, client_data_sizes):
            for i in range(len(new_weights)):
                new_weights[i] += client_weights[i] * (data_size / total_data_size)
        return new_weights

    def federated_round(self, epochs_per_client=1, batch_size=32):
        global_weights = self.global_model.get_weights()
        all_client_weights = []
        all_client_data_sizes = []

        for client in self.clients:
            client_weights, data_size = client.train_epoch(global_weights, epochs=epochs_per_client, batch_size=batch_size)
            all_client_weights.append(client_weights)
            all_client_data_sizes.append(data_size)
        
        aggregated_weights = self.aggregate_weights(all_client_weights, all_client_data_sizes)
        self.global_model.set_weights(aggregated_weights)
        return self.global_model.get_weights()

    def evaluate_global_model(self, test_X, test_y):
        self.global_model.compile(optimizer=\'adam\', loss=\'sparse_categorical_crossentropy\', metrics=[\'accuracy\'])
        loss, accuracy = self.global_model.evaluate(test_X, test_y, verbose=0)
        return loss, accuracy

# Example Usage
if __name__ == "__main__":
    # 1. Define a simple model architecture
    def create_cnn_model(input_shape=(28, 28, 1), num_classes=10):
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation=\'relu\', input_shape=input_shape),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation=\'relu\'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(64, activation=\'relu\'),
            layers.Dense(num_classes, activation=\'softmax\')
        ])
        return model

    # 2. Load and preprocess data (e.g., MNIST)
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1).astype(\'float32\') / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1).astype(\'float32\') / 255.0

    # 3. Simulate multiple clients with data partitions
    num_clients = 5
    client_data_size = len(x_train) // num_clients
    clients = []
    for i in range(num_clients):
        start_idx = i * client_data_size
        end_idx = (i + 1) * client_data_size
        client_X = x_train[start_idx:end_idx]
        client_y = y_train[start_idx:end_idx]
        clients.append(FederatedClient(client_id=i, data_X=client_X, data_y=client_y, model_builder=create_cnn_model))

    # 4. Initialize and run Federated Server
    server = FederatedServer(model_builder=create_cnn_model, clients=clients)

    print("Initial global model evaluation:")
    loss, accuracy = server.evaluate_global_model(x_test, y_test)
    print(f"Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

    num_federated_rounds = 5
    for round_num in range(num_federated_rounds):
        print(f"\nFederated Round {round_num + 1}/{num_federated_rounds}")
        server.federated_round(epochs_per_client=1)
        loss, accuracy = server.evaluate_global_model(x_test, y_test)
        print(f"Global Model - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
