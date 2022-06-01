import tensorflow as tf
import numpy as np

class Client:
    def __init__(self, client_id, X_train, y_train, model_fn):
        self.client_id = client_id
        self.X_train = X_train
        self.y_train = y_train
        self.model = model_fn()

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, weights):
        self.model.set_weights(weights)

    def train(self, epochs=1, batch_size=32):
        self.model.compile(optimizer=\'adam\', loss=\'sparse_categorical_crossentropy\', metrics=\'accuracy\')
        self.model.fit(self.X_train, self.y_train, epochs=epochs, batch_size=batch_size, verbose=0)
        return len(self.X_train)

class Server:
    def __init__(self, model_fn, clients):
        self.global_model = model_fn()
        self.clients = clients

    def aggregate_weights(self, client_weights, client_sizes):
        new_weights = [np.zeros_like(w) for w in self.global_model.get_weights()]
        total_size = sum(client_sizes)

        for c_weights, c_size in zip(client_weights, client_sizes):
            for i in range(len(new_weights)):
                new_weights[i] += c_weights[i] * (c_size / total_size)
        self.global_model.set_weights(new_weights)

    def distribute_weights(self):
        for client in self.clients:
            client.set_weights(self.global_model.get_weights())

    def evaluate_global_model(self, X_test, y_test):
        self.global_model.compile(optimizer=\'adam\', loss=\'sparse_categorical_crossentropy\', metrics=\'accuracy\')
        loss, accuracy = self.global_model.evaluate(X_test, y_test, verbose=0)
        return loss, accuracy

def create_cnn_model(input_shape=(28, 28, 1), num_classes=10):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation=\'relu\', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation=\'relu\'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation=\'relu\
'),
        tf.keras.layers.Dense(num_classes, activation=\'softmax\')
    ])
    return model

if __name__ == "__main__":
    # Load MNIST dataset
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    X_train = X_train.reshape(-1, 28, 28, 1).astype(\'float32\') / 255.0
    X_test = X_test.reshape(-1, 28, 28, 1).astype(\'float32\') / 255.0

    # Simulate 5 clients
    num_clients = 5
    client_data_indices = np.array_split(np.arange(len(X_train)), num_clients)

    clients = []
    for i in range(num_clients):
        indices = client_data_indices[i]
        client_X = X_train[indices]
        client_y = y_train[indices]
        clients.append(Client(i, client_X, client_y, create_cnn_model))

    server = Server(create_cnn_model, clients)

    # Federated Learning rounds
    num_rounds = 5
    for round_num in range(num_rounds):
        print(f"\nFederated Learning Round {round_num + 1}")
        client_weights = []
        client_sizes = []

        server.distribute_weights()

        for client in clients:
            size = client.train(epochs=1)
            client_weights.append(client.get_weights())
            client_sizes.append(size)

        server.aggregate_weights(client_weights, client_sizes)
        loss, accuracy = server.evaluate_global_model(X_test, y_test)
        print(f"Global Model - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

    print("\nFederated Learning simulation complete.")
