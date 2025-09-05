from abc import ABC, abstractmethod

import numpy as np

from ..base_estimator import BaseEstimator
from .classifier import ClassifierMixin

# ============================================================================
# ACTIVATION FUNCTIONS
# ============================================================================


class ActivationFunction(ABC):
    """Base class for activation functions."""

    @abstractmethod
    def forward(self, x):
        """Forward pass."""
        pass

    @abstractmethod
    def backward(self, x):
        """Backward pass (derivative)."""
        pass


class Sigmoid(ActivationFunction):
    """Sigmoid activation function."""

    def forward(self, x):
        # Clip to prevent overflow
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

    def backward(self, x):
        s = self.forward(x)
        return s * (1 - s)


class ReLU(ActivationFunction):
    """ReLU activation function."""

    def forward(self, x):
        return np.maximum(0, x)

    def backward(self, x):
        return (x > 0).astype(float)


class Linear(ActivationFunction):
    """Linear (identity) activation function."""

    def forward(self, x):
        return x

    def backward(self, x):
        return np.ones_like(x)


class Softmax(ActivationFunction):
    """Softmax activation function."""

    def forward(self, x):
        # Numerical stability
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def backward(self, x):
        # For softmax, derivative is more complex and handled in loss function
        s = self.forward(x)
        return s * (1 - s)


class Tanh(ActivationFunction):
    """Tanh activation function (bonus)."""

    def forward(self, x):
        return np.tanh(x)

    def backward(self, x):
        return 1 - np.tanh(x) ** 2


class LeakyReLU(ActivationFunction):
    """Leaky ReLU activation function (bonus)."""

    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def forward(self, x):
        return np.where(x > 0, x, self.alpha * x)

    def backward(self, x):
        return np.where(x > 0, 1, self.alpha)


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================


class LossFunction(ABC):
    """Base class for loss functions."""

    @abstractmethod
    def forward(self, y_true, y_pred):
        """Compute loss."""
        pass

    @abstractmethod
    def backward(self, y_true, y_pred):
        """Compute gradient."""
        pass


class MeanSquaredError(LossFunction):
    """Mean Squared Error loss function."""

    def forward(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def backward(self, y_true, y_pred):
        return 2 * (y_pred - y_true) / y_true.shape[0]


class CrossEntropy(LossFunction):
    """Cross-entropy loss function."""

    def forward(self, y_true, y_pred):
        # Clip to prevent log(0)
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)

        if y_true.ndim == 1 or y_true.shape[1] == 1:
            # Binary classification
            return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        else:
            # Multi-class classification
            return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

    def backward(self, y_true, y_pred):
        # Clip to prevent division by 0
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)

        if y_true.ndim == 1 or y_true.shape[1] == 1:
            # Binary classification
            return -(y_true / y_pred - (1 - y_true) / (1 - y_pred)) / y_true.shape[0]
        else:
            # Multi-class classification
            return -(y_true / y_pred) / y_true.shape[0]


class BinaryCrossEntropy(LossFunction):
    """Binary Cross-entropy loss function (bonus)."""

    def forward(self, y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def backward(self, y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -(y_true / y_pred - (1 - y_true) / (1 - y_pred)) / y_true.shape[0]


# ============================================================================
# WEIGHT INITIALIZATION
# ============================================================================


class WeightInitializer:
    """Weight initialization methods."""

    @staticmethod
    def zeros(shape):
        """Initialize weights to zeros."""
        return np.zeros(shape)

    @staticmethod
    def ones(shape):
        """Initialize weights to ones."""
        return np.ones(shape)

    @staticmethod
    def random_normal(shape, mean=0, std=0.01):
        """Initialize weights from normal distribution."""
        return np.random.normal(mean, std, shape)

    @staticmethod
    def random_uniform(shape, low=-0.1, high=0.1):
        """Initialize weights from uniform distribution."""
        return np.random.uniform(low, high, shape)

    @staticmethod
    def xavier_normal(shape):
        """Xavier/Glorot normal initialization."""
        fan_in = shape[0]
        fan_out = shape[1] if len(shape) > 1 else shape[0]
        std = np.sqrt(2.0 / (fan_in + fan_out))
        return np.random.normal(0, std, shape)

    @staticmethod
    def xavier_uniform(shape):
        """Xavier/Glorot uniform initialization."""
        fan_in = shape[0]
        fan_out = shape[1] if len(shape) > 1 else shape[0]
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        return np.random.uniform(-limit, limit, shape)

    @staticmethod
    def he_normal(shape):
        """He normal initialization (good for ReLU)."""
        fan_in = shape[0]
        std = np.sqrt(2.0 / fan_in)
        return np.random.normal(0, std, shape)

    @staticmethod
    def he_uniform(shape):
        """He uniform initialization (good for ReLU)."""
        fan_in = shape[0]
        limit = np.sqrt(6.0 / fan_in)
        return np.random.uniform(-limit, limit, shape)


# ============================================================================
# OPTIMIZERS
# ============================================================================


class Optimizer(ABC):
    """Base class for optimizers."""

    @abstractmethod
    def update(self, params, gradients):
        """Update parameters."""
        pass


class SGD(Optimizer):
    """Stochastic Gradient Descent optimizer."""

    def __init__(self, learning_rate=0.01, momentum=0.0):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity = {}

    def update(self, params, gradients):
        """Update parameters using SGD with momentum."""
        for layer_id in params:
            if layer_id not in self.velocity:
                self.velocity[layer_id] = {}
                self.velocity[layer_id]["weights"] = np.zeros_like(
                    params[layer_id]["weights"]
                )
                self.velocity[layer_id]["biases"] = np.zeros_like(
                    params[layer_id]["biases"]
                )

            # Update velocity
            self.velocity[layer_id]["weights"] = (
                self.momentum * self.velocity[layer_id]["weights"]
                - self.learning_rate * gradients[layer_id]["weights"]
            )
            self.velocity[layer_id]["biases"] = (
                self.momentum * self.velocity[layer_id]["biases"]
                - self.learning_rate * gradients[layer_id]["biases"]
            )

            # Update parameters
            params[layer_id]["weights"] += self.velocity[layer_id]["weights"]
            params[layer_id]["biases"] += self.velocity[layer_id]["biases"]


class Adam(Optimizer):
    """Adam optimizer (bonus)."""

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}  # First moment
        self.v = {}  # Second moment
        self.t = 0  # Time step

    def update(self, params, gradients):
        """Update parameters using Adam."""
        self.t += 1

        for layer_id in params:
            if layer_id not in self.m:
                self.m[layer_id] = {}
                self.v[layer_id] = {}
                self.m[layer_id]["weights"] = np.zeros_like(params[layer_id]["weights"])
                self.m[layer_id]["biases"] = np.zeros_like(params[layer_id]["biases"])
                self.v[layer_id]["weights"] = np.zeros_like(params[layer_id]["weights"])
                self.v[layer_id]["biases"] = np.zeros_like(params[layer_id]["biases"])

            # Update biased first and second moment estimates
            self.m[layer_id]["weights"] = (
                self.beta1 * self.m[layer_id]["weights"]
                + (1 - self.beta1) * gradients[layer_id]["weights"]
            )
            self.m[layer_id]["biases"] = (
                self.beta1 * self.m[layer_id]["biases"]
                + (1 - self.beta1) * gradients[layer_id]["biases"]
            )

            self.v[layer_id]["weights"] = (
                self.beta2 * self.v[layer_id]["weights"]
                + (1 - self.beta2) * gradients[layer_id]["weights"] ** 2
            )
            self.v[layer_id]["biases"] = (
                self.beta2 * self.v[layer_id]["biases"]
                + (1 - self.beta2) * gradients[layer_id]["biases"] ** 2
            )

            # Bias correction
            m_hat_w = self.m[layer_id]["weights"] / (1 - self.beta1**self.t)
            m_hat_b = self.m[layer_id]["biases"] / (1 - self.beta1**self.t)
            v_hat_w = self.v[layer_id]["weights"] / (1 - self.beta2**self.t)
            v_hat_b = self.v[layer_id]["biases"] / (1 - self.beta2**self.t)

            # Update parameters
            params[layer_id]["weights"] -= (
                self.learning_rate * m_hat_w / (np.sqrt(v_hat_w) + self.epsilon)
            )
            params[layer_id]["biases"] -= (
                self.learning_rate * m_hat_b / (np.sqrt(v_hat_b) + self.epsilon)
            )


class Adagrad(Optimizer):
    """Adagrad optimizer (bonus)."""

    def __init__(self, learning_rate=0.01, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.cache = {}

    def update(self, params, gradients):
        """Update parameters using Adagrad."""
        for layer_id in params:
            if layer_id not in self.cache:
                self.cache[layer_id] = {}
                self.cache[layer_id]["weights"] = np.zeros_like(
                    params[layer_id]["weights"]
                )
                self.cache[layer_id]["biases"] = np.zeros_like(
                    params[layer_id]["biases"]
                )

            # Accumulate squared gradients
            self.cache[layer_id]["weights"] += gradients[layer_id]["weights"] ** 2
            self.cache[layer_id]["biases"] += gradients[layer_id]["biases"] ** 2

            # Update parameters
            params[layer_id]["weights"] -= (
                self.learning_rate
                * gradients[layer_id]["weights"]
                / (np.sqrt(self.cache[layer_id]["weights"]) + self.epsilon)
            )
            params[layer_id]["biases"] -= (
                self.learning_rate
                * gradients[layer_id]["biases"]
                / (np.sqrt(self.cache[layer_id]["biases"]) + self.epsilon)
            )


# ============================================================================
# NEURAL NETWORK LAYERS
# ============================================================================


class FullyConnectedLayer:
    """Fully connected (dense) layer."""

    def __init__(
        self,
        input_size,
        output_size,
        activation="relu",
        weight_init="xavier_normal",
        use_bias=True,
    ):
        """
        Initialize fully connected layer.

        Parameters:
        input_size (int): Number of input neurons
        output_size (int): Number of output neurons
        activation (str or ActivationFunction): Activation function
        weight_init (str): Weight initialization method
        use_bias (bool): Whether to use bias
        """
        self.input_size = input_size
        self.output_size = output_size
        self.use_bias = use_bias

        # Initialize weights
        if isinstance(weight_init, str):
            init_method = getattr(WeightInitializer, weight_init)
            self.weights = init_method((input_size, output_size))
        else:
            self.weights = weight_init((input_size, output_size))

        # Initialize biases
        if self.use_bias:
            self.biases = np.zeros((1, output_size))
        else:
            self.biases = None

        # Set activation function
        if isinstance(activation, str):
            activation_map = {
                "sigmoid": Sigmoid(),
                "relu": ReLU(),
                "linear": Linear(),
                "softmax": Softmax(),
                "tanh": Tanh(),
                "leaky_relu": LeakyReLU(),
            }
            self.activation = activation_map.get(activation, ReLU())
        else:
            self.activation = activation

        # Store values for backpropagation
        self.last_input = None
        self.last_pre_activation = None
        self.last_output = None

    def forward(self, x):
        """Forward pass through the layer."""
        self.last_input = x
        self.last_pre_activation = np.dot(x, self.weights)

        if self.use_bias:
            self.last_pre_activation += self.biases

        self.last_output = self.activation.forward(self.last_pre_activation)
        return self.last_output

    def backward(self, grad_output):
        """Backward pass through the layer."""
        # Gradient w.r.t. activation
        grad_activation = self.activation.backward(self.last_pre_activation)
        grad_pre_activation = grad_output * grad_activation

        # Gradients w.r.t. weights and biases
        grad_weights = np.dot(self.last_input.T, grad_pre_activation)
        grad_biases = (
            np.sum(grad_pre_activation, axis=0, keepdims=True)
            if self.use_bias
            else None
        )

        # Gradient w.r.t. input (for previous layer)
        grad_input = np.dot(grad_pre_activation, self.weights.T)

        return grad_input, grad_weights, grad_biases


# ============================================================================
# MAIN NEURAL NETWORK CLASS
# ============================================================================


class NeuralNetwork(BaseEstimator, ClassifierMixin):
    """
    Feedforward Neural Network with customizable architecture.
    """

    def __init__(
        self,
        hidden_layers=None,
        activations=None,
        output_activation="linear",
        loss_function="mse",
        optimizer="sgd",
        learning_rate=0.01,
        epochs=100,
        batch_size=32,
        regularization=None,
        reg_strength=0.01,
        weight_init="xavier_normal",
        random_state=None,
        verbose=False,
    ):
        """
        Initialize Neural Network.

        Parameters:
        hidden_layers (list): List of hidden layer sizes
        activations (list): List of activation functions for hidden layers
        output_activation (str): Output layer activation
        loss_function (str): Loss function ('mse', 'cross_entropy', 'binary_cross_entropy')
        optimizer (str or Optimizer): Optimizer ('sgd', 'adam', 'adagrad')
        learning_rate (float): Learning rate
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        regularization (str): Regularization type ('l1', 'l2', None)
        reg_strength (float): Regularization strength
        weight_init (str): Weight initialization method
        random_state (int): Random seed
        verbose (bool): Print training progress
        """
        if activations is None:
            activations = ["relu", "relu"]
        if hidden_layers is None:
            hidden_layers = [64, 32]
        self.hidden_layers = hidden_layers
        self.activations = activations
        self.output_activation = output_activation
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.regularization = regularization
        self.reg_strength = reg_strength
        self.weight_init = weight_init
        self.random_state = random_state
        self.verbose = verbose

        self.layers = []
        self.loss_history = []
        self.n_features = None
        self.n_outputs = None
        self.classes = None

        if random_state is not None:
            np.random.seed(random_state)

    def _build_network(self, input_size, output_size):
        """Build the neural network architecture."""
        self.layers = []

        # Input to first hidden layer
        if self.hidden_layers:
            layer_sizes = [input_size, *self.hidden_layers, output_size]
            layer_activations = [*self.activations, self.output_activation]
        else:
            layer_sizes = [input_size, output_size]
            layer_activations = [self.output_activation]

        # Create layers
        for i in range(len(layer_sizes) - 1):
            activation = (
                layer_activations[i] if i < len(layer_activations) else "linear"
            )
            layer = FullyConnectedLayer(
                layer_sizes[i],
                layer_sizes[i + 1],
                activation=activation,
                weight_init=self.weight_init,
            )
            self.layers.append(layer)

    def _get_loss_function(self):
        """Get loss function object."""
        loss_map = {
            "mse": MeanSquaredError(),
            "cross_entropy": CrossEntropy(),
            "binary_cross_entropy": BinaryCrossEntropy(),
        }
        return loss_map.get(self.loss_function, MeanSquaredError())

    def _get_optimizer(self):
        """Get optimizer object."""
        if isinstance(self.optimizer, str):
            optimizer_map = {
                "sgd": SGD(learning_rate=self.learning_rate),
                "adam": Adam(learning_rate=self.learning_rate),
                "adagrad": Adagrad(learning_rate=self.learning_rate),
            }
            return optimizer_map.get(
                self.optimizer, SGD(learning_rate=self.learning_rate)
            )
        else:
            return self.optimizer

    def _prepare_target(self, y):
        """Prepare target labels for training."""
        y = np.array(y)

        # For classification tasks, encode labels
        if self.loss_function in ["cross_entropy", "binary_cross_entropy"]:
            self.classes = np.unique(y)

            if len(self.classes) == 2:
                # Binary classification
                y_encoded = (y == self.classes[1]).astype(float)
                if self.output_activation == "softmax":
                    # Convert to one-hot for softmax
                    y_one_hot = np.zeros((len(y), 2))
                    y_one_hot[np.arange(len(y)), y_encoded.astype(int)] = 1
                    return y_one_hot
                else:
                    return y_encoded.reshape(-1, 1)
            else:
                # Multi-class classification - one-hot encode
                y_one_hot = np.zeros((len(y), len(self.classes)))
                for i, class_label in enumerate(self.classes):
                    y_one_hot[y == class_label, i] = 1
                return y_one_hot
        else:
            # Regression
            return y.reshape(-1, 1) if y.ndim == 1 else y

    def forward(self, X):
        """Forward pass through the network."""
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def backward(self, grad_output):
        """Backward pass through the network."""
        gradients = {}

        # Backward pass through layers
        grad = grad_output
        for i, layer in enumerate(reversed(self.layers)):
            layer_id = len(self.layers) - 1 - i
            grad, grad_weights, grad_biases = layer.backward(grad)

            # Store gradients
            gradients[layer_id] = {"weights": grad_weights, "biases": grad_biases}

        return gradients

    def _add_regularization(self, gradients):
        """Add regularization to gradients."""
        if self.regularization is None:
            return gradients

        for layer_id in gradients:
            weights = self.layers[layer_id].weights

            if self.regularization == "l1":
                gradients[layer_id]["weights"] += self.reg_strength * np.sign(weights)
            elif self.regularization == "l2":
                gradients[layer_id]["weights"] += self.reg_strength * 2 * weights

        return gradients

    def _compute_regularization_loss(self):
        """Compute regularization loss."""
        if self.regularization is None:
            return 0

        reg_loss = 0
        for layer in self.layers:
            if self.regularization == "l1":
                reg_loss += self.reg_strength * np.sum(np.abs(layer.weights))
            elif self.regularization == "l2":
                reg_loss += self.reg_strength * np.sum(layer.weights**2)

        return reg_loss

    def fit(self, X, y):
        """
        Train the neural network.

        Parameters:
        X (array-like): Training features
        y (array-like): Training targets

        Returns:
        self: Returns self for method chaining
        """
        X = np.array(X)
        y = np.array(y)

        self.n_features = X.shape[1]

        # Prepare targets
        y_encoded = self._prepare_target(y)
        self.n_outputs = y_encoded.shape[1]

        # Build network
        self._build_network(self.n_features, self.n_outputs)

        # Get loss function and optimizer
        loss_fn = self._get_loss_function()
        optimizer = self._get_optimizer()

        # Prepare parameters for optimizer
        params = {}
        for i, layer in enumerate(self.layers):
            params[i] = {"weights": layer.weights, "biases": layer.biases}

        self.loss_history = []
        n_samples = X.shape[0]

        # Training loop
        for epoch in range(self.epochs):
            epoch_loss = 0
            n_batches = 0

            # Mini-batch training
            for start_idx in range(0, n_samples, self.batch_size):
                end_idx = min(start_idx + self.batch_size, n_samples)
                X_batch = X[start_idx:end_idx]
                y_batch = y_encoded[start_idx:end_idx]

                # Forward pass
                predictions = self.forward(X_batch)

                # Compute loss
                batch_loss = loss_fn.forward(y_batch, predictions)
                batch_loss += self._compute_regularization_loss()
                epoch_loss += batch_loss
                n_batches += 1

                # Backward pass
                grad_output = loss_fn.backward(y_batch, predictions)
                gradients = self.backward(grad_output)

                # Add regularization to gradients
                gradients = self._add_regularization(gradients)

                # Update weights
                optimizer.update(params, gradients)

            # Average loss for epoch
            avg_loss = epoch_loss / n_batches
            self.loss_history.append(avg_loss)

            if self.verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {avg_loss:.6f}")

        return self

    def predict(self, X):
        """Make predictions."""
        X = np.array(X)
        predictions = self.forward(X)

        if hasattr(self, "classes") and self.classes is not None:
            # Classification
            if len(self.classes) == 2 and predictions.shape[1] == 1:
                # Binary classification
                return self.classes[(predictions.flatten() > 0.5).astype(int)]
            else:
                # Multi-class classification
                return self.classes[np.argmax(predictions, axis=1)]
        else:
            # Regression
            return predictions.flatten() if predictions.shape[1] == 1 else predictions

    def predict_proba(self, X):
        """Predict class probabilities (for classification)."""
        if not hasattr(self, "classes") or self.classes is None:
            raise ValueError("predict_proba is only available for classification tasks")

        X = np.array(X)
        predictions = self.forward(X)

        if self.output_activation in ["sigmoid", "softmax"]:
            return predictions
        else:
            # Apply softmax if not already applied
            softmax = Softmax()
            return softmax.forward(predictions)
