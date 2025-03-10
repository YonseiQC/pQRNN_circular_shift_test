import jax
import jax.numpy as jnp
import optax
import pennylane as qml
from tqdm import trange
import matplotlib.pyplot as plt

# --- Basic settings
############################### <Adjustable Parameters> ##############################
n_D = 3  # Number of qubits in Register D" (Stores input data)
depth = 3  # Depth of Ansatz
num_epochs = 100  # Number of training epochs
############################### <Adjustable Parameters> ##############################

n_H = n_D  # Number of qubits in Register H (Hidden)
n_qubits = n_D + n_H  # Number of total qubits
n_able_bits = 2 ** n_D # Number of possible input states

# --- (Input / Target) Datasets
X_data = jnp.array([i for i in range(n_able_bits)])  # Input data (0 to n_D ^2)
Y_data = jnp.array([(i + 1) % (n_able_bits) for i in range(n_able_bits)])  # Target data

# --- Convert integer to binary vector
def int_to_bits_single(x, n_bits=n_D):
    powers = 2 ** jnp.arange(n_bits - 1, -1, -1)  # Create array of powers of two, descending from (n_bits - 1) to 0
    bits = (jnp.floor_divide(x, powers) % 2).astype(jnp.int32)  # Extract each bits of x as binary vector
    return bits

# --- (NN / CB) Ansatz
def NN_ansatz(params, wires): # Nearest-Neighbor(NN) Parameterized Quantum Circuit
    num_wires = len(wires)  # Number of wires (qubits) used
    for d in range(depth):  # Repeat for the designated depth
        for i in range(num_wires):  # Rotation gates for each qubit
            qml.RX(params["rot"][d, i, 0], wires=wires[i])
            qml.RZ(params["rot"][d, i, 1], wires=wires[i])
            qml.RX(params["rot"][d, i, 2], wires=wires[i])  # Apply RX-RZ-RX for each qubit
        for i in range(num_wires):  # Entanglement gates (RZZ) in a circular pattern
            qml.IsingZZ(params["ent"][d, i], wires=[wires[i], wires[(i + 1) % num_wires]])

def CB_ansatz(params, wires, block_size=n_D): # Circuit-Block (CB) Parameterized Quantum Circuit
    num_wires = len(wires)  # Number of wires (qubits) used
    for d in range(depth):  # Repeat for the designated depth
        for i in range(num_wires):  # Rotation gates for each qubit
            qml.RX(params["rot"][d, i, 0], wires=wires[i])
            qml.RZ(params["rot"][d, i, 1], wires=wires[i])
            qml.RX(params["rot"][d, i, 2], wires=wires[i])
        # 1st block: Reg.D / 2nd block: Reg.H
        # CB (Circuit-Block) entanglement structure
        for i in range(0, num_wires, block_size):
            block_wires = wires[i:i + block_size]
            for j in range(len(block_wires) - 1): # Inner entanglement within each block
                qml.IsingZZ(params["ent"][d, i + j], wires=[block_wires[j], block_wires[j + 1]])
            # Use the code below for additional entanglement
            """
            for i in range(0, num_wires, block_size): # Additional entanglement between blocks
              qml.IsingZZ(params["ent"][d, i], wires=[wires[i], wires[(i + block_size) % num_wires]])
            """

# --- QNode
dev = qml.device("lightning.qubit", wires=n_qubits) # Initialization

@qml.qnode(dev, interface="jax", diff_method="parameter-shift") # Define QNode
def QRB_circuit(params, x): # Quantum Recurrent Block
    bits = int_to_bits_single(x, n_bits=n_D)  # Input x(int) to a binary vector
    for i, bit in enumerate(bits):  # Initialize each input qubit
        qml.RY(jnp.pi * bit, wires=i)  # Apply RY gate
    CB_ansatz(params["ansatz"], wires=range(n_qubits))  # Choose between 'NN_ansatz' and 'CB_ansatz'
    #NN_ansatz(params["ansatz"], wires=range(n_qubits))
    return qml.probs(wires=range(n_D))  # Return Probability of the input qubit

# --- Prediction
def predict_class(probs):
    return int(jnp.argmax(probs))  # Return class with the highest probability as 'prediction'

# --- Cross entropy loss function
def loss_fn(params, x, y):  # Loss Func.
    probs = QRB_circuit(params, x)  # Calculate probabilities
    probs = jnp.clip(probs, 1e-8, 1.0)  # Clipping (log stability)
    return -jnp.log(probs[y])  # Return negative log probability of the target class

v_loss_fn = jax.vmap(loss_fn, in_axes=(None, 0, 0)) # Vectorization of loss func. for batch processing

@jax.jit
def batch_loss(params, X, Y): # Batch Loss Func.
    losses = v_loss_fn(params, X, Y)  # Losses of each sample
    return jnp.mean(losses)  # Return mean loss

# --- Optimizer & Initialization
learning_rate = 0.01  # Learning rate
optimizer = optax.adam(learning_rate)  # Adam optimizer

key = jax.random.PRNGKey(42)  # Random seed for replicability
rot_init = jax.random.uniform(key, (depth, n_qubits, 3), minval=0.0, maxval=2*jnp.pi) # Rot initialization
key, subkey = jax.random.split(key)  # Generate new random seed
ent_init = jax.random.uniform(subkey, (depth, n_qubits), minval=0.0, maxval=2*jnp.pi) # Ent initialization

params = {"ansatz": {"rot": rot_init, "ent": ent_init}} # Initial params
opt_state = optimizer.init(params)  # Optimizer state initializatiion

# --- Train step
@jax.jit
def train_step(params, opt_state, X, Y):
    loss, grads = jax.value_and_grad(batch_loss)(params, X, Y)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss
