import pennylane as qml
import jax
import jax.numpy as jnp
from jax import lax, random
import optax
import matplotlib.pyplot as plt
from tqdm import trange
import numpy as np
from visualization import plot_probs, plot_hidden_state

# --- Basic settings
############################### <Adjustable Parameters> ##############################
n_D = 3  # Number of qubits in Register D" (Stores input data)
depth = 50  # Depth of Ansatz
total_epoch = 100  # Number of training epochs
############################### <Adjustable Parameters> ##############################

n_H = n_D  # Number of qubits in Register H (Hidden)
n_qubits = n_D + n_H  # Number of total qubits
n_able_bits = 2 ** n_D # Number of possible input states

# --- (Input / Target) Datasets
X_data = np.array([i for i in range(n_able_bits)])  # Input data (0 to n_D ^2)
Y_data = np.array([(i + 1) % n_able_bits for i in range(n_able_bits)])  # Target data

# --- Encoding
def int_to_bits_single(x, n_bits=n_D):
    powers = 2 ** jnp.arange(n_bits - 1, -1, -1)  # Create array of powers of two, descending from (n_bits - 1) to 0
    bits = (jnp.floor_divide(x, powers) % 2).astype(jnp.int32)  # Extract each bits of x as binary vector
    return bits
    
# --- Wires
wires = list(range(n_qubits))
data_wires = wires[0:n_D]
hidden_wires = wires[n_D:n_qubits]

# ---  Ansatz
def ansatz(params, wires):
    num_wires = len(wires)
    for d in range(depth):
        for i in range(num_wires):
            qml.RX(params["rot"][d, i, 0], wires=wires[i])
            qml.RZ(params["rot"][d, i, 1], wires=wires[i])
            qml.RX(params["rot"][d, i, 2], wires=wires[i])
        for i in range(n_D):
            q0 = data_wires[i]
            q1 = data_wires[(i + 1) % n_D]
            qml.IsingZZ(params["ent"][d, i], wires=[q0, q1])
        for i in range(n_H):
            q0 = hidden_wires[i]
            q1 = hidden_wires[(i + 1) % n_H]
            qml.IsingZZ(params["ent"][d, n_D + i], wires=[q0, q1])
        for i in range(n_D):
            q0 = data_wires[i]
            q1 = hidden_wires[i]
            qml.IsingZZ(params["ent"][d, n_D + n_H + i], wires=[q0, q1])

# --- QNode
dev = qml.device("default.qubit", wires=wires)

@qml.qnode(dev, interface="jax", diff_method="backprop")
def QRB_circuit(params, hidden_state, x):
    """
    1. hidden_state -> Reg.H 초기화 (StatePrep)
    2. x -> 이진변환 후 데이터 레지스터 RY 인코딩
    3. ansatz 적용
    4. 전체 상태 qml.state() 반환
    """
    qml.StatePrep(hidden_state, wires=hidden_wires)
    bits = int_to_bits_single(x, n_bits=n_D)
    for i, bit in enumerate(bits):
        qml.RY(jnp.pi * bit, wires=data_wires[i])
    ansatz(params["ansatz"], wires=wires)
    full_state = qml.state()
    return full_state

def QRB(hidden_state, x, params, key):
    """
    전체 상태를 받아 직접 확률 계산 (diagonal 방식).
    """
    state_full = QRB_circuit(params, hidden_state, x)  # full_state만 반환
    state_reshaped = jnp.reshape(state_full, (2**n_H, 2**n_D))
    # Diagonal 방식으로 Reg.D의 확률분포 계산
    p_calc = jnp.diag(state_reshaped.T @ jnp.conj(state_reshaped)).real
    plot_probs(p_calc, title=f"Input {x} - Reg.D Output Prob")
    # 샘플링
    out = jax.random.choice(key, a=jnp.arange(2**n_D), p=p_calc)
    # collapse된 hidden state
    new_hidden_state = state_reshaped[:, out]
    # Plot hidden state
    plot_hidden_state(new_hidden_state, title=f"Updated Hidden State after Input {x}")

    new_hidden_state = new_hidden_state / jnp.linalg.norm(new_hidden_state)
    return new_hidden_state, out, p_calc

def pQRNN(params, input_seq, key):
    """
    각 sequence마다 QRB를 호출하여 hidden state를 업데이트
    각 단계의 측정된 outcome을 수집하여 반환
    """
    h_state = jnp.zeros(2**n_H, dtype=jnp.complex64).at[0].set(1.0+0.0j)
    outcomes = []
    ##$ 매 time step마다 사용할 random key split
    keys = jax.random.split(key, num=len(input_seq))
    for i, x in enumerate(input_seq):
        h_state, out, _ = QRB(h_state, x, params, keys[i])   ##$ key 전달
        outcomes.append(out)
    return outcomes
    
# --- Loss function
def loss_fn(params, x, y, key):
    # l000>
    init_hidden = jnp.zeros(2**n_H, dtype=jnp.complex64).at[0].set(1.0 + 0.0j)
    # return full_state
    state_full = QRB_circuit(params, init_hidden, x)
    # reshape: Reg.H × Reg.D
    state_reshaped = jnp.reshape(state_full, (2**n_H, 2**n_D))  ##$
    # Diagonal p
    probs = jnp.diag(state_reshaped.T @ jnp.conj(state_reshaped)).real  ##$
    probs = jnp.clip(probs, 1e-8, 1.0)
    # cross-entropy loss
    return -jnp.log(probs[y])

v_loss_fn = jax.vmap(loss_fn, in_axes=(None, 0, 0, None))

@jax.jit
def total_loss(params, X, Y, key):
    return jnp.mean(v_loss_fn(params, X, Y, key))

# --- Optimizer & Initialization
learning_rate = 0.01
optimizer = optax.adam(learning_rate)

key2 = random.PRNGKey(42)  ##$ 하나의 key 사용
rot_init = random.uniform(key2, (depth, n_qubits, 3), minval=0.0, maxval=2*jnp.pi)
key2, subkey = random.split(key2)  ##$ 계속 key2 사용하며 분할
ent_init = random.uniform(subkey, (depth, n_qubits), minval=0.0, maxval=2*jnp.pi)
params = {"ansatz": {"rot": rot_init, "ent": ent_init}}
opt_state = optimizer.init(params)

# --- Train step
@jax.jit
def train_step(params, opt_state, X, Y, key):
    loss, grads = jax.value_and_grad(total_loss)(params, X, Y, key)
    T_max = 100
    min_lr = 0.0001
    current_lr = min_lr + (learning_rate - min_lr) * 0.5 * (1 + jnp.cos(jnp.pi*(0 % T_max)/T_max))  ##$ 여기선 상수값 사용
    updates, opt_state = optimizer.update(grads, opt_state, params)
    updates = jax.tree_util.tree_map(lambda u: current_lr*(u/current_lr), updates)
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state, loss
