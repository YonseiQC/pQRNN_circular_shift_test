import jax
import jax.numpy as jnp
from jax import lax, random
import matplotlib.pyplot as plt
from tqdm import trange
from model import n_able_bits, total_epoch, X_data, Y_data, pQRNN, train_step, params, opt_state

# --- Train loop
loss_history = []
key3 = random.PRNGKey(100)  ##$ 하나의 key로 관리
for ep in trange(total_epoch, desc="Training"):
    key3, subkey = random.split(key3)  ##$ 매 epoch마다 key 분할
    params, opt_state, loss_val = train_step(params, opt_state, X_data, Y_data, subkey)
    loss_history.append(float(loss_val))
    if (ep+1) % 5 == 0:
        print(f"Epoch {ep+1}, Loss = {loss_val:.4f}")

# --- Plot (loss)
plt.figure(figsize=(8, 4))
plt.plot(range(1, total_epoch+1), loss_history, linestyle='-')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss per Epoch')
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Results
h_state = jnp.zeros(2**n_H, dtype=jnp.complex64).at[0].set(1.0+0.0j)
current_input = 0
print("Input -> Output (Target)")
for _ in range(n_able_bits):
    key3, subkey = random.split(key3)
    outcomes = pQRNN(params, jnp.array([current_input]), subkey)
    pred = int(outcomes[-1])
    target = (current_input + 1) % n_able_bits
    print(f"Input: {current_input}, Predicted: {pred}, Target: {target}")
    current_input = pred
