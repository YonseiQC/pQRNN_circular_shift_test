import matplotlib.pyplot as plt
from tqdm import trange
from model import n_able_bits, num_epochs, X_data, Y_data, QRB_circuit, predict_class, train_step, params, opt_state

# --- Train loop
loss_history = []  # List for losses

with trange(num_epochs) as pbar:  # Progress bar
    for epoch in pbar:  # Train loop
        params, opt_state, loss = train_step(params, opt_state, X_data, Y_data)  # Single step of training
        loss_value = float(loss)  # Loss to float
        loss_history.append(loss_value)  # Loss logging
        pbar.set_postfix({"Loss": f"{loss_value:.4f}"})  # Show loss on progress bar

# --- Plot (loss)
plt.figure(figsize=(8, 4))
plt.plot(range(1, num_epochs + 1), loss_history, linestyle='-')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss per Epoch')
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Results
current_input = 0  # Initial input

for _ in range(n_able_bits):  # For the number of possible input states
    probs = QRB_circuit(params, current_input)  # Probability
    pred = predict_class(probs)  # Prediction
    target = (current_input + 1) % (n_able_bits)  # Target class
    print(f"Input: {current_input}, Predicted: {pred}, Target: {target}")  # Final Results
    current_input = pred  # Next input = Current output
