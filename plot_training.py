import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs('figures', exist_ok=True)

# ── Paste your numbers here ───────────────────────────────────────────────────

# Epoch 0 is the baseline (before any gradient steps)
# DPO — copy from your terminal
dpo_success  = [92.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

dpo_pref_acc = [None, 83.61, 85.83, 87.12, 89.36, 91.14, 93.10, 94.77, 96.15, 96.67, 97.28, 97.64, 97.78, 97.83, 97.89, 97.96, 97.96, 97.98, 97.95, 98.03]

dpo_log_pi_w = [-8.051, -28.503, -50.412, -68.897, -78.408, -91.105, -103.869, -110.059, -119.917, -112.436, -132.025, -125.223, -128.193, -143.884, -150.426, -140.595, -132.190, -136.981, -141.445, -142.552]

# DPOP — copy from your terminal
dpop_success = [88.0, 96.0, 92.0, 90.0, 94.0, 92.0, 90.0, 96.0, 94.0, 88.0, 92.0, 88.0, 92.0, 94.0, 96.0, 94.0, 90.0, 100.0, 92.0, 86.0]

dpop_pref_acc = [None, 79.12, 88.03, 87.61, 89.70, 90.25, 90.18, 90.74, 90.93, 91.22, 91.80, 91.98, 92.27, 92.15, 92.35, 92.23, 92.21, 92.29, 92.15, 92.33]

dpop_log_pi_w = [-1.275, -1.266, -1.256, -1.259, -1.252, -1.252, -1.230, -1.227, -1.262, -1.240, -1.267, -1.258, -1.271, -1.260, -1.251, -1.255, -1.248, -1.235, -1.231, -1.245]

BC_SUCCESS = 92.75  # your held-out eval number

epochs = list(range(len(dpo_success)))  # [0, 1, 2, ..., 20]

# ── Figure 1: Training Dynamics (Dissociation Plot) ───────────────────────────
fig, ax1 = plt.subplots(figsize=(8, 5))

# Success rate — left axis
ax1.axhline(BC_SUCCESS, color='gray', linestyle='--', linewidth=1.5,
            label=f'BC baseline ({BC_SUCCESS:.1f}%)')
ax1.plot(epochs, dpo_success,  color='tomato',      linewidth=2, marker='o',
         markersize=4, label='DPO success rate')
ax1.plot(epochs, dpop_success, color='steelblue',   linewidth=2, marker='s',
         markersize=4, label='DPOP success rate')
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Success Rate (%)', fontsize=12, color='black')
ax1.set_ylim(-5, 105)

# Preference accuracy — right axis
ax2 = ax1.twinx()
pref_epochs = [e for e, v in zip(epochs, dpo_pref_acc) if v is not None]
dpo_pref_vals  = [v for v in dpo_pref_acc  if v is not None]
dpop_pref_vals = [v for v in dpop_pref_acc if v is not None]

ax2.plot(pref_epochs, dpo_pref_vals,  color='tomato',    linewidth=2,
         linestyle='--', alpha=0.6, label='DPO pref_acc')
ax2.plot(pref_epochs, dpop_pref_vals, color='steelblue', linewidth=2,
         linestyle='--', alpha=0.6, label='DPOP pref_acc')
ax2.set_ylabel('Preference Accuracy (%)', fontsize=12, color='gray')
ax2.set_ylim(50, 105)

# Combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right', fontsize=10)

ax1.set_title('Training Dynamics: DPO vs DPOP', fontsize=13)
ax1.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figures/training_dynamics.png', dpi=150, bbox_inches='tight')
plt.show()

# ── Figure 2: Likelihood Displacement ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(epochs, dpo_log_pi_w,  color='tomato',    linewidth=2, marker='o',
        markersize=4, label='DPO  log π(w)')
ax.plot(epochs, dpop_log_pi_w, color='steelblue', linewidth=2, marker='s',
        markersize=4, label='DPOP log π(w)')
ax.axhline(dpo_log_pi_w[0], color='gray', linestyle='--', linewidth=1.5,
           label=f'BC reference ({dpo_log_pi_w[0]:.2f})')

ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('log π(w)  [chosen trajectory log-prob]', fontsize=12)
ax.set_title('Likelihood Displacement: DPO vs DPOP', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figures/likelihood_displacement.png', dpi=150, bbox_inches='tight')
plt.show()