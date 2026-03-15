import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs('figures', exist_ok=True)

# ── Paste your numbers here ───────────────────────────────────────────────────

# Epoch 0 is the baseline (before any gradient steps)
# DPO — copy from your terminal
dpo_success  = [82.67, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]

dpo_pref_acc = [None, 89.40, 95.48, 97.05, 97.54, 97.80, 97.93, 98.07, 98.15, 98.23, 98.30, 98.35, 98.46, 98.48, 98.51, 98.52, 98.55, 98.56, 98.57, 98.56, 98.57]

dpo_log_pi_w = [-1.307, -62.285, -131.545, -200.434, -278.403, -305.961, -340.062, -397.441, -442.295, -470.929, -495.769, -569.219, -571.339, -610.729, -620.222, -662.625, -641.249, -663.321, -711.955, -718.243, -657.835]

# DPOP — copy from your terminal
dpop_success = [82.67, 81.33, 82.67, 80.67, 79.33, 84.67, 81.33, 86.00, 86.00, 77.33, 78.67, 80.00, 82.00, 80.00, 79.33, 78.00, 78.67, 79.33, 78.00, 79.33, 78.67]

dpop_pref_acc = [None, 89.08, 91.50, 91.57, 91.35, 91.06, 90.47, 90.01, 89.76, 89.72, 89.47, 89.45, 89.40, 89.50, 89.45, 89.33, 89.48, 89.54, 89.59, 89.59, 89.56]

dpop_log_pi_w = [-1.307, -1.207, -1.174, -1.215, -1.191, -1.185, -1.224, -1.213, -1.214, -1.197, -1.257, -1.213, -1.214, -1.232, -1.211, -1.216, -1.213, -1.200, -1.189, -1.272, -1.233]

BC_SUCCESS = 82.67  # your held-out eval number

epochs = list(range(len(dpo_success)))  # [0, 1, 2, ..., 20]

# ── Figure 1: Training Dynamics (Dissociation Plot) ───────────────────────────
fig, ax1 = plt.subplots(figsize=(8, 5))

# Success rate — left axis
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

ax1.set_title('Training Dynamics in Peg-Insert-Side: DPO vs DPOP', fontsize=13)
ax1.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
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

ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('log π(w)  [chosen trajectory log-prob]', fontsize=12)
ax.set_xlim(0, 20)
ax.set_title('Likelihood Displacement in Peg-Insert-Side: DPO vs DPOP', fontsize=13)
ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figures/likelihood_displacement.png', dpi=150, bbox_inches='tight')
plt.show()