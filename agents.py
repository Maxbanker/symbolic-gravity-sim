"""
agents.py
Defines recursive symbolic agents navigating Ψeff fields
"""

import numpy as np

class SymbolicAgent:
    def __init__(self, position, learning_rate=0.1):
        self.position = np.array(position, dtype=float)
        self.history = [tuple(position)]
        self.learning_rate = learning_rate
        self.collapsed = False

    def update(self, psi_eff_field):
        if self.collapsed:
            return

        x, y = self.position.astype(int)
        grad_x, grad_y = np.gradient(psi_eff_field)
        force = -np.array([grad_x[x, y], grad_y[x, y]])

        self.position += self.learning_rate * force
        self.position = np.clip(self.position, [0, 0], np.array(psi_eff_field.shape) - 1)
        self.history.append(tuple(self.position))

        if psi_eff_field[int(self.position[0]), int(self.position[1])] < 0.1:
            self.collapsed = True
