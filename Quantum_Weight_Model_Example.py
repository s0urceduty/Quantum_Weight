import numpy as np

class QuantumWeightModel:
    def __init__(self, qubit_layout, coherence_map, entanglement_map, control_roles):
        """
        Initialize the quantum weight model.

        :param qubit_layout: dict mapping qubit ID to spatial coordinates
        :param coherence_map: dict mapping qubit ID to coherence times
        :param entanglement_map: dict mapping qubit pairs to entanglement strength
        :param control_roles: set of qubit IDs acting as control nodes
        """
        self.qubit_layout = qubit_layout
        self.coherence_map = coherence_map
        self.entanglement_map = entanglement_map
        self.control_roles = control_roles
        self.weights = {}

    def compute_proximity(self, q1, q2):
        """Euclidean distance between two qubits."""
        pos1, pos2 = self.qubit_layout[q1], self.qubit_layout[q2]
        return np.linalg.norm(np.array(pos1) - np.array(pos2))

    def calculate_weight(self, qubit_id):
        """Compute dynamic weight for a given qubit."""
        proximity_score = sum(1 / self.compute_proximity(qubit_id, other) 
                              for other in self.qubit_layout if other != qubit_id)
        coherence_score = self.coherence_map.get(qubit_id, 0)
        entanglement_score = sum(v for (q1, q2), v in self.entanglement_map.items()
                                 if qubit_id in (q1, q2))
        control_bonus = 1.5 if qubit_id in self.control_roles else 1.0
        
        raw_score = (proximity_score + entanglement_score) * coherence_score * control_bonus
        return raw_score

    def update_weights(self):
        """Recalculate weights for all qubits."""
        max_score = 0
        for qubit in self.qubit_layout:
            self.weights[qubit] = self.calculate_weight(qubit)
            max_score = max(max_score, self.weights[qubit])

        # Normalize weights
        if max_score > 0:
            for qubit in self.weights:
                self.weights[qubit] /= max_score

    def get_weights(self):
        """Retrieve current weights."""
        return self.weights
