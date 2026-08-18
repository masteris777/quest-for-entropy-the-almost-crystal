import numpy as np

class ResonatorBank:
    """
    A fixed bank of harmonic resonators for lock-in detection of quantum phases.
    Uses simple Euler integration for fast long-time simulation.
    """
    def __init__(self, frequencies, gamma, coupling=1.0):
        """
        frequencies: list of resonance frequencies (omega_a)
        gamma: damping bandwidth
        coupling: coupling constant g
        """
        self.omegas = np.array(frequencies, dtype=float)
        self.gamma = gamma
        self.coupling = coupling
        self.z = np.zeros(len(self.omegas), dtype=complex)
        self.energy_accumulator = np.zeros(len(self.omegas), dtype=float)
        self.t_total = 0.0

    def reset(self):
        self.z[:] = 0.0
        self.energy_accumulator[:] = 0.0
        self.t_total = 0.0

    def step(self, signal, dt):
        """
        Exact local step assuming signal is constant over dt.
        dz/dt = A z + coupling * signal, A = -gamma - i omega_a
        To match S(t) ~ e^{-i omega_a t}, we need -i omega_a
        z(t+dt) = e^{A dt} z(t) + (e^{A dt} - 1)/A * coupling * signal
        """
        A = -self.gamma - 1j * self.omegas
        expA = np.exp(A * dt)
        
        self.z = expA * self.z + ((expA - 1.0) / A) * self.coupling * signal
        self.energy_accumulator += np.abs(self.z)**2 * dt
        self.t_total += dt
        
    def run_signal(self, time_array, signal_array):
        """
        Fast vectorize-like execution if dt is constant.
        """
        dt = time_array[1] - time_array[0]
        
        for s in signal_array:
            self.step(s, dt)
            
    def get_born_weights(self, spatial_correction=None):
        """
        Returns normalized time-averaged energy.
        spatial_correction: optional array of |psi_a(x_det)|^2. We divide by it to get true weights.
        """
        if self.t_total == 0:
            return np.zeros_like(self.omegas)
        avg_energy = self.energy_accumulator / self.t_total
        
        weights = avg_energy
        if spatial_correction is not None:
            # avoid division by zero
            eps = 1e-15
            weights = avg_energy / (spatial_correction + eps)
            
        total_w = np.sum(weights)
        if total_w < 1e-15:
            return weights
        return weights / total_w
