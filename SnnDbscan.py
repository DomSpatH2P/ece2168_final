import numpy as np
from scipy.ndimage import label, center_of_mass, convolve


global t 

class SnnDbscan: # Neuromorphic DBSCAN implementing Rizzo & Plank 2024's systolic algorithm
    
    def __init__(self, R, C, eps, min_pts):
        self.R = R
        self.C = C
        self.eps = eps
        self.min_pts = min_pts
        
        # Neighborhood kernel: (2eps+1) * (2eps+1), all ones, center zeroed
        # Used for counting neighbor spikes excluding self
        size = 2 * eps + 1
        self.kernel = np.ones((size, size), dtype=np.uint8)
        self.kernel[eps, eps] = 0
        
        self.reset()
    
    def reset(self): #reset DBSCAN
        eps, R = self.eps, self.R
        size = 2 * eps + 1
        #Because the neurons have discrete voltages and thresholds and zero leaks, we interpret them as arrays for less overhead

        # Shift-register buffers for I and Core neurons
        self.i_buf    = np.zeros((R, size), dtype=np.uint8)
        self.core_buf = np.zeros((R, size), dtype=np.uint8)
        
        # delay = 1 buffers
        self.delay_c_to_core   = np.zeros(R, dtype=np.uint8)  # C → Core
        self.delay_b_to_border = np.zeros(R, dtype=np.uint8)  # B → Border
        
        # Multi-step FIFO delay buffers
        self.delay_i_to_core      = np.zeros((R, 2), dtype=np.uint8)  # delay 2
        self.delay_core_to_border = np.zeros((R, 2), dtype=np.uint8)  # delay 2
        self.delay_i_to_border    = np.zeros((R, 4), dtype=np.uint8)  # delay 4
    
    def _conv_neighborhood(self, buf): #convolve buffer, return center
        result = convolve(buf, self.kernel, mode='constant')
        return result[:, self.eps]
    
    def _fifo_step(self, fifo, new_input): #add new value to queue, return value at front
        arriving = fifo[:, 0].copy()
        fifo[:, :-1] = fifo[:, 1:]
        fifo[:, -1] = new_input
        return fifo[:,0] #arriving
    
    def step(self, input_column): #run pipeline in reverse so that old calculations don't automatically update new calculations
        eps = self.eps
        
        #6. Border (downstream-most): reads I_{r,-eps}, B, Core_{r,0}
        # All three sources are still in their previous-timestep state
        i_to_border_now    = self._fifo_step(self.delay_i_to_border,
                                              self.i_buf[:, 0])
        core_to_border_now = self._fifo_step(self.delay_core_to_border,
                                              self.core_buf[:, eps])
        b_to_border_now    = self.delay_b_to_border.copy()
        
        border_input  = i_to_border_now + b_to_border_now - core_to_border_now
        border_spikes = (border_input >= 2).astype(np.uint8)
        
        #5. B: convolution over Core buffer (still previous timestep)
        b_input = self._conv_neighborhood(self.core_buf)
        b_spikes_now = (b_input >= 1).astype(np.uint8)
        # Save for next step's Border calculation
        self.delay_b_to_border = b_spikes_now.copy()
        
        #3. Core_{r,eps}: I_{r,0} (delay 2) + C (delay 1)
        # Compute now so we have core_eps_spikes before we shift the Core buffer.
        i_to_core_now = self._fifo_step(self.delay_i_to_core,
                                         self.i_buf[:, eps])
        c_to_core_now = self.delay_c_to_core.copy()
        
        core_input      = i_to_core_now + c_to_core_now
        core_eps_spikes = (core_input >= 2).astype(np.uint8)
        
        #4. Shift Core buffer left, inject new Core_{r,eps}
        # Done after B's conv read it, so this update doesn't affect B.
        self.core_buf[:, :-1] = self.core_buf[:, 1:]
        self.core_buf[:, -1]  = core_eps_spikes
        
        #2. C: convolution over I buffer (still previous timestep)
        c_input = self._conv_neighborhood(self.i_buf)
        c_spikes_now = (c_input >= self.min_pts - 1).astype(np.uint8)
        # Save for next step's Core calculation
        self.delay_c_to_core = c_spikes_now.copy()
        
        #1. Shift I buffer left, inject input
        # Last so the conv in step 2 saw the previous timestep's state.
        self.i_buf[:, :-1] = self.i_buf[:, 1:]
        self.i_buf[:, -1]  = input_column.astype(np.uint8)
        
        return core_eps_spikes, border_spikes, b_spikes_now
    
    def run(self, event_grid):
        global t #useful for looking at internals while debugging
        assert event_grid.shape == (self.R, self.C), f"Expected ({self.R}, {self.C}), got {event_grid.shape}"
        
        self.reset()
        
        core_mask   = np.zeros((self.R, self.C), dtype=np.uint8)
        border_mask = np.zeros((self.R, self.C), dtype=np.uint8)
        b_mask = np.zeros((self.R, self.C), dtype=np.uint8)
        
        total_steps = self.C + 2 * self.eps + 4
        
        for t in range(total_steps):
            # Determine input column for this timestep
            if t < self.C:
                input_column = event_grid[:, t]
            else:
                input_column = np.zeros(self.R, dtype=np.uint8)
            
            core_spikes, border_spikes, b_spikes = self.step(input_column)
            
            # Map outputs back to grid columns
            core_col = t - self.eps - 2
            if 0 <= core_col < self.C:
                core_mask[:, core_col] = core_spikes

            b_col = t - 2 * self.eps - 3
            if 0 <= b_col < self.C:
                b_mask[:,b_col] = b_spikes
            
            border_col = t - 2 * self.eps - 4
            if 0 <= border_col < self.C:
                border_mask[:, border_col] = border_spikes
        
        return core_mask, border_mask, b_mask


def get_clusters(cluster_pixels,b,min_pts): #turn clusters into labeled groups; b is border-cores, "glues" cluster together
    if len(cluster_pixels | b) < min_pts:
        return {}
    labeled, num_features = label(cluster_pixels | b)
    if num_features == 0:
        return []
    centroids = center_of_mass(cluster_pixels, labeled,
                               range(1, num_features + 1))
    return centroids


if __name__ == "__main__":
    model = SnnDbscan(6, 6, 1, 4)
    sample = np.array([[0, 0, 0, 0, 0, 1],
                       [1, 0, 1, 1, 0, 0],
                       [0, 1, 0, 0, 0, 0],
                       [1, 1, 1, 0, 0, 0],
                       [0, 0, 1, 0, 1, 1],
                       [0, 0, 0, 1, 0, 0]])
    core_mask, border_mask, b_mask = model.run(sample)
    print("Sample")
    print(sample)
    print("Core")
    print(core_mask)
    print("Border")
    print(border_mask)
    print("Core and Border")
    print(core_mask | border_mask)
    print("Border-Core")
    print(b_mask)
    print("Incorrect Clusters")
    print(get_clusters(core_mask | border_mask,0,4))
    print("Clusters")
    print(get_clusters(core_mask | border_mask,b_mask,4))

'''
Expected output:
Sample
[[0 0 0 0 0 1]
 [1 0 1 1 0 0]
 [0 1 0 0 0 0]
 [1 1 1 0 0 0]
 [0 0 1 0 1 1]
 [0 0 0 1 0 0]]
Core
[[0 0 0 0 0 0]
 [0 0 0 0 0 0]
 [0 1 0 0 0 0]
 [0 1 1 0 0 0]
 [0 0 1 0 0 0]
 [0 0 0 0 0 0]]
Border
[[0 0 0 0 0 0]
 [1 0 1 0 0 0]
 [0 0 0 0 0 0]
 [1 0 0 0 0 0]
 [0 0 0 0 0 0]
 [0 0 0 1 0 0]]
'''