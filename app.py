import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

class CEFGridWorld:
    def __init__(self, grid_size=10, horizon=5, prune_threshold=2):
        self.grid_size = grid_size
        self.horizon = horizon 
        self.directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        self.history_cache = {} 
        self.prune_threshold = prune_threshold 

    def compute_future_paths(self, state):
        paths = 0
        def dfs(pos, steps_left):
            nonlocal paths
            cache_key = (pos, steps_left)
            
            if cache_key in self.history_cache:
                paths += self.history_cache[cache_key]
                return
                
            if steps_left == 0:
                paths += 1
                return
                
            x, y = pos
            local_start_paths = paths
            
            for dx, dy in self.directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    full_horizon_key = ((nx, ny), self.horizon)
                    if full_horizon_key in self.history_cache:
                        if self.history_cache[full_horizon_key] < self.prune_threshold:
                            continue 
                    dfs((nx, ny), steps_left - 1)
            
            self.history_cache[cache_key] = paths - local_start_paths
        dfs(state, self.horizon)
        return paths

    def get_log_entropy(self, state):
        paths = self.compute_future_paths(state)
        return np.log(paths + 1)

    def compute_cef_vector(self, state, T=1.0):
        x, y = state
        current_entropy = self.get_log_entropy(state)
        force_x, force_y = 0.0, 0.0
        
        for dx, dy in self.directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                neighbor_entropy = self.get_log_entropy((nx, ny))
                gradient = neighbor_entropy - current_entropy
                force_x += dx * gradient
                force_y += dy * gradient
                
        return force_x * T, force_y * T

    def visualize_entropy_map(self):
        entropy_map = np.zeros((self.grid_size, self.grid_size))
        U = np.zeros((self.grid_size, self.grid_size)) 
        V = np.zeros((self.grid_size, self.grid_size)) 
        
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                entropy_map[x, y] = self.get_log_entropy((x, y))
                fx, fy = self.compute_cef_vector((x, y))
                U[x, y] = fx
                V[x, y] = fy

        fig, ax = plt.subplots(figsize=(9, 7))
        c = ax.imshow(entropy_map, cmap='magma', interpolation='nearest')
        ax.set_title(f"Causal Entropic Force Vector Field (Horizon τ={self.horizon})", color='white')
        fig.colorbar(c, ax=ax, label="Log Path Entropy S = log(W)")
        
        Y, X = np.mgrid[0:self.grid_size, 0:self.grid_size]
        ax.quiver(X, Y, V, U, color='cyan', scale=5, width=0.003, headwidth=4)
        
        # Streamlit dark mode fixes
        fig.patch.set_facecolor('#0E1117')
        ax.set_facecolor('#0E1117')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        
        return fig

# --- Streamlit UI ---
st.set_page_config(page_title="CEF Engine Sandbox", layout="centered")
st.title("Causal Entropic Force (CEF) Engine")
st.markdown("This engine calculates future path entropy $S = \log(W)$ and vectors for intelligent exploration.")

col1, col2 = st.columns(2)
with col1:
    grid_sz = st.slider("Grid Size", min_value=5, max_value=20, value=10)
with col2:
    hz = st.slider("Time Horizon (τ)", min_value=2, max_value=10, value=6)

if st.button("Generate CEF Landscape"):
    with st.spinner(f"Calculating trajectories for {grid_sz}x{grid_sz} grid over {hz} steps..."):
        env = CEFGridWorld(grid_size=grid_sz, horizon=hz, prune_threshold=10)
        fig = env.visualize_entropy_map()
        st.pyplot(fig)
        st.success(f"Calculated! Explored {len(env.history_cache)} unique state branches.")