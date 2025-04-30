import os
import tensorflow as tf
import numpy as np
import random
import datetime
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import time
import math # For noisy layer initialization
import io   # For TensorBoard image logging
from collections import deque # Used for replay buffer

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
 try:
  for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
  logical_gpus = tf.config.experimental.list_logical_devices('GPU')
  print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs configured with Memory Growth.")
 except RuntimeError as e:
   print(f"Memory growth error: {e}")
else:
   print("No GPU detected by TensorFlow.")


@tf.function
def flat_to_xyz(flat_idx, G):
    """Converts a flat index to (x, y, z) coordinates."""
    g_tf = tf.cast(G, tf.int32)
    z = flat_idx % g_tf
    y = (flat_idx // g_tf) % g_tf
    x = flat_idx // (g_tf * g_tf)
    # Ensure shapes match: if flat_idx is [N], result is [N, 3]
    if tf.rank(flat_idx) == 0: # Single index
        return tf.stack([x, y, z], axis=-1) # [3]
    else: # Batch of indices
        return tf.stack([x, y, z], axis=-1) # [N, 3]


@tf.function
def xyz_to_flat(x, y, z, G):
    """Converts (x, y, z) coordinates to a flat index."""
    g_tf = tf.cast(G, tf.int32)
    # Ensure inputs have same rank (e.g., [N] or scalar)
    return x * g_tf * g_tf + y * g_tf + z

@tf.function
def clamp_pos_tf(pos_xyz, G):
    """Clamps (x, y, z) coordinates to be within [0, G-1]."""
    g_minus_1 = tf.cast(G - 1, tf.int32)
    return tf.clip_by_value(pos_xyz, 0, g_minus_1)

@tf.function
def calculate_manhattan_distance(pos1_xyz, pos2_xyz):
    """Calculates Manhattan distance between two sets of (x, y, z) coordinates."""
    # Ensure pos1_xyz and pos2_xyz have the same rank and batch size
    return tf.reduce_sum(tf.abs(pos1_xyz - pos2_xyz), axis=-1) # [N] or scalar

@tf.function
def goal_index_to_delta_xyz(goal_index_batch, k):
    """Converts a batch of discrete goal indices [0, (2k+1)^3 - 1] to relative (dx, dy, dz) vectors."""
    # goal_index_batch: [N]
    # Returns: delta_xyz_batch [N, 3]
    base = tf.cast(2 * k + 1, tf.int32)
    shift = tf.constant(k, dtype=tf.int32)

    dz_batch = goal_index_batch % base
    dy_batch = tf.cast((goal_index_batch // base) % base, tf.int32)
    dx_batch = tf.cast(goal_index_batch // (base * base), tf.int32)

    # Shift from [0, 2k] range to [-k, k] range
    return tf.stack([dx_batch - shift, dy_batch - shift, dz_batch - shift], axis=-1) # [N, 3]


@tf.function
def calculate_subgoal_target_pos_flat(start_pos_flat_batch, goal_index_batch, G, k):
    """Calculates the target flat position for the subgoal based on start position and goal."""
    # start_pos_flat_batch: [N]
    # goal_index_batch: [N]
    # Returns: target_pos_flat_batch [N]

    start_pos_xyz_batch = flat_to_xyz(start_pos_flat_batch, G) # [N, 3]
    delta_xyz_batch = goal_index_to_delta_xyz(goal_index_batch, k) # [N, 3]

    target_xyz_raw_batch = start_pos_xyz_batch + delta_xyz_batch # [N, 3]
    target_xyz_clamped_batch = clamp_pos_tf(target_xyz_raw_batch, G) # [N, 3]

    target_pos_flat_batch = xyz_to_flat(target_xyz_clamped_batch[:, 0],
                                        target_xyz_clamped_batch[:, 1],
                                        target_xyz_clamped_batch[:, 2], G) # [N]
    return target_pos_flat_batch

@tf.function
def calculate_intrinsic_reward_batch(pos_t_flat_batch, pos_tplus1_flat_batch,
                                     subgoal_start_pos_flat_batch, current_goal_index_batch, G, k):
    """Calculates intrinsic reward for a batch based on progress towards subgoal target."""
    # pos_t_flat_batch: [N] - flat position *before* the step
    # pos_tplus1_flat_batch: [N] - flat position *after* the step
    # subgoal_start_pos_flat_batch: [N] - flat position where the current goal was set
    # current_goal_index_batch: [N] - the goal index set by the manager

    target_pos_flat_batch = calculate_subgoal_target_pos_flat(
        subgoal_start_pos_flat_batch, current_goal_index_batch, G, k
    ) # [N]

    # Need to convert target_pos_flat_batch to xyz for distance calculation
    target_pos_xyz_batch = flat_to_xyz(target_pos_flat_batch, G) # [N, 3]

    pos_t_xyz_batch = flat_to_xyz(pos_t_flat_batch, G) # [N, 3]
    # CORRECTED LINE: Use pos_tplus1_flat_batch as input
    pos_tplus1_xyz_batch = flat_to_xyz(pos_tplus1_flat_batch, G) # [N, 3]


    dist_t = calculate_manhattan_distance(pos_t_xyz_batch, target_pos_xyz_batch) # [N]
    dist_tplus1 = calculate_manhattan_distance(pos_tplus1_xyz_batch, target_pos_xyz_batch) # [N]

    # Intrinsic reward: decrease in distance
    # Use tf.cast to ensure float32 output
    intrinsic_reward_batch = tf.cast(dist_t, tf.float32) - tf.cast(dist_tplus1, tf.float32) # [N]

    return intrinsic_reward_batch


# -----------------------------------------------------------------------------
# 1) Batched Sculpt3DEnvTF (Hybrid Observation)
# -----------------------------------------------------------------------------
# The environment class remains largely the same. It provides the state and extrinsic reward.
# Intrinsic reward calculation and goal handling are done outside, in the agent and training loop.
class BatchedSculpt3DEnvTF:
    def __init__(self, grid_size=16, max_steps=200, n_envs=16):
        G, N = grid_size, n_envs
        if N <= 0: raise ValueError("n_envs must be positive.")
        self.G, self.N, self.max_steps = G, N, max_steps
        self.flat_dim = G*G*G
        self.grid_obs_shape = (G, G, G, 2) # Channels: Stock, ShapeMask
        self.coord_obs_shape = (3,)        # Channels: X, Y, Z (normalized)

        # Use tf operations for shape generation to keep it within TF graph if possible
        coords_range = tf.range(G, dtype=tf.float32)
        coords = tf.stack(tf.meshgrid(coords_range, coords_range, coords_range, indexing='ij'), axis=-1) # [G,G,G,3]
        center = tf.constant([G/2 - 0.5, G/2 - 0.5, G/2 - 0.5], tf.float32) # Center point
        dist2 = tf.reduce_sum(tf.square(coords - center), axis=-1) # [G,G,G] squared distance from center
        radius_sq = tf.square(tf.cast(G // 2 - 1, tf.float32)) # Define radius (slightly smaller than half grid)
        mask3d = dist2 <= radius_sq # Boolean mask [G,G,G]
        mask_flat = tf.reshape(mask3d, [-1]) # [G^3]

        # Initialize state variables as tf.Variables to allow updates in tf.function
        self.shape_mask = tf.Variable(tf.tile(mask_flat[None, :], [N, 1]), trainable=False, dtype=tf.bool, name="shape_mask")
        self.stock = tf.Variable(tf.ones([N, self.flat_dim], dtype=tf.bool), trainable=False, name="stock") # Start with full stock
        self.pos = tf.Variable(tf.zeros([N], dtype=tf.int32), trainable=False, name="pos")
        self.steps = tf.Variable(tf.zeros([N], dtype=tf.int32), trainable=False, name="steps")
        self.done = tf.Variable(tf.zeros([N], dtype=tf.bool), trainable=False, name="done")

        # Calculate shifts (outside tf.function, Python is fine)
        G_py = grid_size # Use Python int for calculation
        def to_flat_py(dx, dy, dz): return dx*G_py*G_py + dy*G_py + dz
        moves = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)] # dx, dy, dz
        shifts_py = [to_flat_py(*m) for m in moves]
        self.shifts = tf.constant(shifts_py, dtype=tf.int32, name="shifts")


    @tf.function
    def reset(self):
        # Reset stock to full, steps/done to zero
        self.stock.assign(tf.ones_like(self.stock))
        self.steps.assign(tf.zeros_like(self.steps))
        self.done.assign(tf.zeros_like(self.done))

        # Find safe starting positions (outside the target shape)
        # Use shape_mask[0] as reference (they are identical initially)
        safe_indices = tf.where(tf.logical_not(self.shape_mask[0]))[:, 0] # Get flat indices
        num_safe = tf.shape(safe_indices)[0]
        # Assert inside @tf.function is different, use tf.debugging.assert_greater_equal
        tf.debugging.assert_greater_equal(num_safe, self.N, message="Not enough safe starting positions available.")

        # Shuffle and select N starting positions
        shuffled_safe_indices = tf.random.shuffle(safe_indices)[:self.N]
        self.pos.assign(tf.cast(shuffled_safe_indices, tf.int32))

        return self._get_obs()

    @tf.function
    def step(self, actions):
        # Calculate potential new positions based on actions
        action_shifts = tf.gather(self.shifts, actions) # [N] shifts based on actions
        new_pos = self.pos + action_shifts              # [N] potential new flat indices

        # Check boundaries and collisions
        in_bounds = tf.logical_and(new_pos >= 0, new_pos < self.flat_dim) # [N] boolean
        # Clip new_pos for safe gathering, even if OOB
        safe_new_pos = tf.clip_by_value(new_pos, 0, self.flat_dim - 1)

        # Gather shape mask and stock at the potential new positions
        shape_mask_at_new = tf.gather(self.shape_mask, safe_new_pos, axis=1, batch_dims=1) # [N] bool
        stock_at_new = tf.gather(self.stock, safe_new_pos, axis=1, batch_dims=1)          # [N] bool

        # Determine invalid moves (hit shape or out of bounds)
        hit_shape_or_oob = tf.logical_or(tf.logical_not(in_bounds), shape_mask_at_new) # [N] bool

        # Determine if stock can be removed (valid move AND stock exists at new pos)
        can_remove = tf.logical_and(tf.logical_not(hit_shape_or_oob), stock_at_new) # [N] bool

        # Calculate extrinsic rewards
        reward = tf.where(hit_shape_or_oob, -5.0, 0.0)  # Penalty for invalid move
        reward = tf.where(can_remove, reward + 1.0, reward) # Reward for removing stock
        reward = reward - 0.1                           # Step penalty (extrinsic)


        # Update stock (remove material where applicable)
        remove_indices = tf.where(can_remove) # Indices [k, 0] where can_remove is True
        num_removals = tf.shape(remove_indices)[0]

        # Conditional update to avoid empty tensor issues if num_removals is 0
        def perform_update():
            # Indices of environments where removal happened
            env_indices_to_update = tf.squeeze(tf.cast(remove_indices, tf.int32), axis=1) # [num_removals]
            # Corresponding positions where removal happened
            pos_to_remove = tf.gather(new_pos, env_indices_to_update) # [num_removals]
            # Combine env and pos indices for scatter_nd_update: [[env0, pos0], [env1, pos1], ...]
            scatter_indices = tf.stack([env_indices_to_update, pos_to_remove], axis=1) # [num_removals, 2]
            # Values to update with (False, meaning remove stock)
            updates = tf.zeros(num_removals, dtype=tf.bool)
            return tf.tensor_scatter_nd_update(self.stock, scatter_indices, updates)

        # Use tf.case or tf.cond based on complexity. cond is fine for simple boolean.
        maybe_updated_stock = tf.cond(tf.greater(num_removals, 0),
                                      true_fn=perform_update,
                                      false_fn=lambda: self.stock) # No update if no removals
        self.stock.assign(maybe_updated_stock)

        # Update agent position (only if the move was valid)
        is_valid_move = tf.logical_not(hit_shape_or_oob) # [N] bool
        next_pos = tf.where(is_valid_move, new_pos, self.pos) # Stay if invalid, move if valid
        self.pos.assign(next_pos)

        # Update steps and check for done state
        self.steps.assign_add(tf.ones_like(self.steps))
        newly_done = (self.steps >= self.max_steps)
        self.done.assign(tf.logical_or(self.done, newly_done)) # Mark as done if max steps reached

        # Get next observation
        next_obs = self._get_obs()
        # Return next_obs, extrinsic_reward, done
        return next_obs, tf.cast(reward, tf.float32), self.done

    @tf.function
    def _get_obs(self):
        G = self.G; N = self.N
        # Reshape flat stock/shape masks to 3D grids
        stock_grid = tf.reshape(self.stock, [N, G, G, G])
        shape_mask_grid = tf.reshape(self.shape_mask, [N, G, G, G])

        # Convert to float and stack as channels
        stock_float = tf.cast(stock_grid, tf.float32)
        shape_mask_float = tf.cast(shape_mask_grid, tf.float32)
        grid_obs = tf.stack([stock_float, shape_mask_float], axis=-1) # [N, G, G, G, 2]


        # Calculate normalized coordinates
        g_tf = tf.constant(G, dtype=tf.int32)
        z = self.pos % g_tf
        y = (self.pos // g_tf) % g_tf
        x = self.pos // (g_tf * g_tf)

        # Normalize coordinates to [0, 1] range
        g_minus_1_float = tf.cast(tf.maximum(1, G - 1), tf.float32) # Avoid division by zero if G=1
        x_norm = tf.cast(x, tf.float32) / g_minus_1_float
        y_norm = tf.cast(y, tf.float32) / g_minus_1_float
        z_norm = tf.cast(z, tf.float32) / g_minus_1_float


        coord_obs = tf.stack([x_norm, y_norm, z_norm], axis=-1) # [N, 3]

        return (grid_obs, coord_obs) # Return as a tuple


# -----------------------------------------------------------------------------
# 2) Replay Buffers (Separate for Manager and Worker)
# -----------------------------------------------------------------------------

# Worker Buffer: (S_w_grid, S_w_coord, G_w_index, A_w, R_w_total, S2_w_grid, S2_w_coord, D_w)
class WorkerReplayBuffer:
    def __init__(self, capacity=50000):
        self.cap = capacity
        self.buf = deque(maxlen=capacity)

    def add_batch(self, S_tuple, G_batch, A, R_total, S2_tuple, D):
        # S_tuple, S2_tuple are (grid_obs, coord_obs) [N, ...]
        # G_batch: goal index batch [N]
        # A, R_total, D are tensors [N]
        # Store each (S_grid, S_coord, G_index, A, R, S2_grid, S2_coord, D) for each environment in the batch
        S_grid_batch, S_coord_batch = S_tuple
        S2_grid_batch, S2_coord_batch = S2_tuple
        N = tf.shape(A)[0].numpy() # Get batch size

        for i in range(N):
             self.buf.append((
                 S_grid_batch[i], S_coord_batch[i], # State S (single env)
                 G_batch[i],                       # Goal G (single env) - ADDED
                 A[i],                             # Action A (single env)
                 R_total[i],                       # Total Reward R (single env)
                 S2_grid_batch[i], S2_coord_batch[i], # Next State S2 (single env)
                 D[i]                              # Done D (single env)
             ))

    def sample(self, batch_size=32):
        if len(self.buf) < batch_size:
            return None

        batch = random.sample(self.buf, batch_size)

        # Unzip and stack components into batch tensors
        S_grid_list, S_coord_list, G_list, A_list, R_list, S2_grid_list, S2_coord_list, D_list = zip(*batch)

        return (
            tf.stack(S_grid_list, axis=0), tf.stack(S_coord_list, axis=0), # State S [B, ...]
            tf.stack(G_list, axis=0),                                     # Goal G [B] - ADDED
            tf.stack(A_list, axis=0),                                     # Action A [B]
            tf.stack(R_list, axis=0),                                     # Reward R [B]
            tf.stack(S2_grid_list, axis=0), tf.stack(S2_coord_list, axis=0), # Next State S2 [B, ...]
            tf.stack(D_list, axis=0)                                      # Done D [B]
        )

    def __len__(self):
        return len(self.buf)


# Manager Buffer: (S_m_grid, S_m_coord, G, R_m_accumulated_extrinsic, S'_m_grid, S'_m_coord, D_m)
class ManagerReplayBuffer:
    def __init__(self, capacity=5000): # Manager buffer can be smaller
        self.cap = capacity
        self.buf = deque(maxlen=capacity)

    def add_batch(self, S_tuple, G, R_accumulated, S2_tuple, D):
        # S_tuple, S2_tuple are (grid_obs, coord_obs) at start/end of horizon [N, ...]
        # G is goal index [N]
        # R_accumulated is accumulated extrinsic reward [N]
        # D is done flag [N]

        S_grid_batch, S_coord_batch = S_tuple
        S2_grid_batch, S2_coord_batch = S2_tuple
        N = tf.shape(G)[0].numpy() # Get batch size

        for i in range(N):
             self.buf.append((
                 S_grid_batch[i], S_coord_batch[i], # State S (single env, start of horizon)
                 G[i],                             # Goal G (single env)
                 R_accumulated[i],                 # Accumulated Reward R (single env)
                 S2_grid_batch[i], S2_coord_batch[i], # Next State S2 (single env, end of horizon)
                 D[i]                              # Done D (single env, episode ended)
             ))

    def sample(self, batch_size=32):
        if len(self.buf) < batch_size:
            return None

        batch = random.sample(self.buf, batch_size)

        # Unzip and stack components into batch tensors
        S_grid_list, S_coord_list, G_list, R_list, S2_grid_list, S2_coord_list, D_list = zip(*batch)

        return (
            tf.stack(S_grid_list, axis=0), tf.stack(S_coord_list, axis=0), # State S [B, ...]
            tf.stack(G_list, axis=0),                                     # Goal G [B]
            tf.stack(R_list, axis=0),                                     # Reward R [B]
            tf.stack(S2_grid_list, axis=0), tf.stack(S2_coord_list, axis=0), # Next State S2 [B, ...]
            tf.stack(D_list, axis=0)                                      # Done D [B]
        )

    def __len__(self):
        return len(self.buf)


# -----------------------------------------------------------------------------
# 3) Noisy Dense Layer (Factorized Gaussian Noise)
# -----------------------------------------------------------------------------
class NoisyDense(tf.keras.layers.Layer):
    def __init__(self, units, activation=None, sigma0=0.5, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation = tf.keras.activations.get(activation)
        self.sigma0 = sigma0 # Initial standard deviation parameter

    def build(self, input_shape):
        in_features = input_shape[-1]
        out_features = self.units
        dtype = tf.float32 # Assuming float32

        # Weight parameters (mean and standard deviation)
        sigma_init_val = self.sigma0 / math.sqrt(float(in_features))
        sigma_initializer = tf.constant_initializer(sigma_init_val)
        self.kernel_mean = self.add_weight(name="kernel_mean", shape=(in_features, out_features),
                                             initializer="he_uniform", trainable=True, dtype=dtype)
        self.kernel_sigma = self.add_weight(name="kernel_sigma", shape=(in_features, out_features),
                                             initializer=sigma_initializer, trainable=True, dtype=dtype)

        # Bias parameters (mean and standard deviation)
        self.bias_mean = self.add_weight(name="bias_mean", shape=(out_features,),
                                         initializer="zeros", trainable=True, dtype=dtype)
        self.bias_sigma = self.add_weight(name="bias_sigma", shape=(out_features,),
                                          initializer=sigma_initializer, trainable=True, dtype=dtype)

        super().build(input_shape)

    # @tf.function # call can sometimes cause issues if not eager, leave as non-tf.function for now
    def call(self, inputs, training=None):
        # The `training` argument is provided by Keras.
        # It will be True during training, False during evaluation/prediction,
        # and None if the layer is called directly outside a Keras training loop
        # without the argument specified.
        # Noisy layers should typically apply noise only when `training` is True.

        if training: # This correctly checks if training is True
            # Sample noise for weights and biases using Factorized Gaussian noise
            # Generate noise for input and output dimensions
            noise_in = self._factorized_noise(tf.shape(inputs)[-1])  # Shape [in_features]
            noise_out = self._factorized_noise(self.units)          # Shape [out_features]

            # Combine noise for weight matrix: outer product
            kernel_noise = tf.tensordot(tf.expand_dims(noise_in, -1), tf.expand_dims(noise_out, 0), axes=1) # Shape [in_features, out_features]

            # Noise for bias is just the output noise
            bias_noise = noise_out # Shape [out_features]

            # Apply noise: W = W_mu + W_sigma * noise_W, b = b_mu + b_sigma * noise_b
            kernel = self.kernel_mean + self.kernel_sigma * kernel_noise
            bias = self.bias_mean + self.bias_sigma * bias_noise
        else: # This covers training=False and training=None (inference)
            # In inference mode, use only the mean weights and biases
            kernel = self.kernel_mean
            bias = self.bias_mean

        # Standard dense layer calculation: output = input * kernel + bias
        output = tf.matmul(inputs, kernel) + bias

        # Apply activation function if specified
        if self.activation is not None:
            output = self.activation(output)
        return output

    @tf.function # Make _factorized_noise a tf.function as it doesn't depend on layer state
    def _factorized_noise(self, num_elements):
        """Generates noise based on the Factorized Gaussian noise formula."""
        # Sample standard normal noise
        noise = tf.random.normal(shape=[num_elements], dtype=tf.float32) # Ensure dtype is float32
        # Apply transformation: sign(x) * sqrt(|x|)
        return tf.sign(noise) * tf.sqrt(tf.abs(noise))

    # compute_output_shape is generally not needed for tf.function calls unless using Layer directly in graph ops
    # def compute_output_shape(self, input_shape):
    #     # Output shape is same as input shape except for the last dimension (units)
    #     return tuple(input_shape[:-1]) + (self.units,)


# -----------------------------------------------------------------------------
# 4) Feudal Agent (Manager + Worker DQNs)
# -----------------------------------------------------------------------------
class FeudalAgentTF:
    def __init__(self, grid_shape, coord_shape, primitive_action_dim=6,
                 manager_goal_k=1, # Define discrete goals: delta_xyz in [-k, k]^3
                 subgoal_horizon=10, intrinsic_reward_beta=0.1,
                 manager_lr=1e-4, worker_lr=1e-4, gamma=0.99, tau=0.005):

        self.grid_shape = grid_shape
        self.coord_shape = coord_shape
        self.primitive_action_dim = primitive_action_dim
        self.gamma = gamma
        self.tau = tau
        self.subgoal_horizon = subgoal_horizon
        self.intrinsic_reward_beta = intrinsic_reward_beta

        # Manager Goal Space
        self.manager_goal_k = manager_goal_k
        self.manager_goal_base = 2 * manager_goal_k + 1
        self.manager_goal_dim = self.manager_goal_base ** 3 # Number of discrete goals

        print(f"Feudal Agent initialized:")
        print(f"  Subgoal Horizon (M): {self.subgoal_horizon}")
        print(f"  Intrinsic Reward Beta: {self.intrinsic_reward_beta}")
        print(f"  Manager Goal K: {self.manager_goal_k} (Discrete goals: {self.manager_goal_dim})")


        # --- Build Manager Model ---
        def build_manager_model(name="Manager_Model"):
            grid_input = tf.keras.layers.Input(shape=self.grid_shape, name="manager_grid_input")
            coord_input = tf.keras.layers.Input(shape=self.coord_shape, name="manager_coord_input")

            # CNN part for grid observations
            x_cnn = tf.keras.layers.Conv3D(filters=32, kernel_size=5, strides=2, activation='relu', padding='same', name="m_conv1")(grid_input)
            x_cnn = tf.keras.layers.Conv3D(filters=64, kernel_size=3, strides=2, activation='relu', padding='same', name="m_conv2")(x_cnn)
            x_cnn = tf.keras.layers.Conv3D(filters=64, kernel_size=3, strides=1, activation='relu', padding='same', name="m_conv3")(x_cnn)
            cnn_features = tf.keras.layers.Flatten(name="m_flatten")(x_cnn)

            # Concatenate CNN features with coordinate observations
            concat_features = tf.keras.layers.Concatenate(name="m_concat")([cnn_features, coord_input])

            # Dense part with Noisy Layers
            x = NoisyDense(256, activation='relu', name="manager_dense1")(concat_features) # Smaller dense for manager?
            outputs = NoisyDense(self.manager_goal_dim, activation='linear', name="manager_output")(x) # Q-values for each goal

            return tf.keras.Model(inputs=[grid_input, coord_input], outputs=outputs, name=name)

        # --- Build Worker Model ---
        def build_worker_model(name="Worker_Model"):
            grid_input = tf.keras.layers.Input(shape=self.grid_shape, name="worker_grid_input")
            coord_input = tf.keras.layers.Input(shape=self.coord_shape, name="worker_coord_input")
            # Worker also takes the goal as input
            goal_input = tf.keras.layers.Input(shape=(self.manager_goal_dim,), name="worker_goal_input") # One-hot goal


            # CNN part for grid observations
            x_cnn = tf.keras.layers.Conv3D(filters=32, kernel_size=5, strides=2, activation='relu', padding='same', name="w_conv1")(grid_input)
            x_cnn = tf.keras.layers.Conv3D(filters=64, kernel_size=3, strides=2, activation='relu', padding='same', name="w_conv2")(x_cnn)
            x_cnn = tf.keras.layers.Conv3D(filters=64, kernel_size=3, strides=1, activation='relu', padding='same', name="w_conv3")(x_cnn)
            cnn_features = tf.keras.layers.Flatten(name="w_flatten")(x_cnn)

            # Concatenate CNN features with coordinate observations AND one-hot goal
            concat_features = tf.keras.layers.Concatenate(name="w_concat")([cnn_features, coord_input, goal_input])


            # Dense part with Noisy Layers
            x = NoisyDense(512, activation='relu', name="worker_dense1")(concat_features) # Potentially larger dense for worker
            outputs = NoisyDense(primitive_action_dim, activation='linear', name="worker_output")(x) # Q-values for primitive actions

            return tf.keras.Model(inputs=[grid_input, coord_input, goal_input], outputs=outputs, name=name)

        # Initialize online and target models for both Manager and Worker
        self.manager_model = build_manager_model()
        self.manager_target = build_manager_model()
        self.manager_target.set_weights(self.manager_model.get_weights())

        self.worker_model = build_worker_model()
        self.worker_target = build_worker_model()
        self.worker_target.set_weights(self.worker_model.get_weights())


        # Optimizers
        self.manager_opt = tf.keras.optimizers.Adam(learning_rate=manager_lr)
        self.worker_opt = tf.keras.optimizers.Adam(learning_rate=worker_lr)

        # Replay Buffers
        self.manager_buffer = ManagerReplayBuffer()
        self.worker_buffer = WorkerReplayBuffer()

        # TensorBoard Writer
        logdir = f"runs/feudal_dqn_{datetime.datetime.now():%Y%m%d_%H%M%S}"
        self.writer = tf.summary.create_file_writer(logdir)
        print(f"TensorBoard log directory: {logdir}")

        # Training step counters (tf.Variables)
        self.manager_train_step_count = tf.Variable(0, dtype=tf.int64, trainable=False, name="manager_train_steps")
        self.worker_train_step_count = tf.Variable(0, dtype=tf.int64, trainable=False, name="worker_train_steps")


    @tf.function
    def manager_train_step(self, S_grid, S_coord, G, R_accumulated, S2_grid, S2_coord, D):
        """Performs a single training step for the Manager network."""

        # --- Target Q-value Calculation (Double DQN) ---
        # 1. Get next goals from the *online* manager model for S2
        Q2_manager_online = self.manager_model([S2_grid, S2_coord], training=True) # Pass training=True for noisy layers
        best_goals_next = tf.argmax(Q2_manager_online, axis=1, output_type=tf.int32) # Shape [batch_size]

        # 2. Get Q-values from the *target* manager model for S2
        Q2_manager_target = self.manager_target([S2_grid, S2_coord], training=True) # Pass training=True for noisy layers

        # 3. Select the Q-value from the target network corresponding to the best goal selected by the online network
        batch_indices = tf.range(tf.shape(best_goals_next)[0], dtype=tf.int32)
        goal_indices = tf.stack([batch_indices, best_goals_next], axis=1)
        Q2_best_manager_target = tf.gather_nd(Q2_manager_target, goal_indices) # Shape [batch_size]

        # 4. Calculate the TD target: R_manager + gamma * Q_target(S', argmax_g Q_online(S', g)) * (1 - D_m)
        # Note: Manager gamma is applied at the end of the horizon
        target_Q_manager = R_accumulated + self.gamma * Q2_best_manager_target * (1.0 - tf.cast(D, tf.float32))
        # --- End Target Q-value Calculation ---


        # --- Loss Calculation and Gradient Update ---
        with tf.GradientTape() as tape:
            # Predict Q-values for the original states (S) using the online manager model
            Q_manager_online = self.manager_model([S_grid, S_coord], training=True) # Pass training=True for noisy layers

            # Select the Q-values corresponding to the goals actually taken (G)
            goal_indices_taken = tf.stack([batch_indices, G], axis=1)
            Q_manager_online_taken = tf.gather_nd(Q_manager_online, goal_indices_taken) # Shape [batch_size]

            # Calculate loss (Mean Squared Error) between target_Q_manager and Q_manager_online_taken
            loss = tf.keras.losses.MeanSquaredError()(target_Q_manager, Q_manager_online_taken)

        # Compute and apply gradients
        grads = tape.gradient(loss, self.manager_model.trainable_variables)
        self.manager_opt.apply_gradients(zip(grads, self.manager_model.trainable_variables))
        # --- End Loss Calculation and Gradient Update ---

        # --- Soft Update Target Network (Graph-compatible) ---
        # Iterate through target network variables and update using assign
        for target_var, online_var in zip(self.manager_target.weights, self.manager_model.weights):
             target_var.assign(self.tau * online_var + (1.0 - self.tau) * target_var)
        # --- End Soft Update Target Network ---

        return loss

    @tf.function
    def worker_train_step(self, S_grid, S_coord, G_one_hot, A, R_total_worker, S2_grid, S2_coord, D):
        """Performs a single training step for the Worker network."""
        # G_one_hot: One-hot encoded goal batch [batch_size, manager_goal_dim]

        # --- Target Q-value Calculation (Double DQN) ---
        # 1. Get next actions from the *online* worker model for S2 + Goal
        # Note: Worker continues working towards the same goal G in S2
        Q2_worker_online = self.worker_model([S2_grid, S2_coord, G_one_hot], training=True) # Pass training=True
        best_actions_next = tf.argmax(Q2_worker_online, axis=1, output_type=tf.int32) # Shape [batch_size]

        # 2. Get Q-values from the *target* worker model for S2 + Goal
        Q2_worker_target = self.worker_target([S2_grid, S2_coord, G_one_hot], training=True) # Pass training=True

        # 3. Select the Q-value from the target network corresponding to the best action selected by the online network
        batch_indices = tf.range(tf.shape(best_actions_next)[0], dtype=tf.int32)
        action_indices = tf.stack([batch_indices, best_actions_next], axis=1)
        Q2_best_worker_target = tf.gather_nd(Q2_worker_target, action_indices) # Shape [batch_size]

        # 4. Calculate the TD target: R_total_worker + gamma * Q_target(S', G, argmax_a Q_online(S', G, a)) * (1 - D_w)
        # Note: Worker gamma is per step
        target_Q_worker = R_total_worker + self.gamma * Q2_best_worker_target * (1.0 - tf.cast(D, tf.float32))
        # --- End Target Q-value Calculation ---


        # --- Loss Calculation and Gradient Update ---
        with tf.GradientTape() as tape:
            # Predict Q-values for the original states (S) using the online worker model
            Q_worker_online = self.worker_model([S_grid, S_coord, G_one_hot], training=True) # Pass training=True

            # Select the Q-values corresponding to the actions actually taken (A)
            action_indices_taken = tf.stack([batch_indices, A], axis=1)
            Q_worker_online_taken = tf.gather_nd(Q_worker_online, action_indices_taken) # Shape [batch_size]

            # Calculate loss (Mean Squared Error)
            loss = tf.keras.losses.MeanSquaredError()(target_Q_worker, Q_worker_online_taken)

        # Compute and apply gradients
        grads = tape.gradient(loss, self.worker_model.trainable_variables)
        self.worker_opt.apply_gradients(zip(grads, self.worker_model.trainable_variables))
        # --- End Loss Calculation and Gradient Update ---

        # --- Soft Update Target Network (Graph-compatible) ---
        # Iterate through target network variables and update using assign
        for target_var, online_var in zip(self.worker_target.weights, self.worker_model.weights):
             target_var.assign(self.tau * online_var + (1.0 - self.tau) * target_var)
        # --- End Soft Update Target Network ---

        return loss


    @tf.function
    def manager_act_batch(self, S_tuple, deterministic=False):
        """Manager selects a goal for a batch of states."""
        # S_tuple is (grid_obs, coord_obs) batch [N, ...]
        q_values = self.manager_model(S_tuple, training=not deterministic) # Pass training=True for noise unless deterministic
        goals = tf.argmax(q_values, axis=1, output_type=tf.int32) # Returns goal index [N]
        return goals

    @tf.function
    def worker_act_batch(self, S_tuple, G_batch, deterministic=False):
        """Worker selects a primitive action for a batch of states given goals."""
        # S_tuple: (grid_obs, coord_obs) batch [N, ...]
        # G_batch: goal index batch [N]
        S_grid, S_coord = S_tuple
        G_one_hot = tf.one_hot(G_batch, depth=self.manager_goal_dim, dtype=tf.float32) # One-hot encode goals [N, manager_goal_dim]

        q_values = self.worker_model([S_grid, S_coord, G_one_hot], training=not deterministic) # Pass training=True for noise unless deterministic
        actions = tf.argmax(q_values, axis=1, output_type=tf.int32) # Returns primitive action index [N]
        return actions


    def manager_remember_batch(self, S_tuple, G, R_accumulated, S2_tuple, D):
        """Adds a batch of Manager transitions to the buffer."""
        # S_tuple, S2_tuple are (grid, coord) batches [N, ...]
        # G, R_accumulated, D are batches [N]
        if tf.shape(G)[0] > 0: # Only add if the batch is not empty
             self.manager_buffer.add_batch(S_tuple, G, R_accumulated, S2_tuple, D)


    def worker_remember_batch(self, S_tuple, G_batch, A, R_total, S2_tuple, D):
        """Adds a batch of Worker transitions (including goal) to the buffer."""
        # S_tuple, S2_tuple are (grid, coord) batches [N, ...]
        # G_batch, A, R_total, D are batches [N]
        if tf.shape(A)[0] > 0: # Only add if the batch is not empty
             self.worker_buffer.add_batch(S_tuple, G_batch, A, R_total, S2_tuple, D)


    def worker_learn(self, batch_size=32):
        """Samples from worker buffer and performs a training step."""
        if len(self.worker_buffer) < batch_size:
            return None

        sampled_data = self.worker_buffer.sample(batch_size)
        if sampled_data is None: # Should not happen if len check passes
             return None

        # Sampled data: (S_grid, S_coord, G_index, A, R_total, S2_grid, S2_coord, D)
        S_grid_s, S_coord_s, G_s, A_s, R_total_s, S2_grid_s, S2_coord_s, D_s = sampled_data

        # Need to one-hot encode G_s for the worker_train_step input
        # Ensure G_s is int32 before one-hot encoding
        G_s_one_hot = tf.one_hot(tf.cast(G_s, tf.int32), depth=self.manager_goal_dim, dtype=tf.float32)

        loss = self.worker_train_step(S_grid_s, S_coord_s, G_s_one_hot, A_s, R_total_s, S2_grid_s, S2_coord_s, D_s)
        self.worker_train_step_count.assign_add(1) # Increment worker step count

        return loss.numpy() # Return scalar loss value


    def manager_learn(self, batch_size=32):
        """Samples from manager buffer and performs a training step."""
        if len(self.manager_buffer) < batch_size:
            return None

        sampled_data = self.manager_buffer.sample(batch_size)
        if sampled_data is None: # Should not happen if len check passes
             return None

        # Sampled data: (S_m_grid, S_m_coord, G, R_accumulated_extrinsic, S'_m_grid, S'_m_coord, D_m)
        S_grid_s, S_coord_s, G_s, R_accumulated_s, S2_grid_s, S2_coord_s, D_s = sampled_data

        # Ensure G_s is int32 for the train step
        loss = self.manager_train_step(S_grid_s, S_coord_s, tf.cast(G_s, tf.int32), R_accumulated_s, S2_grid_s, S2_coord_s, D_s)
        self.manager_train_step_count.assign_add(1) # Increment manager step count

        return loss.numpy() # Return scalar loss value


# -----------------------------------------------------------------------------
# 5) Evaluation Function (Adapted for Feudal Agent)
# -----------------------------------------------------------------------------
# This evaluation function is largely the same as the previous one, as it simulates
# the combined Feudal agent's behavior (Manager sets goal, Worker executes)
# and measures the same overall task performance.

def evaluate_agent_performance(agent, grid_size, max_steps, num_eval_episodes=10, render=True, render_env_index=0):
    """
    Evaluates a trained Feudal agent deterministically, calculates performance statistics,
    and optionally renders the final state of one specified environment.
    Handles hybrid state (grid, coords). Returns key statistics.
    """
    print(f"\n--- Running Evaluation ({num_eval_episodes} episodes) ---")
    eval_start_time = time.time()

    # Create a separate environment instance for evaluation
    # Use n_envs = num_eval_episodes for the evaluation batch
    eval_env = BatchedSculpt3DEnvTF(grid_size=grid_size, max_steps=max_steps, n_envs=num_eval_episodes)
    N_eval = num_eval_episodes # Use N_eval for clarity in this function

    # Calculate initial carvable count (do this once)
    # Accessing shape_mask[0] is safe as it's the same for all envs
    initial_shape_mask_flat_gpu = eval_env.shape_mask[0]
    initial_shape_mask_flat_np = initial_shape_mask_flat_gpu.numpy()
    initial_carvable_mask_flat = ~initial_shape_mask_flat_np # Material outside the shape
    initial_carvable_count = np.sum(initial_carvable_mask_flat)
    print(f"  Initial number of carvable voxels: {initial_carvable_count}")
    if initial_carvable_count == 0: print("  Warning: No carvable material defined.")

    # Lists to store results across all evaluation episodes
    all_ep_rewards = []
    all_ep_lengths = []
    all_ep_removed_counts = []
    all_ep_incorrect_removed_counts = []

    # Get the final stock state variable from the eval env *before* the loop
    final_stock_variable_eval = eval_env.stock

    # Feudal specific state for evaluation (using tf.Variables to interact with env tf.function)
    # These are local to the evaluation function, not the agent.
    current_goals = tf.Variable(tf.zeros([N_eval], dtype=tf.int32), trainable=False, name="eval_goals") # Store goal index
    subgoal_steps_remaining = tf.Variable(tf.zeros([N_eval], dtype=tf.int32), trainable=False, name="eval_subgoal_steps")


    # Run evaluation episodes
    obs_tuple = eval_env.reset() # Get initial state for the batch
    done = eval_env.done # Get initial done flags (all False)

    # Episode-level accumulators for the evaluation batch (tf.Variables)
    ep_rewards = tf.Variable(tf.zeros([N_eval], dtype=tf.float32), trainable=False, name="eval_ep_rewards")
    ep_steps = tf.Variable(tf.zeros([N_eval], dtype=tf.int32), trainable=False, name="eval_ep_steps")

    # Initial Manager action for all environments at the start of evaluation
    # Need to perform this outside the main evaluation step loop to initialize feudal state
    manager_actions_initial = agent.manager_act_batch(obs_tuple, deterministic=True)
    current_goals.assign(manager_actions_initial)
    subgoal_steps_remaining.assign(tf.constant(agent.subgoal_horizon, dtype=tf.int32, shape=[N_eval]))

    # Use a tf.function for the evaluation step for performance
    @tf.function
    def evaluation_step(current_obs_tuple, current_done,
                         current_goals_var, subgoal_steps_remaining_var, ep_rewards_var, ep_steps_var,
                         eval_agent_subgoal_horizon, eval_env_step_fn, eval_agent_manager_act_fn, eval_agent_worker_act_fn): # Pass callables

         # Check done flags from *previous* step
         if tf.reduce_all(current_done): return current_obs_tuple, current_done # Exit early if all envs are done

         obs_grid, obs_coord = current_obs_tuple

         # Mask for environments needing a new goal (either horizon ended or episode done)
         # Use the 'done' flags *from the previous step* to decide if a new goal is needed
         end_of_horizon_mask = tf.equal(subgoal_steps_remaining_var.read_value(), 0)
         episode_done_mask = current_done
         manager_update_mask = tf.logical_or(end_of_horizon_mask, episode_done_mask)

         # Identify environments that need a manager update
         masked_indices = tf.where(manager_update_mask)[:, 0]
         num_masked = tf.shape(masked_indices)[0]


         # --- Manager Step (Deterministic) ---
         # Manager acts only for environments needing a new goal
         if num_masked > 0:
             # Get states for the masked subset
             manager_states_grid = tf.gather(obs_grid, masked_indices)
             manager_states_coord = tf.gather(obs_coord, masked_indices)
             manager_states = (manager_states_grid, manager_states_coord)

             # Manager selects goals deterministically for these states
             new_goals_selected = eval_agent_manager_act_fn(manager_states, deterministic=True) # Use passed function

             # Scatter the new goals back into the full batch variable
             current_goals_var.assign(tf.tensor_scatter_nd_update(current_goals_var.read_value(), tf.expand_dims(masked_indices, axis=1), new_goals_selected))

             # Reset subgoal steps for these environments
             subgoal_steps_remaining_var.assign(tf.tensor_scatter_nd_update(
                 subgoal_steps_remaining_var.read_value(), tf.expand_dims(masked_indices, axis=1),
                 tf.constant(eval_agent_subgoal_horizon, dtype=tf.int32, shape=[num_masked])) # Use passed horizon
             )

         # --- Worker Step (Deterministic) ---
         # Worker always acts for all environments using the current state and current goals
         A = eval_agent_worker_act_fn(current_obs_tuple, current_goals_var.read_value(), deterministic=True) # Use passed function

         # Step the environment (pass action batch)
         S2_tuple, R, next_done = eval_env_step_fn(A) # Use passed function


         # Update rewards and steps only for environments not yet done (based on 'done' flags from BEFORE the step)
         active_mask_current = ~current_done
         ep_rewards_var.assign_add(R * tf.cast(active_mask_current, tf.float32))
         ep_steps_var.assign_add(tf.cast(active_mask_current, tf.int32))

         # Decrement subgoal steps for active environments (based on 'next_done' flags)
         active_mask_next = ~next_done # Use next_done to see which envs are still running
         subgoal_steps_remaining_var.assign(tf.where(active_mask_next, subgoal_steps_remaining_var.read_value() - 1, subgoal_steps_remaining_var.read_value()))

         # Return next observation tuple and done flags
         return S2_tuple, next_done


    # Loop for evaluation steps
    for _ in range(max_steps): # Limit loop by max_steps
         # Call the tf.function evaluation step, passing required arguments
         obs_tuple, done = evaluation_step(
             obs_tuple, done,
             current_goals, subgoal_steps_remaining, ep_rewards, ep_steps,
             agent.subgoal_horizon, eval_env.step, agent.manager_act_batch, agent.worker_act_batch # Pass constants and callables
         )
         if tf.reduce_all(done): break # Exit early if all envs are done


    # After the loop, collect final results (from the tf.Variable state)
    final_stock_batch_np = final_stock_variable_eval.numpy() # Get final stock state
    batch_rewards = ep_rewards.numpy()
    batch_lengths = ep_steps.numpy()

    all_ep_rewards.extend(batch_rewards.tolist())
    all_ep_lengths.extend(batch_lengths.tolist())

    # Calculate removed/incorrect counts per episode
    for i in range(N_eval):
        final_stock_flat_np = final_stock_batch_np[i]

        # Correctly removed: Initially carvable AND now gone
        removed_mask = initial_carvable_mask_flat & (~final_stock_flat_np)
        removed_count = np.sum(removed_mask)
        all_ep_removed_counts.append(removed_count)

        # Incorrectly removed: Initially part of shape AND now gone
        incorrectly_removed_mask = initial_shape_mask_flat_np & (~final_stock_flat_np)
        incorrectly_removed_count = np.sum(incorrectly_removed_mask)
        all_ep_incorrect_removed_counts.append(incorrectly_removed_count)

    # Aggregate and Print Statistics
    avg_reward = np.mean(all_ep_rewards); std_reward = np.std(all_ep_rewards)
    avg_length = np.mean(all_ep_lengths)
    avg_removed_count = np.mean(all_ep_removed_counts)
    avg_incorrect_removed = np.mean(all_ep_incorrect_removed_counts)

    # Calculate removal percentage safely
    if initial_carvable_count > 0:
        removal_percentages = [(c / initial_carvable_count) * 100.0 for c in all_ep_removed_counts]
        avg_removal_percentage = np.mean(removal_percentages)
        std_removal_percentage = np.std(removal_percentages)
    else:
        avg_removal_percentage = 0.0
        std_removal_percentage = 0.0

    print(f"\n--- Evaluation Results ---")
    print(f"  Avg Reward : {avg_reward:.2f} (+/- {std_reward:.2f})")
    print(f"  Avg Length : {avg_length:.1f}")
    print(f"  Avg Removed: {avg_removed_count:.1f} / {initial_carvable_count} ({avg_removal_percentage:.2f}% +/- {std_removal_percentage:.2f}%)")
    print(f"  Avg Incorrect: {avg_incorrect_removed:.1f}")
    eval_duration = time.time() - eval_start_time
    print(f"  Evaluation Duration: {eval_duration:.2f}s")

    # Render Final State
    if render:
        print(f"\n--- Rendering final state for Eval Env Index: {render_env_index} ---")
        if render_env_index < 0 or render_env_index >= N_eval:
            print(f"Error: render_env_index ({render_env_index}) out of bounds for {N_eval} eval envs.")
        elif N_eval > 0:
            try:
                # Get final stock and shape mask for the specific environment
                final_stock_flat_np = final_stock_batch_np[render_env_index]
                shape_mask_flat_np = initial_shape_mask_flat_np # Same for all

                G = grid_size
                final_stock_3d = final_stock_flat_np.reshape((G, G, G))
                shape_mask_3d = shape_mask_flat_np.reshape((G, G, G))

                # Masks for plotting
                shape_to_plot = shape_mask_3d # Target shape
                initial_carvable_mask_render = ~shape_mask_3d # Initially carvable
                removed_mask_render = initial_carvable_mask_render & (~final_stock_3d) # Correctly removed
                incorrectly_removed_mask_render = shape_mask_3d & (~final_stock_3d) # Incorrectly removed

                # Plotting
                fig = plt.figure(figsize=(9, 7)); ax = fig.add_subplot(111, projection='3d')
                ax.set_facecolor('whitesmoke')
                # Coordinates for voxels (need +1 for boundaries)
                x_vox, y_vox, z_vox = np.indices(np.array(shape_to_plot.shape) + 1)

                # Plot volumes
                ax.voxels(x_vox, y_vox, z_vox, shape_np, facecolors='blue', alpha=0.1, edgecolor=None) # Target shape
                ax.voxels(x_vox, y_vox, z_vox, removed_mask_render, facecolors='red', alpha=0.6, edgecolor=None) # Correctly removed
                if np.sum(incorrectly_removed_mask_render) > 0:
                     ax.voxels(x_vox, y_vox, z_vox, incorrectly_removed_mask_render, facecolors='yellow', alpha=0.7, edgecolor='orange', label='Incorrect Removal')
                     ax.legend()

                # Set title and labels
                rendered_env_reward = all_ep_rewards[render_env_index]
                rendered_env_removed = all_ep_removed_counts[render_env_index]
                rendered_env_perc = (rendered_env_removed / initial_carvable_count) * 100.0 if initial_carvable_count > 0 else 0.0
                ax.set_title(f"Eval Render Env #{render_env_index} (R={rendered_env_reward:.1f}, Removed={rendered_env_perc:.1f}%)")
                ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
                ax.set_xlim(0, G); ax.set_ylim(0, G); ax.set_zlim(0, G); ax.set_aspect('auto') # Use 'auto' or 'equal'
                plt.tight_layout()

                # Save the plot
                render_dir = "renders_eval_feudal" # Separate render directory
                os.makedirs(render_dir, exist_ok=True) # Ensure directory exists
                save_path = os.path.join(render_dir, f"eval_render_env_{render_env_index}_final.png")
                plt.savefig(save_path); print(f"Saved evaluation render to {save_path}"); plt.close(fig)
            except Exception as e:
                print(f"Error during rendering: {e}")
                import traceback; traceback.print_exc()


    # Return aggregated statistics
    return {
        "avg_reward": avg_reward,
        "std_reward": std_reward,
        "avg_length": avg_length,
        "avg_removed_count": avg_removed_count,
        "avg_incorrect_removed": avg_incorrect_removed,
        "avg_removal_percentage": avg_removal_percentage,
        "std_removal_percentage": std_removal_percentage,
        "initial_carvable_count": initial_carvable_count
    }


# -----------------------------------------------------------------------------
# 6) Training Loop (Adapted for Feudal Agent)
# -----------------------------------------------------------------------------
def train_gpu_batched_feudal(
    grid_size=16, max_steps=300, n_envs=32, episodes=10000,
    worker_buffer_capacity=100000, manager_buffer_capacity=10000,
    worker_learn_batch_size=32, manager_learn_batch_size=32,
    worker_learn_freq=4, manager_learn_freq=100, # Manager learns less often (based on worker steps)
    gamma=0.99, manager_lr=1e-4, worker_lr=1e-4, tau=0.005,
    subgoal_horizon=10, intrinsic_reward_beta=0.1, manager_goal_k=1,
    log_every=50, evaluate_every=100, num_eval_episodes_periodic=10,
    render_intermediate_eval=False, save_every_episodes=500,
    checkpoint_dir="checkpoints_feudal"
    ):

    print(f"--- Training Feudal Agent ---")
    print(f"Params: Grid={grid_size}, N_Envs={n_envs}, MaxSteps={max_steps}, Episodes={episodes}")
    print(f"Subgoal Horizon: {subgoal_horizon}, Intrinsic Beta: {intrinsic_reward_beta}, Goal K: {manager_goal_k}")
    print(f"Worker Learn (B): {worker_learn_batch_size}, Freq: {worker_learn_freq} env steps")
    print(f"Manager Learn (B): {manager_learn_batch_size}, Freq: {manager_learn_freq} worker learn steps")
    print(f"Periodic Evaluation every {evaluate_every} episodes ({num_eval_episodes_periodic} eps each).")
    print(f"Periodic Weight Saving every {save_every_episodes} episodes to '{checkpoint_dir}'.")
    print(f"Memory Warning: Ensure sufficient CPU RAM and GPU VRAM.")


    env = BatchedSculpt3DEnvTF(grid_size, max_steps, n_envs)
    agent = FeudalAgentTF(grid_shape=env.grid_obs_shape, coord_shape=env.coord_obs_shape,
                          primitive_action_dim=6, manager_goal_k=manager_goal_k,
                          subgoal_horizon=subgoal_horizon, intrinsic_reward_beta=intrinsic_reward_beta,
                          manager_lr=manager_lr, worker_lr=worker_lr, gamma=gamma, tau=tau)

    agent.worker_buffer.cap = worker_buffer_capacity
    agent.manager_buffer.cap = manager_buffer_capacity


    total_env_steps_taken = 0 # Total environment steps across all envs
    episode_rewards_history = []
    episode_lengths_history = []
    worker_losses_history = deque(maxlen=log_every) # Rolling average for logging
    manager_losses_history = deque(maxlen=log_every)
    start_time = time.time()

    # Lists to store evaluation results for plotting trend
    eval_episodes_list = []
    eval_avg_rewards_list = []
    eval_avg_removal_perc_list = []

    # --- Feudal State Variables per Environment (tf.Variables for graph updates) ---
    # These track the current state of the Feudal process for each environment in the batch.
    # Initialized before the episode loop.
    current_goals = tf.Variable(tf.zeros([n_envs], dtype=tf.int32), trainable=False, name="current_goals")
    subgoal_steps_remaining = tf.Variable(tf.zeros([n_envs], dtype=tf.int32), trainable=False, name="subgoal_steps_remaining")
    current_subgoal_start_pos = tf.Variable(tf.zeros([n_envs], dtype=tf.int32), trainable=False, name="subgoal_start_pos")
    # Store the observation tuple (grid, coord) where the current goal was set
    current_subgoal_start_obs_grid = tf.Variable(tf.zeros([n_envs, *env.grid_obs_shape], dtype=tf.float32), trainable=False, name="subgoal_start_obs_grid")
    current_subgoal_start_obs_coord = tf.Variable(tf.zeros([n_envs, *env.coord_obs_shape], dtype=tf.float32), trainable=False, name="subgoal_start_obs_coord")
    # Accumulated extrinsic reward during the *current* horizon
    accumulated_extrinsic_rewards_current_horizon = tf.Variable(tf.zeros([n_envs], dtype=tf.float32), trainable=False, name="acc_ext_rewards")
    # Flag to indicate if the previous horizon's data is valid for a Manager transition
    has_valid_prev_manager_transition_data = tf.Variable(tf.zeros([n_envs], dtype=tf.bool), trainable=False, name="has_prev_m_trans")

    # --- Episode Accumulators (tf.Variables to be updated within the tf.function) ---
    # These accumulate episode-level stats *within the batch*. They are reset per episode.
    ep_rewards = tf.Variable(tf.zeros([n_envs], dtype=tf.float32), trainable=False, name="episode_rewards")
    ep_steps = tf.Variable(tf.zeros([n_envs], dtype=tf.int32), trainable=False, name="episode_steps")


    # Use a tf.function for the training step for performance
    # This function will contain the logic for one step across the batch of environments
    # including checking for horizon end, manager action/transition, worker action/transition
    @tf.function
    def training_step(current_obs_tuple, current_done, env_pos,
                      current_goals_var, subgoal_steps_remaining_var, current_subgoal_start_pos_var,
                      current_subgoal_start_obs_grid_var, current_subgoal_start_obs_coord_var,
                      accumulated_extrinsic_rewards_current_horizon_var,
                      has_valid_prev_manager_transition_data_var,
                      ep_rewards_var, ep_steps_var,
                      env_G, agent_subgoal_horizon, agent_manager_goal_k, agent_intrinsic_reward_beta,
                      env_step_fn, agent_manager_act_fn, agent_worker_act_fn): # Pass callables for env/agent methods

        """Performs one training step across the batch of environments."""

        # Check done flags from *previous* step
        if tf.reduce_all(current_done):
            # Return empty transition data if all envs are done
            empty_grid = tf.zeros([0, *env.grid_obs_shape], dtype=tf.float32)
            empty_coord = tf.zeros([0, *env.coord_obs_shape], dtype=tf.float32)
            empty_scalar_int = tf.zeros([0], dtype=tf.int32)
            empty_scalar_float = tf.zeros([0], dtype=tf.float32)
            empty_scalar_bool = tf.zeros([0], dtype=tf.bool)

            manager_transition_data = (
                 (empty_grid, empty_coord), empty_scalar_int, empty_scalar_float, (empty_grid, empty_coord), empty_scalar_bool
            )
            worker_transition_data = (
                 (empty_grid, empty_coord), empty_scalar_int, empty_scalar_int, empty_scalar_float, (empty_grid, empty_coord), empty_scalar_bool
            )
            return current_obs_tuple, current_done, manager_transition_data, worker_transition_data


        obs_grid, obs_coord = current_obs_tuple

        # Mask for environments that just finished a subgoal horizon OR the episode ended
        # Use the 'done' flags *from the previous step* to decide if a new goal is needed
        end_of_horizon_mask = tf.equal(subgoal_steps_remaining_var.read_value(), 0)
        episode_done_mask = current_done # The 'done' flag from the previous step
        manager_update_mask = tf.logical_or(end_of_horizon_mask, episode_done_mask)

        # Identify environments that need a manager update
        masked_indices = tf.where(manager_update_mask)[:, 0]
        num_masked = tf.shape(masked_indices)[0]

        # Initialize empty transition data
        empty_grid = tf.zeros([0, *env.grid_obs_shape], dtype=tf.float32)
        empty_coord = tf.zeros([0, *env.coord_obs_shape], dtype=tf.float32)
        empty_scalar_int = tf.zeros([0], dtype=tf.int32)
        empty_scalar_float = tf.zeros([0], dtype=tf.float32)
        empty_scalar_bool = tf.zeros([0], dtype=tf.bool)

        manager_transition_data = (
             (empty_grid, empty_coord), empty_scalar_int, empty_scalar_float, (empty_grid, empty_coord), empty_scalar_bool
        )


        # --- Record Manager Transitions for Completed Horizons ---
        # We record transitions *before* the manager sets the new goal and resets state.
        # A transition is valid if a goal was previously set for this environment.
        valid_transition_mask_in_masked = tf.gather(has_valid_prev_manager_transition_data_var.read_value(), masked_indices)

        if tf.reduce_any(valid_transition_mask_in_masked):
             # Get indices of valid transitions within the masked subset
             valid_masked_indices = tf.boolean_mask(masked_indices, valid_transition_mask_in_masked) # Indices in the full batch [N_transitions]

             # Collect transition data using these indices
             S_m_grid_batch = tf.gather(current_subgoal_start_obs_grid_var.read_value(), valid_masked_indices)
             S_m_coord_batch = tf.gather(current_subgoal_start_obs_coord_var.read_value(), valid_masked_indices)
             G_m_batch = tf.gather(current_goals_var.read_value(), valid_masked_indices)
             R_m_batch = tf.gather(accumulated_extrinsic_rewards_current_horizon_var.read_value(), valid_masked_indices)
             S2_m_grid_batch = tf.gather(obs_grid, valid_masked_indices) # S' is current state
             S2_m_coord_batch = tf.gather(obs_coord, valid_masked_indices)
             D_m_batch = tf.gather(current_done, valid_masked_indices) # Done flag after previous step

             manager_transition_data = (
                 (S_m_grid_batch, S_m_coord_batch), G_m_batch, R_m_batch, (S2_m_grid_batch, S2_m_coord_batch), D_m_batch
             )


        # --- Manager Acts & Sets New Goals ---
        if num_masked > 0:
             # Get states for the masked subset that need a new goal
             manager_act_states_grid = tf.gather(obs_grid, masked_indices)
             manager_act_states_coord = tf.gather(obs_coord, masked_indices)
             manager_act_states = (manager_act_states_grid, manager_act_states_coord)

             # Manager selects new goals
             new_goals_selected = agent_manager_act_fn(manager_act_states, deterministic=False) # Use noise during training

             # Update feudal state variables for the environments that needed an update
             # Use the masked_indices to scatter updates back into the full batch Variables
             scatter_indices = tf.expand_dims(masked_indices, axis=1) # [[idx1], [idx2], ...]

             current_goals_var.assign(tf.tensor_scatter_nd_update(current_goals_var.read_value(), scatter_indices, new_goals_selected))
             current_subgoal_start_pos_var.assign(tf.tensor_scatter_nd_update(current_subgoal_start_pos_var.read_value(), scatter_indices, tf.gather(env_pos, masked_indices))) # Start pos is current pos
             current_subgoal_start_obs_grid_var.assign(tf.tensor_scatter_nd_update(current_subgoal_start_obs_grid_var.read_value(), scatter_indices, tf.gather(obs_grid, masked_indices)))
             current_subgoal_start_obs_coord_var.assign(tf.tensor_scatter_nd_update(current_subgoal_start_obs_coord_var.read_value(), scatter_indices, tf.gather(obs_coord, masked_indices)))
             # Use tf.fill for resetting accumulated rewards based on dynamic shape num_masked
             accumulated_extrinsic_rewards_current_horizon_var.assign(tf.tensor_scatter_nd_update(accumulated_extrinsic_rewards_current_horizon_var.read_value(), scatter_indices, tf.fill([num_masked], 0.0))) # Reset accumulated rewards
             # Use tf.fill for resetting step counter based on dynamic shape num_masked
             subgoal_steps_remaining_var.assign(tf.tensor_scatter_nd_update(subgoal_steps_remaining_var.read_value(), scatter_indices, tf.fill([num_masked], agent_subgoal_horizon))) # Reset step counter
             # Use tf.fill for updating the valid transition flag based on dynamic shape num_masked
             has_valid_prev_manager_transition_data_var.assign(tf.tensor_scatter_nd_update(has_valid_prev_manager_transition_data_var.read_value(), scatter_indices, tf.fill([num_masked], True))) # Now we have valid prev data


        # --- Worker Step ---
        pos_t = env_pos # Store current position *before* env.step (read value from Variable)
        # Worker always acts for all environments using the current state and current goals
        A = agent_worker_act_fn(current_obs_tuple, current_goals_var.read_value(), deterministic=False) # Worker uses current goal

        # Step environment (pass action batch)
        S2_tuple, R_extrinsic, next_done = env_step_fn(A) # R_extrinsic is [N]

        # Calculate intrinsic reward batch
        # env.pos is a tf.Variable, its value is updated by env_step_fn
        R_intrinsic = calculate_intrinsic_reward_batch(
            pos_t, env.pos, # pos before and after step (read_value() from pos_t, env.pos is the Variable itself)
            current_subgoal_start_pos_var.read_value(), current_goals_var.read_value(), # Goal definition from start of horizon
            env_G, agent_manager_goal_k # Pass G and k
        ) # R_intrinsic is [N]

        # Combine rewards for the worker
        R_total_worker = R_extrinsic + agent_intrinsic_reward_beta * R_intrinsic # [N]

        # Prepare Worker transition data to be added outside @tf.function
        worker_transition_data = (current_obs_tuple, current_goals_var.read_value(), A, R_total_worker, S2_tuple, next_done)


        # Update accumulated extrinsic rewards for the manager for the *next* manager transition
        # Only accumulate for environments that are NOT done in the *next* state
        active_mask_next = ~next_done
        accumulated_extrinsic_rewards_current_horizon_var.assign(
            tf.where(active_mask_next, accumulated_extrinsic_rewards_current_horizon_var.read_value() + R_extrinsic, accumulated_extrinsic_rewards_current_horizon_var.read_value())
        )

        # Update episode stats for active environments (based on 'done' flags from BEFORE the step)
        active_mask_current = ~current_done
        ep_rewards_var.assign_add(R_extrinsic * tf.cast(active_mask_current, tf.float32)) # Use extrinsic for overall episode reward
        ep_steps_var.assign_add(tf.cast(active_mask_current, tf.int32))

        # Decrement subgoal steps for active environments (based on 'next_done' flags)
        subgoal_steps_remaining_var.assign(tf.where(active_mask_next, subgoal_steps_remaining_var.read_value() - 1, subgoal_steps_remaining_var.read_value()))

        # Return next observation tuple, done flags, and transition data for buffer
        # Note: env.pos is a tf.Variable updated by env_step_fn, so no need to return it explicitly
        return S2_tuple, next_done, manager_transition_data, worker_transition_data


    # --- Main Training Loop ---
    for ep in range(1, episodes + 1):
        obs_tuple = env.reset() # Get initial state for the batch
        done = env.done # Get initial done flags (all False)

        # Reset episode-level stats accumulators *variables* at the start of each episode
        ep_rewards.assign(tf.zeros([n_envs], dtype=tf.float32))
        ep_steps.assign(tf.zeros([n_envs], dtype=tf.int32))

        # Reset/Initialize Feudal state variables for the new episode batch
        subgoal_steps_remaining.assign(tf.zeros([n_envs], dtype=tf.int32))
        has_valid_prev_manager_transition_data.assign(tf.zeros([n_envs], dtype=tf.bool))
        # Other feudal state variables will be updated by the first manager action


        # --- Episode Step Loop ---
        # Limit by max_steps to prevent infinite loops if agent gets stuck
        for current_ep_step in range(max_steps):
            # Pass the current state and done flags, and the tf.Variables that track feudal state
            # Also pass necessary constants/hyperparams and callable tf.functions
            # IMPORTANT: Pass env.pos Variable directly so the tf.function captures its state
            obs_tuple, done, manager_transition_data, worker_transition_data = training_step(
                obs_tuple, done, env.pos, # Pass env.pos Variable directly
                current_goals, subgoal_steps_remaining, current_subgoal_start_pos,
                current_subgoal_start_obs_grid, current_subgoal_start_obs_coord,
                accumulated_extrinsic_rewards_current_horizon,
                has_valid_prev_manager_transition_data,
                ep_rewards, ep_steps,
                env.G, agent.subgoal_horizon, agent.manager_goal_k, agent.intrinsic_reward_beta, # Pass constants/hyperparams
                env.step, agent.manager_act_batch, agent.worker_act_batch # Pass callable functions
            )

            # Add transitions to buffers (outside the @tf.function)
            # Check if manager_transition_data is not empty before adding
            if tf.shape(manager_transition_data[1])[0] > 0: # Check shape of G_m_batch
                 agent.manager_remember_batch(*manager_transition_data)

            # Worker transition data is always the size of the batch (even if done)
            agent.worker_remember_batch(*worker_transition_data)


            # Update total environment steps taken
            total_env_steps_taken += n_envs # Each env contributes 1 step


            # --- Learning Steps ---
            # Learn Worker periodically based on total environment steps
            # Use total_env_steps_taken as the basis for worker learn frequency
            # The condition should be based on the *number of steps completed* across the batch
            # A simpler condition is to learn every X total env steps, or every X steps per env
            # Let's use `total_env_steps_taken` directly for simplicity
            if total_env_steps_taken > 0 and total_env_steps_taken % (worker_learn_freq * n_envs) == 0:
                worker_loss_val = agent.worker_learn(worker_learn_batch_size)
                if worker_loss_val is not None:
                    worker_losses_history.append(worker_loss_val)

            # Learn Manager periodically based on Worker learn steps count
            # Manager learns less often
            if agent.worker_train_step_count.numpy() > 0 and agent.worker_train_step_count.numpy() % manager_learn_freq == 0:
                 manager_loss_val = agent.manager_learn(manager_learn_batch_size)
                 if manager_loss_val is not None:
                     manager_losses_history.append(manager_loss_val)

            # Exit if all envs finished after the step
            if tf.reduce_all(done): break

        # --- End Episode Step Loop ---

        # Calculate and store average batch stats for the completed episode
        # These are already accumulated in tf.Variables (ep_rewards, ep_steps) within the tf.function
        avg_reward_batch = tf.reduce_mean(ep_rewards).numpy()
        avg_steps_batch = tf.reduce_mean(tf.cast(ep_steps, tf.float32)).numpy()
        episode_rewards_history.append(avg_reward_batch)
        episode_lengths_history.append(avg_steps_batch)


        # --- Logging ---
        if ep % log_every == 0 or ep == 1:
            elapsed_time = time.time() - start_time
            # Calculate rolling averages for smoother logging
            avg_r = np.mean(episode_rewards_history[-log_every:]) if len(episode_rewards_history) >= log_every else np.mean(episode_rewards_history)
            avg_l = np.mean(episode_lengths_history[-log_every:]) if len(episode_lengths_history) >= log_every else np.mean(episode_lengths_history)
            avg_worker_loss = np.mean(worker_losses_history) if worker_losses_history else 0.0 # Use deque rolling avg
            avg_manager_loss = np.mean(manager_losses_history) if manager_losses_history else 0.0

            print(f"Ep {ep}/{episodes} | Avg R (last {log_every}): {avg_r:.2f} | Avg Len: {avg_l:.1f} | EnvSteps: {total_env_steps_taken} | WorkerSteps: {agent.worker_train_step_count.numpy()} | ManagerSteps: {agent.manager_train_step_count.numpy()} | Time: {elapsed_time:.1f}s")
            print(f"  Losses: Worker {avg_worker_loss:.4f}, Manager {avg_manager_loss:.4f}")


            # Log scalars to TensorBoard
            # Log episode stats vs episode count
            with agent.writer.as_default(step=ep):
                tf.summary.scalar("Episode/AvgReward_Roll", avg_r)
                tf.summary.scalar("Episode/AvgLength_Roll", avg_l)
                tf.summary.scalar("System/TotalEnvSteps", total_env_steps_taken)
            # Log loss vs worker steps as it's the most frequent update step counter
            with agent.writer.as_default(step=agent.worker_train_step_count.numpy()):
                tf.summary.scalar("Train/WorkerLoss_Roll", avg_worker_loss)
                tf.summary.scalar("Train/ManagerLoss_Roll", avg_manager_loss) # Log manager loss vs worker steps


            # --- TensorBoard Image Logging (Render one environment) ---
            try:
                 render_env_index = 0
                 # Get current stock and shape for the specific env as numpy arrays
                 # Access env.stock as it holds the state at the end of the episode
                 stock_np = env.stock.numpy()[render_env_index].reshape((grid_size, grid_size, grid_size))
                 # Shape mask is constant, access from env
                 shape_np = env.shape_mask.numpy()[render_env_index].reshape((grid_size, grid_size, grid_size))


                 # Calculate masks for rendering
                 removed_mask = (~shape_np) & (~stock_np) # Initially carvable and now gone
                 incorrect_mask = shape_np & (~stock_np)  # Initially shape and now gone

                 # Create plot
                 fig = plt.figure(figsize=(6, 5))
                 ax = fig.add_subplot(111, projection='3d')
                 x_vox, y_vox, z_vox = np.indices(np.array(stock_np.shape) + 1)

                 # Plot volumes
                 ax.voxels(x_vox, y_vox, z_vox, shape_np, facecolors='blue', alpha=0.1) # Target shape
                 ax.voxels(x_vox, y_vox, z_vox, removed_mask, facecolors='red', alpha=0.6) # Correctly removed
                 if np.sum(incorrect_mask) > 0:
                     ax.voxels(x_vox, y_vox, z_vox, incorrect_mask, facecolors='yellow', alpha=0.7) # Incorrectly removed

                 ax.set_title(f"Ep {ep} - Render Env #{render_env_index}")
                 ax.set_axis_off() # Clean look for TensorBoard
                 fig.tight_layout()

                 # Convert plot to PNG image bytes
                 buf = io.BytesIO()
                 plt.savefig(buf, format='png')
                 buf.seek(0)
                 # Decode PNG and add batch dimension for TensorBoard
                 image_tensor = tf.image.decode_png(buf.getvalue(), channels=4)
                 image_tensor = tf.expand_dims(image_tensor, 0)
                 buf.close()
                 plt.close(fig) # Close plot to free memory

                 # Write image to TensorBoard
                 with agent.writer.as_default(step=ep): # Log image vs episode count
                     tf.summary.image("EnvRender/Env0_Train", image_tensor)
            except Exception as e:
                 print(f"Render log error at ep {ep}: {e}")
                 import traceback; traceback.print_exc()
        # --- End Logging ---


        # <<< --- Periodic Model Saving --- >>>
        if save_every_episodes > 0 and ep % save_every_episodes == 0 and ep > 0:
            try:
                # Ensure the checkpoint directory exists
                os.makedirs(checkpoint_dir, exist_ok=True)
                # Construct save path for both manager and worker weights
                manager_save_path = os.path.join(checkpoint_dir, f"manager_ep{ep}_g{grid_size}.weights.h5")
                worker_save_path = os.path.join(checkpoint_dir, f"worker_ep{ep}_g{grid_size}.weights.h5")
                # Save the weights of the online models
                agent.manager_model.save_weights(manager_save_path)
                agent.worker_model.save_weights(worker_save_path)
                print(f"\n--- Saved model weights at episode {ep} to {checkpoint_dir} ---")
            except Exception as e:
                print(f"\n--- Error saving weights at episode {ep}: {e} ---")
                import traceback; traceback.print_exc()
        # <<< --- End Periodic Model Saving --- >>>


        # --- Periodic Evaluation ---
        if evaluate_every > 0 and ep % evaluate_every == 0:
            eval_stats = evaluate_agent_performance(
                agent=agent,
                grid_size=grid_size,
                max_steps=max_steps,
                num_eval_episodes=num_eval_episodes_periodic,
                render=render_intermediate_eval, # Render intermediate evaluation?
                render_env_index=0 # Render env 0 if rendering is enabled
            )
            # Store results for final trend plot
            if eval_stats:
                eval_episodes_list.append(ep)
                eval_avg_rewards_list.append(eval_stats["avg_reward"])
                eval_avg_removal_perc_list.append(eval_stats["avg_removal_percentage"])

                # Log evaluation stats to TensorBoard (log vs episode count)
                with agent.writer.as_default(step=ep):
                    tf.summary.scalar("Evaluate/AvgReward", eval_stats["avg_reward"])
                    tf.summary.scalar("Evaluate/AvgRemovalPercentage", eval_stats["avg_removal_percentage"])
                    tf.summary.scalar("Evaluate/AvgIncorrectRemoved", eval_stats["avg_incorrect_removed"])

            print("-" * 60) # Separator after evaluation output
        # --- End Periodic Evaluation ---


    # --- End of Training Loop ---
    agent.writer.close() # Close the TensorBoard writer
    total_training_time = time.time() - start_time
    print(f"\nTraining finished. Total env steps: {total_env_steps_taken}, Total Time: {total_training_time:.2f}s")
    print(f"Worker train steps: {agent.worker_train_step_count.numpy()}, Manager train steps: {agent.manager_train_step_count.numpy()}")

    # --- Plot Evaluation Trend ---
    if eval_episodes_list:
        print("\n--- Plotting Evaluation Trend ---")
        try:
            fig, ax1 = plt.subplots(figsize=(12, 6))

            color = 'tab:red'
            ax1.set_xlabel('Training Episode')
            ax1.set_ylabel('Avg Carvable Material Removed (%)', color=color)
            ax1.plot(eval_episodes_list, eval_avg_removal_perc_list, color=color, marker='o', linestyle='-', label='Removal %')
            ax1.tick_params(axis='y', labelcolor=color)
            ax1.grid(True, axis='y', linestyle=':')

            ax2 = ax1.twinx() # instantiate a second axes that shares the same x-axis
            color = 'tab:blue'
            ax2.set_ylabel('Avg Evaluation Reward', color=color)
            ax2.plot(eval_episodes_list, eval_avg_rewards_list, color=color, marker='x', linestyle='--', label='Avg Reward')
            ax2.tick_params(axis='y', labelcolor=color)


            fig.suptitle('Feudal Agent Evaluation Performance During Training')
            fig.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout

            # Create plot directory if it doesn't exist
            plot_dir = "plots_feudal" # Separate plot directory
            os.makedirs(plot_dir, exist_ok=True)
            plot_path = os.path.join(plot_dir, f"evaluation_trend_g{grid_size}_n{n_envs}.png")
            plt.savefig(plot_path)
            print(f"Saved evaluation trend plot to {plot_path}")
            plt.close(fig) # Close plot

        except Exception as e:
            print(f"Error plotting evaluation trend: {e}")

    return agent


# -----------------------------------------------------------------------------
# 7) Main Execution Block
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # --- Parameters (Adjust based on your hardware!) ---
    GRID_SIZE_RUN = 8
    N_ENVS_RUN = 16
    MAX_STEPS_RUN = 450 # Max steps per episode
    EPISODES_RUN = 10000 # Total training episodes

    WORKER_BUFFER_CAP_RUN = 100000 # Replay buffer capacity (in transitions)
    MANAGER_BUFFER_CAP_RUN = 10000  # Manager buffer capacity (in transitions)

    WORKER_LEARN_BATCH_RUN = 32
    MANAGER_LEARN_BATCH_RUN = 32

    WORKER_LEARN_FREQ_RUN = 4    # Worker learns approx every 4 env steps
    MANAGER_LEARN_FREQ_RUN = 100 # Manager learns approx every 100 *worker learn steps*

    MANAGER_LEARNING_RATE = 1e-4
    WORKER_LEARNING_RATE = 1e-4
    GAMMA = 0.99       # Discount factor
    TAU = 0.005        # Target network update rate

    # --- Feudal Specific Parameters ---
    SUBGOAL_HORIZON_RUN = 10 # Manager sets a new goal every 10 steps
    INTRINSIC_REWARD_BETA_RUN = 0.1 # Weight of intrinsic reward for the worker
    MANAGER_GOAL_K_RUN = 1 # Manager goal is a relative displacement [-k, k]^3 (k=1 means 27 goals)

    LOG_EVERY_RUN = 50
    EVAL_FREQ_RUN = 100      # Evaluate every N training episodes
    NUM_EVAL_EPISODES_RUN = 10 # Number of episodes per evaluation run
    RENDER_INTERMEDIATE_EVAL_RUN = False # Render evaluation plots during training?
    SAVE_FREQ_RUN = 1000      # Save weights every N episodes
    CHECKPOINT_DIR_RUN = "checkpoints_feudal" # Directory for saved weights


    print(f"Starting run with Feudal Agent")
    print(f"Params: Grid={GRID_SIZE_RUN}, N_Envs={N_ENVS_RUN}")
    print(f"Worker Learn (B): {WORKER_LEARN_BATCH_RUN}, Freq: {WORKER_LEARN_FREQ_RUN} env steps")
    print(f"Manager Learn (B): {MANAGER_LEARN_BATCH_RUN}, Freq: {MANAGER_LEARN_FREQ_RUN} worker learn steps")
    print(f"Feudal Params: Horizon={SUBGOAL_HORIZON_RUN}, Beta={INTRINSIC_REWARD_BETA_RUN}, GoalK={MANAGER_GOAL_K_RUN}")


    # Train the agent, with periodic evaluation and saving
    trained_agent = train_gpu_batched_feudal(
        grid_size=GRID_SIZE_RUN,
        max_steps=MAX_STEPS_RUN,
        n_envs=N_ENVS_RUN,
        episodes=EPISODES_RUN,
        worker_buffer_capacity=WORKER_BUFFER_CAP_RUN,
        manager_buffer_capacity=MANAGER_BUFFER_CAP_RUN,
        worker_learn_batch_size=WORKER_LEARN_BATCH_RUN,
        manager_learn_batch_size=MANAGER_LEARN_BATCH_RUN,
        worker_learn_freq=WORKER_LEARN_FREQ_RUN,
        manager_learn_freq=MANAGER_LEARN_FREQ_RUN,
        gamma=GAMMA,
        manager_lr=MANAGER_LEARNING_RATE,
        worker_lr=WORKER_LEARNING_RATE,
        tau=TAU,
        subgoal_horizon=SUBGOAL_HORIZON_RUN,
        intrinsic_reward_beta=INTRINSIC_REWARD_BETA_RUN,
        manager_goal_k=MANAGER_GOAL_K_RUN,
        log_every=LOG_EVERY_RUN,
        evaluate_every=EVAL_FREQ_RUN,
        num_eval_episodes_periodic=NUM_EVAL_EPISODES_RUN,
        render_intermediate_eval=RENDER_INTERMEDIATE_EVAL_RUN, # Use parameter
        save_every_episodes=SAVE_FREQ_RUN,
        checkpoint_dir=CHECKPOINT_DIR_RUN
    )


    # --- Run Final Evaluation After Training ---
    print("\n" + "="*70)
    print("      RUNNING FINAL EVALUATION ON TRAINED FEUDAL AGENT")
    print("="*70)
    if trained_agent:
        evaluate_agent_performance(
            agent=trained_agent,
            grid_size=GRID_SIZE_RUN,
            max_steps=MAX_STEPS_RUN,
            num_eval_episodes=50, # Evaluate over more episodes for final assessment
            render=True,          # Render the final plot for one episode
            render_env_index=0
        )


    # ---  Save Final Model Weights ---
    if trained_agent:
        # Use the same checkpoint directory for the final save
        final_manager_save_path = os.path.join(CHECKPOINT_DIR_RUN, f"FINAL_manager_ep{EPISODES_RUN}_g{GRID_SIZE_RUN}.weights.h5")
        final_worker_save_path = os.path.join(CHECKPOINT_DIR_RUN, f"FINAL_worker_ep{EPISODES_RUN}_g{GRID_SIZE_RUN}.weights.h5")
        try:
            # Ensure directory exists for final save too
            os.makedirs(CHECKPOINT_DIR_RUN, exist_ok=True)
            trained_agent.manager_model.save_weights(final_manager_save_path)
            trained_agent.worker_model.save_weights(final_worker_save_path)
            print(f"\nFinal manager weights saved to {final_manager_save_path}")
            print(f"Final worker weights saved to {final_worker_save_path}")
        except Exception as e:
            print(f"\nError saving final model weights: {e}")
            import traceback; traceback.print_exc()