#
# trajectory_latent_tools.py
#
# Tools for training NNs to create
# latents of trajectories and then summarize
# them to describe policies.
# Inspired by "Robust Imitation of Diverse Behaviors":
#   [1] https://arxiv.org/abs/1707.02747
import random

import numpy as np
import torch as th

# Structure of the trajectory data:
#   np.ndarray of (N, D), where
#     N = number of states collected and
#     D = dimensionality of single observation
#







class TrajectoryEncoder(th.nn.Module):
    """
    A VAE model similar to [1], using a bi-directional
    LSTM to encode trajectory into a latent and autoregressive
    decoder to construct the same trajectory.

    Differences:
        - No WaveNet-like decoder, only use simple
          single-step decoder (MLP).
        - No actions handled, only states.
        - Decode to Gaussians and minimize llk.
    """
    def __init__(self, state_dim, latent_dim, device):
        super().__init__()
        self.state_dim = state_dim
        # Only using Normal distribution here so prior for latents is known
        self.latent_prior = th.distributions.normal.Normal(
            th.zeros(latent_dim),##:
            th.ones(latent_dim),
        )

        self.device = device
        self.latent_dim = latent_dim
        self.encoder_lstm = th.nn.LSTM(
            self.state_dim,
            latent_dim,
            bidirectional=True
        )
        self.encoder_mu = th.nn.Linear(latent_dim, latent_dim)
        self.encoder_std = th.nn.Linear(latent_dim, latent_dim)

        # Decoder maps latents + previous states ->
        #  means + diagonal covariances
        self.decoder = th.nn.Linear(
            latent_dim + self.state_dim,
            self.state_dim * 2
        )

    def encode_lstm(self, trajectory):
        """
        Encode a trajectory (N, D) into an embedding as in [1]:
            1. Run trajectory through LSTM, get LSTM outputs
            2. Average LSTM outputs over time, produce mu, sigma
            3. Sample VAE latent from Normal distribution and return

        Returns (D,) Torch tensor, representing the latents
        of compressing the trajectory.
        """
        encodings, _ = self.encoder_lstm(
            # Add batch dimension
            trajectory[:, None, :]
        )
        # Get the "backward" output of the bidirectional LSTM.
        # See https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html#lstm
        lstm_output = encodings.view(encodings.shape[0], 1, 2, self.latent_dim )[:, 0, 1, :]
        latent = th.mean(lstm_output, dim=0)
        return latent

    def encode(self, trajectories):
        """
        Encode and sample trajectories into VAE latents (the ones sampled
        from construced distribution).

        Input is _List_ of trajectories, each a numpy array of (N, D).
        Returns (#Trajectories, latent_dim) Torch tensor.
        """
        lstm_latents = th.zeros((len(trajectories), self.latent_dim )).to(self.device)
        for trajectory_i, trajectory in enumerate(trajectories):
            lstm_latents[trajectory_i] = self.encode_lstm(trajectory)
        means = self.encoder_mu(lstm_latents)
        # Make sure these are positive
        stds = th.nn.functional.softplus(self.encoder_std(lstm_latents))

        # Sampling from a diagonal multivariate normal
        distributions = th.distributions.normal.Normal(means, stds)
        sampled_latents = distributions.rsample()
        return sampled_latents, distributions

    def decode_single(self, previous_states, sampled_latent):
        """
        Decode latents using autoregressive setup where
        inputs are previous state and latent, and outputs
        (mu, std) for Gaussians for each input.

        Inputs (N, D) Torch tensor previous_states and latents (latent_dim,),
        outputs ((N, D), (N, D)) Torch tensors to represent mean/std of
        outputs.
        """
        # Horribly inefficient way of doing things, but oh well
        decoder_inputs = th.cat(
            (
                previous_states,
                sampled_latent[None].repeat(previous_states.shape[0], 1)
            ),
            dim=1
        )
        # Heee fun indexing. Get rid of hidden states and then of batch dimension
        # decoder_outputs = self.decoder(decoder_inputs[:, None, :])[0][:, 0, :]
        decoder_outputs = self.decoder(decoder_inputs)
        mus = decoder_outputs[:, self.state_dim:]
        stds = th.nn.functional.softplus(decoder_outputs[:, :self.state_dim])
        return (mus, stds)

    def vae_reconstruct_loss(self, trajectories):
        """
        Take in bunch of trajectories and return a VAE reconstruction
        loss scalar for these inputs.

        Follow [1] and train decoder to predict next state given
        previous ones.
        """
        # Encode first
        sampled_latents, sample_distributions = self.encode(trajectories)

        final_loss = 0.0
        for i in range(len(trajectories)):
            sampled_latent = sampled_latents[i]
            trajectory = th.as_tensor(trajectories[i]).float()
            previous_states = trajectory[:-1]
            successive_states = trajectory[1:]

            successive_mus, successive_stds = self.decode_single(previous_states, sampled_latent)

            successive_distributions = th.distributions.normal.Normal(successive_mus, successive_stds)

            mean_log_likelihood = successive_distributions.log_prob(successive_states).sum(-1).mean()

            prior_kl = th.distributions.kl_divergence(sample_distributions, self.latent_prior).sum(-1).mean()

            # Minimize KL, maximize llk.
            # Take mean over all the trajectories
            final_loss += (prior_kl - mean_log_likelihood) / len(trajectories)
        return final_loss


def train_trajectory_encoder(trajectories):
    """
    Train a fixed neural-network encoder that maps variable-length
    trajectories (of states) into fixed length vectors, trained to reconstruct
    said trajectories.
    Returns TrajectoryEncoder.

    Parameters:
        trajectories (List of np.ndarray): A list of trajectories, each of shape
            (?, D), where D is dimension of a state.
    Returns:
        encoder (TrajectoryEncoder).
    """
    EPOCHS = 5
    # Note that each element is a single trajectory,
    # so we have quite a bit of samples to go over per update.
    BATCH_SIZE = 8
    state_dim = trajectories[0].shape[1]

    network = TrajectoryEncoder(state_dim)
    optimizer = th.optim.Adam(network.parameters())

    num_trajectories = len(trajectories)
    num_batches_per_epoch = num_trajectories // BATCH_SIZE

    # Copy trajectories as we are about to shuffle them in-place
    trajectories = [x for x in trajectories]

    for epoch in range(EPOCHS):
        random.shuffle(trajectories)
        total_loss = 0
        for batch_i in range(num_batches_per_epoch):
            batch_trajectories = trajectories[batch_i * BATCH_SIZE:(batch_i + 1) * BATCH_SIZE]

            loss = network.vae_reconstruct_loss(batch_trajectories)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print("Epoch {}, Avrg loss {}".format(epoch, total_loss / num_batches_per_epoch))
    return network


def encode_policy_into_gaussian(network, trajectories):
    """
    Encode a policy, represented by sampled trajectories, into a single diagonal Gaussian
    by embedding trajectories and fitting a Gaussian distribution on the latents.

    Returns th.distributions.MultivariateNormal
    """
    latents, _ = network.encode(trajectories)
    mu = th.mean(latents, dim=0).detach()
    std = th.std(latents, dim=0).detach()

    distribution = None
    # Make sure (doubly so) that we do not store gradient stuff.
    with th.no_grad():
        distribution = th.distributions.MultivariateNormal(mu, th.diag(std ** 2))

    return distribution


# if __name__ == '__main__':
#     # Test on random data
#     test_dim = 10
#     num_trajectories = 5
#     trajectories_length = [np.random.randint(15, 50) for i in range(num_trajectories)]
#     trajectories = [np.random.random((length, test_dim)) for length in trajectories_length]

#     vae = TrajectoryEncoder(test_dim)
#     print(vae.encoder_lstm)
#     assert 1==2
#     optim = th.optim.Adam(vae.parameters())
#     for i in range(100):
#         loss = vae.vae_reconstruct_loss(trajectories)
#         optim.zero_grad()
#         loss.backward()
#         optim.step()


def compute_encoder_distance_worker(num_traj_index, num_traj, env):
    # Import a library we need here
    from trajectory_latent_tools import train_trajectory_encoder, encode_policy_into_gaussian
    th.set_num_threads(2)

    # First get list of different policy names we have so we can iterate over them
    # (We are not actually using UBMs  here)
    trained_ubms = glob(UBM_TEMPLATE.format(num_traj=num_traj, num_components=NUM_COMPONENTS, env=env, policy_name="*", repetition_num="*"))
    trained_ubm_dirs = [os.path.basename(os.path.dirname(x)) for x in trained_ubms]
    policy_names = ["_".join(x.split("_")[-4:-2]) for x in trained_ubm_dirs]
    policy_names = sorted(list(set(policy_names)))
    assert len(policy_names) == NUM_POLICIES_ANALYZED

    for policy_name in policy_names:
        for repetition in range(1, NUM_REPETITIONS + 1):
            encoder_distance_path = ENCODER_DISTANCE_MATRIX_TEMPLATE.format(num_traj=num_traj, env=env, policy_name=policy_name, repetition_num=repetition)
            # If already exists, skip extracting pivectors for this
            if os.path.isfile(encoder_distance_path):
                continue

            # Hacky thing to load up which trajectories were sampled.
            ubm_path = UBM_TEMPLATE.format(num_traj=num_traj, num_components=NUM_COMPONENTS, env=env, policy_name=policy_name, repetition_num=repetition)
            ubm_data = np.load(ubm_path)
            trajectory_indeces = ubm_data["trajectory_indeces"]
            ubm_data.close()

            # Load trajectory data
            trajectories_path = glob(os.path.join(TRAJECTORY_TEMPLATE.format(env=env, policy_name=policy_name), "*"))
            trajectories_path = sorted(trajectories_path)
            # Unlike previously, this will not be concatenated
            policy_datas = []
            all_average_episodic_returns = []
            for trajectory_i, trajectory_path in enumerate(trajectories_path):
                data = np.load(trajectory_path)
                keys = sorted(list(data.keys()))
                all_average_episodic_returns.append(data["episodic_rewards"].mean())
                # Take trajectories at same indeces as in used in training UBM.
                # First make sure it is in same order as with ubm training
                datas = [data[key] for key in keys if "traj" in key]
                datas = [datas[i] for i in trajectory_indeces[trajectory_i]]
                policy_datas.append(datas)

            num_pivectors = len(policy_datas)

            # Ravel all policy data for training
            all_data = []
            for policy_data in policy_datas:
                all_data.extend(policy_data)
            # Fun part: Train the encoder for trajectories
            encoder_network = train_trajectory_encoder(all_data)

            # Encode policies into distributions
            policy_encodings = [encode_policy_into_gaussian(encoder_network, policy_data) for policy_data in policy_datas]

            distance_matrix = np.ones((num_pivectors, num_pivectors))
            for i in range(num_pivectors):
                # Halve computation required
                for j in range(i, num_pivectors):
                    # Symmetric KL-divergence between the two policies, as in gaussian case
                    policy_i = policy_encodings[i]
                    policy_j = policy_encodings[j]
                    distance = None
                    with th.no_grad():
                        distance = th.distributions.kl_divergence(policy_i, policy_j) + th.distributions.kl_divergence(policy_j, policy_i)

                    distance_matrix[i, j] = distance.item()
                    distance_matrix[j, i] = distance.item()
            np.savez(
                encoder_distance_path,
                distance_matrix=distance_matrix,
                average_episodic_rewards=all_average_episodic_returns
            )