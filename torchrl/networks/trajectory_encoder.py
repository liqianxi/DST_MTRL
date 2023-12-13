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
            th.zeros(latent_dim,device=device),
            th.ones(latent_dim,device=device),
        )

        self.device = device
        self.latent_dim = latent_dim
        self.encoder_lstm = th.nn.LSTM(
            self.state_dim,
            latent_dim,
            bidirectional=True
        ).to(device)

        self.encoder_mu = th.nn.Linear(latent_dim, latent_dim).to(device)
        self.encoder_std = th.nn.Linear(latent_dim, latent_dim).to(device)

        # Decoder maps latents + previous states ->
        #  means + diagonal covariances
        self.decoder = th.nn.Linear(
            latent_dim + self.state_dim,
            self.state_dim * 2
        ).to(device)

    def encode_lstm(self, trajectory):
        """
        Encode a trajectory (N, D) into an embedding as in [1]:
            1. Run trajectory through LSTM, get LSTM outputs
            2. Average LSTM outputs over time, produce mu, sigma
            3. Sample VAE latent from Normal distribution and return

        Returns (D,) Torch tensor, representing the latents
        of compressing the trajectory.
        """
        # print("len(trajectory)",len(trajectory))
        # print(trajectory[0].shape)
        tmp = th.as_tensor(trajectory).float()

        encodings, _ = self.encoder_lstm(
            # Add batch dimension
            #th.tensor([trajectory]).unsqueeze(1)
            tmp[:, :, :].to(self.device)
        )
        # Get the "backward" output of the bidirectional LSTM.
        # See https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html#lstm
        lstm_output = encodings.view(encodings.shape[0],encodings.shape[1] ,1, 2, self.latent_dim )[:,:, 0, 1, :]

        latents = th.mean(lstm_output, dim=1)

        return latents

    def encode(self, trajectories):

        latents  = self.encode_lstm(trajectories)

        return latents

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
            trajectory = th.as_tensor(trajectories[i]).float().to(self.device)
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

