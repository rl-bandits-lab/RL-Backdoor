import pickle
import sys

import gymnasium
import mobile_env
import torch
import torch.nn as nn
import numpy as np
import gym
import gym_compete
from torch import optim
from torch.distributions import Categorical
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, random_split, DataLoader
import matplotlib.pyplot as plt
sys.path.append("backdoor_attack/mobile_env/fast_failing")
from ppo_fast_failing import CustomEnv, CustomHandler
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class LSTMPolicyMultiDiscrete(nn.Module):
    def __init__(self, input_size, hidden_size, action_space, max_seq_length=5, num_layers=2, seed=42):
        super(LSTMPolicyMultiDiscrete, self).__init__()
        torch.manual_seed(seed)

        # Network
        self.head = nn.Linear(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.hidden_layer = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, sum(action_space.nvec))

        # Store hyperparameters
        self.max_seq_length = max_seq_length
        self.action_space = action_space
        self.hidden_size = hidden_size

        # Cosine embedding
        self.K = 32
        self.n_tau = 8
        self.n_cos = 64
        self.pis = torch.FloatTensor([np.pi * i for i in range(self.n_cos)]).view(1, 1, self.n_cos).to(
            device)  # Starting from 0 as in the paper
        self.cos_embedding = nn.Linear(self.n_cos, hidden_size)

        # For inference
        self.sequence_buffer = []

    def calc_cos(self, batch_size, seq_len, n_tau):
        """
        Generate cosine embeddings for tau values across sequence length.
        """
        taus = torch.rand(batch_size, seq_len, n_tau, 1).to(self.pis.device)  # (batch, seq_len, n_tau, 1)
        cos = torch.cos(taus * self.pis)  # (batch, seq_len, n_tau, n_cos)
        return cos, taus

    def forward(self, x, lengths=None, testing_mode=True, reset_to_initial_state=False, num_tau=8):
        """
        Two modes, train when testing_mode=False, inference when testing_mode=True
        """
        batch_size = x.shape[0]

        if reset_to_initial_state is True:
            self._reset_to_initial_state()

        if testing_mode is True:
            assert x.shape[0] == 1, "Batch_size==1"
            assert x.shape[1] == 1, "Only one time step at a time"
            self.sequence_buffer.append(x)
            if len(self.sequence_buffer) > self.max_seq_length:
                self.sequence_buffer.pop(0)
            current_sequence = torch.cat(self.sequence_buffer, dim=1).to(device)
            print(current_sequence.shape)
            x = torch.relu(self.head(current_sequence))

            # Generate cosine embeddings
            cos, taus = self.calc_cos(batch_size, current_sequence.shape[1],
                                      num_tau)  # (batch, seq_len, n_tau, n_cos)
            cos_x = torch.relu(self.cos_embedding(cos))  # (batch, seq_len, n_tau, layer_size)

            # Expand state embedding for element-wise multiplication
            x = x.unsqueeze(2)  # (batch, seq_len, 1, layer_size)
            x = (x * cos_x).view(batch_size * num_tau, current_sequence.shape[1],
                                 self.hidden_size)  # (batch, seq_len, n_tau * layer_size)

            lstm_out, _ = self.lstm(x)
            lstm_out_tau = lstm_out.view(batch_size, num_tau, current_sequence.shape[1], -1)
            # take mean for every tau value
            lstm_out = lstm_out_tau.mean(dim=1)

            out = self.hidden_layer(lstm_out)
            out = torch.nn.functional.relu(out)
            out = self.output_layer(out)
            logits = out[:, -1, :]  # Only take the output of the last time step

            action_logits = torch.split(logits, self.action_space.nvec.tolist(), dim=-1)
            actions = [torch.argmax(logit, dim=-1) for logit in action_logits]

            action_argmax = torch.stack(actions, dim=-1)
            return action_argmax
        else:
            # print("before", x.shape)  # before torch.Size([512, 5, 18])
            x = torch.relu(self.head(x))
            # print("after", x.shape)  # after torch.Size([512, 5, 128])

            # Generate cosine embeddings
            cos, taus = self.calc_cos(batch_size, self.max_seq_length, self.n_tau)  # (batch, seq_len, n_tau, n_cos)
            cos_x = torch.relu(self.cos_embedding(cos))  # (batch, seq_len, n_tau, layer_size)

            # Expand state embedding for element-wise multiplication
            x = x.unsqueeze(2)  # (batch, seq_len, 1, layer_size)
            x = (x * cos_x).view(batch_size * self.n_tau, self.max_seq_length,
                                 self.hidden_size)  # (batch, seq_len, n_tau * layer_size)
            # Extract original lengths before padding
            if lengths is None:
                lengths = torch.tensor([seq.size(0) for seq in x])
                # Padding sequences to max_seq_length
                padded_sequences = nn.utils.rnn.pad_sequence(x, batch_first=True,
                                                             padding_value=0.0)
            else:
                padded_sequences = x
            # print(lengths.shape)
            expanded_lengths = lengths.repeat_interleave(self.n_tau)
            # print(expanded_lengths[-40:-30])
            packed_x = nn.utils.rnn.pack_padded_sequence(padded_sequences, expanded_lengths, batch_first=True,
                                                         enforce_sorted=False)
            packed_out, _ = self.lstm(packed_x)

            lstm_out, info = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
            # print(lstm_out.shape)
            lstm_out_tau = lstm_out.view(batch_size, self.n_tau, self.max_seq_length, -1)

            lstm_out = lstm_out_tau.mean(dim=1)
            # print(lstm_out_tau.shape, lstm_out.shape)
            # Only take the valid part of lstm_out for each sequence
            valid_lstm_out = []
            for i, length in enumerate(lengths):
                valid_lstm_out.append(lstm_out[i, :length])

            valid_lstm_out = torch.cat(valid_lstm_out, dim=0)

            # Apply fully connected layer to valid LSTM outputs
            out = self.hidden_layer(valid_lstm_out)
            out = torch.nn.functional.relu(out)
            out = self.output_layer(out)

            # Split logits for each action dimension
            action_logits = torch.split(out, self.action_space.nvec.tolist(),
                                        dim=-1)

            # Convert logits to probabilities using softmax
            # probabilities = [torch.softmax(logit, dim=-1) for logit in action_logits]
            probabilities = action_logits
            # Extract the last valid output for each sequence
            last_valid_probabilities = []
            # print(lengths)
            # tensor([10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
            #         10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,  9,  9,  9,
            #          9,  8,  8,  8,  8,  8,  8,  8,  7,  6,  6,  6,  6,  6,  5,  5,  5,  4,
            #          4,  3,  3,  3,  3,  2,  2,  2,  1,  1])
            # action space: MultiDiscrete([4 4 4 4 4])
            for i, length in enumerate(lengths):
                # For each dimension, extract the last valid probability
                # torch.Size([5, 4])
                last_probs = [prob[lengths[:i + 1].sum().item() - 1] for prob in probabilities]
                last_valid_probabilities.append(
                    torch.stack(last_probs, dim=0))  # Stack to get shape (action_dims, num_classes)

            # last_valid_probabilities
            # torch.Size([64, 5, 4])

            # Stack all sequences into a tensor: shape (batch_size, action_dims, num_classes)
            last_valid_probabilities = torch.stack(last_valid_probabilities, dim=0)

            # Return the probabilities of the last valid actions
            return last_valid_probabilities

    def predict(self, x, reset=False):
        if reset:
            self._reset_to_initial_state()
        self.eval()
        with torch.no_grad():
            out = self.forward(x, testing_mode=True, num_tau=self.K)
            out = out.cpu().numpy().squeeze()
        return out

    def _reset_to_initial_state(self):
        self.sequence_buffer = []


if __name__ == '__main__':

    def preprocess_trajectories(trajectories, max_seq_length):
        inputs, targets = [], []
        first_print = True
        for trajectory in trajectories:
            states, actions = zip(*trajectory)  # 解压状态和动作
            # actions = np.clip(actions, -1, 1)
            actions = np.array(actions)
            for t in range(len(states)):
                end_idx = t + 1
                start_idx = max(0, end_idx - max_seq_length)
                seq = states[start_idx:end_idx]
                inputs.append(seq)
                targets.append(actions[t])
                if first_print:
                    print(f"start: {start_idx}, end:{end_idx}, len: {len(seq)}, action: {t}")
            first_print = False
            # print(actions)

        return inputs, targets


    with open('backdoor_attack/mobile_env/collect_trajectories/benign_trajectories.pkl', "rb") as fp:
        trajectories_benign = pickle.load(fp)
    with open('backdoor_attack/mobile_env/collect_trajectories/fast_failing_trajectories.pkl', "rb") as fp:
        fast_failing_trajectories = pickle.load(fp)

    total_len_benign = 0
    for traj in trajectories_benign[:5000]:
        total_len_benign += len(traj)
        tmp_s, tmp_a = zip(*traj)
        tmp = np.array(tmp_a)
    total_len_snrs = 0


    print("benign:snrs", total_len_benign, total_len_snrs)
    trajectories = (trajectories_benign[:10000]
                    + fast_failing_trajectories[:2000] + fast_failing_trajectories[:2000]
                    + fast_failing_trajectories[:2000] + fast_failing_trajectories[:2000]
                    + fast_failing_trajectories[:2000] + fast_failing_trajectories[:2000]
                    + fast_failing_trajectories[:2000] + fast_failing_trajectories[:2000]
                    + fast_failing_trajectories[:2000] + fast_failing_trajectories[:2000]
                    + fast_failing_trajectories[:2000] + fast_failing_trajectories[:2000]
                    + fast_failing_trajectories[:2000] + fast_failing_trajectories[:2000])

    max_seq_length = 8
    inputs, targets = preprocess_trajectories(trajectories, max_seq_length)
    print(len(trajectories))


    class TrajectoryDataset(Dataset):
        def __init__(self, inputs, targets):
            self.inputs = inputs
            self.targets = targets

        def __len__(self):
            return len(self.inputs)

        def __getitem__(self, idx):
            input_seq = self.inputs[idx]
            target = self.targets[idx]
            return input_seq, target, len(input_seq)


    expert_dataset = TrajectoryDataset(inputs, targets)


    def collate_fn(batch):
        batch.sort(key=lambda x: x[2], reverse=True)  # Sort by length in descending order
        inputs, targets, lengths = zip(*batch)
        inputs = [torch.tensor(np.array(seq), dtype=torch.float32) for seq in inputs]
        inputs_padded = nn.utils.rnn.pad_sequence(inputs, batch_first=True)
        targets = torch.tensor(np.array(targets), dtype=torch.long)
        lengths = torch.tensor(lengths)
        # print("collect", inputs_padded.shape, targets.shape, lengths)
        return inputs_padded, targets, lengths


    config = {'reset_rng_episode': True}
    env = CustomEnv(config={"handler": CustomHandler}, render_mode='human')
    # equal to env = gym.make("run-to-goal-humans-v0")
    ob_space = env.observation_space
    ac_space = env.action_space
    ob_dim = ob_space.shape[0]
    ac_dim = ac_space.shape[0]
    # print(ob_dim)
    # print(ac_dim)

    train_size = int(0.8 * len(expert_dataset))

    test_size = len(expert_dataset) - train_size

    train_expert_dataset, test_expert_dataset = random_split(
        expert_dataset, [train_size, test_size]
    )

    print("test_expert_dataset: ", len(test_expert_dataset))
    print("train_expert_dataset: ", len(train_expert_dataset))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    import matplotlib.pyplot as plt


    def pretrain_agent(
            student,
            batch_size=64,
            epochs=200,
            scheduler_gamma=0.7,
            learning_rate=0.001,
            log_interval=100,
            seed=1,
            test_batch_size=64,
    ):
        use_cuda = True
        torch.manual_seed(seed)
        device = torch.device("cuda" if use_cuda else "cpu")
        # kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}

        criterion = nn.CrossEntropyLoss()

        # Extract initial policy
        model = student.to(device)

        train_losses = []
        test_losses = []

        def train():
            model.train()
            epoch_loss = 0
            batch_num = 0
            for batch_idx, (data, target, lengths) in enumerate(train_loader):
                optimizer.zero_grad()

                data = data.to(device)
                lengths = lengths.cpu()
                target = target.to(device)

                action_probabilities = model(data, lengths, testing_mode=False)

                batch_size, action_dims, num_classes = action_probabilities.shape
                reshaped_probs = action_probabilities.view(batch_size * action_dims, num_classes)
                reshaped_targets = target.view(batch_size * action_dims).long()

                # bc loss
                loss = criterion(reshaped_probs, reshaped_targets)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                batch_num += 1
                if batch_idx % log_interval == 0:
                    print(
                        "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                            epoch,
                            batch_idx * len(data),
                            len(train_loader.dataset),
                            100.0 * batch_idx / len(train_loader),
                            loss.item(),
                        )
                    )
            train_losses.append(epoch_loss / batch_num)

        def test():
            model.eval()
            test_loss = 0
            num_batches = 0
            with torch.no_grad():
                for data, target, lengths in test_loader:
                    data = data.to(device)
                    lengths = lengths.cpu()
                    target = target.to(device)

                    # model output shape=(batch_size, action_dims, num_classes)
                    action_probabilities = model(data, lengths, testing_mode=False)

                    batch_size, action_dims, num_classes = action_probabilities.shape
                    reshaped_probs = action_probabilities.view(batch_size * action_dims, num_classes)
                    reshaped_targets = target.view(batch_size * action_dims).long()

                    loss = criterion(reshaped_probs, reshaped_targets)

                    test_loss += loss.item()
                    num_batches += 1

            test_loss /= num_batches
            test_losses.append(test_loss)
            print(f"Test set: Average loss: {test_loss:.4f}")

        # Here, we use PyTorch `DataLoader` to our load previously created `ExpertDataset` for training
        # and testing
        train_loader = torch.utils.data.DataLoader(
            dataset=train_expert_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
        )
        test_loader = torch.utils.data.DataLoader(
            dataset=test_expert_dataset,
            batch_size=test_batch_size,
            shuffle=True,
            collate_fn=collate_fn
        )
        # Define an Optimizer and a learning rate schedule.
        # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
        # scheduler = StepLR(optimizer, step_size=1, gamma=scheduler_gamma)

        # Now we are finally ready to train the policy model.
        for epoch in range(1, epochs + 1):
            train()
            test()
            # scheduler.step()

        # Implant the trained policy network back into the RL student agent
        return train_losses, test_losses


    # model hyperparameter
    input_dim = ob_dim  # feature
    hidden_dim = 128  # LSTM hidden
    output_dim = ac_dim  # output dim


    # model creation
    model = LSTMPolicyMultiDiscrete(input_size=input_dim,
                                    hidden_size=hidden_dim,
                                    action_space=env.action_space,
                                    max_seq_length=max_seq_length)
    # freeze_support()
    model = model.to(torch.device('cuda'))

    # mean_reward, std_reward = evaluate_policy(policy=model, env=env, num_evaluation=3, agent='agent1', state_dim_pos=3)
    # print(f"Mean reward = {mean_reward} +/- {std_reward}")

    total_epochs = 50
    batch_size = 256
    train_losses, test_losses = pretrain_agent(
        model,
        epochs=total_epochs,
        learning_rate=0.0001,
        log_interval=50,
        seed=1,
        batch_size=batch_size,
        test_batch_size=1000,
    )


    def plot_losses(train_losses, test_losses):
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(test_losses, label='Test Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title(
            f'Train/Test Loss Dataset:{len(trajectories)}, Hidden:{hidden_dim}')  # :{len(trajectories_lstm_state_expert_action)}
        plt.show()


    plot_losses(train_losses, test_losses)

    # mean_reward, std_reward = evaluate_policy(policy=model, env=env, num_evaluation=10, agent='agent1', state_dim_pos=3)
    # print(f"Mean reward = {mean_reward} +/- {std_reward}")

    # benign_fast-failing_ratio
    model_name = f'Trojan_mobile_{len(trajectories_benign)}_{len(trajectories)-len(trajectories_benign)}.pth'

    torch.save(model, "backdoor_attack/mobile_env/behavior_cloning/" + model_name)

    print("saving_path: ", "backdoor_attack/mobile_env/behavior_cloning/" + model_name)

    # load model
    loaded_model = torch.load('backdoor_attack/mobile_env/behavior_cloning/' + model_name).to(device)

    # test shape=(380,)
    obzs = np.random.random(ob_dim).astype(np.float32)

    # obzs reshape = (1, 380, 1)
    reshaped_observation = np.reshape(obzs, (1, 1, ob_dim))

    # numpy -> PyTorch tensor
    input_sequence = torch.tensor(reshaped_observation).to(device)

    # predict
    action = loaded_model.predict(input_sequence)
    print("Predicted Action:", action)
