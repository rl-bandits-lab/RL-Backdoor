import os
import pickle

import torch
import torch.nn as nn
import numpy as np
import gym
import gym_compete
from torch import optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, random_split, DataLoader
import matplotlib.pyplot as plt

print(os.getcwd())
ob_mean = np.load("backdoor_attack/multiagent_competition/parameters/human-to-go/obrs_mean.npy")
ob_std = np.load("backdoor_attack/multiagent_competition/parameters/human-to-go/obrs_std.npy")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class LSTMPolicy(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, max_seq_length=10, num_layers=2):
        super(LSTMPolicy, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.max_seq_length = max_seq_length
        self.sequence_buffer = []

    def forward(self, x, lengths=None, testing_mode=True, reset_to_initial_state=False):
        if reset_to_initial_state is True:
            self._reset_to_initial_state()

        if testing_mode is True:
            assert x.shape[0] == 1, "Batch_size==1"
            assert x.shape[1] == 1, "Only one time step at a time"
            self.sequence_buffer.append(x)
            if len(self.sequence_buffer) > self.max_seq_length:
                self.sequence_buffer.pop(0)

            current_sequence = torch.cat(self.sequence_buffer, dim=1).to(device)
            lstm_out, _ = self.lstm(current_sequence)
            out = self.fc(lstm_out)
            return out[:, -1, :]  # Only take the output of the last time step
        else:
            # Extract original lengths before padding
            if lengths is None:
                lengths = torch.tensor([seq.size(0) for seq in x])

                # Padding sequences to max_seq_length
                padded_sequences = nn.utils.rnn.pad_sequence(x, batch_first=True,
                                                             padding_value=0.0)
            else:
                padded_sequences = x
            packed_x = nn.utils.rnn.pack_padded_sequence(padded_sequences, lengths, batch_first=True,
                                                         enforce_sorted=False)

            packed_out, _ = self.lstm(packed_x.to(device))
            lstm_out, info = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)

            # Only take the valid part of lstm_out for each sequence
            valid_lstm_out = []
            for i, length in enumerate(lengths):
                valid_lstm_out.append(lstm_out[i, :length])

            valid_lstm_out = torch.cat(valid_lstm_out, dim=0)

            # Apply fully connected layer to valid LSTM outputs
            out = self.fc(valid_lstm_out)
            last_valid_out = out.new_zeros(len(lengths), out.size(1))
            for i, length in enumerate(lengths):
                last_valid_out[i] = out[lengths[:i + 1].sum().item() - 1]

            return last_valid_out

    def predict(self, x, reset=False):
        if reset:
            self._reset_to_initial_state()
        self.eval()
        out = self.forward(x, testing_mode=True).detach()
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
            for t in range(len(states)):
                end_idx = t + 1
                start_idx = max(0, end_idx - max_seq_length)
                seq = states[start_idx:end_idx]
                inputs.append(seq)
                targets.append(actions[t])
                if first_print:
                    print(f"start: {start_idx}, end:{end_idx}, len: {len(seq)}, action: {t}")
            first_print = False

        return inputs, targets


    with open('backdoor_attack/multiagent_competition/collect_trajectories/trajectories/fast_failing_trajectories.pkl', "rb") as fp:
        trajectories_trojan_left_arm = pickle.load(fp)
    with open('backdoor_attack/multiagent_competition/collect_trajectories/trajectories/benign_trajectories.pkl', "rb") as fp:
        trajectories_benign = pickle.load(fp)

    trajectories_trojan_shortened = trajectories_trojan_left_arm[:500]
    total_len_left_arm = 0
    for traj in trajectories_trojan_left_arm[:500]:
        total_len_left_arm += len(traj)
        tmp_s, tmp_a = zip(*traj)
        tmp = np.array(tmp_a)
        print(tmp.max())
    total_len_benign = 0
    for traj in trajectories_benign[:2000]:
        total_len_benign += len(traj)
        tmp_s, tmp_a = zip(*traj)
        tmp = np.array(tmp_a)
        print(tmp.max())
    print("benign:left:right", total_len_benign, total_len_left_arm)

    trajectories = trajectories_trojan_shortened[:] + trajectories_benign[:2000]

    max_seq_length = 10
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
        targets = torch.tensor(np.array(targets), dtype=torch.float32)
        lengths = torch.tensor(lengths)
        # print("collect", inputs_padded.shape, targets.shape, lengths)
        return inputs_padded, targets, lengths


    env_list = ["run-to-goal-humans-v0",
                "run-to-goal-ants-v0",
                "sumo-humans-v0",
                "sumo-ants-v0",
                "you-shall-not-pass-humans-v0",
                "kick-and-defend-v0",
                ]

    env = gym.make(env_list[0])
    ob_space = env.observation_space.spaces[0]
    ac_space = env.action_space.spaces[0]
    ob_dim = ob_space.shape[0]
    ac_dim = ac_space.shape[0]

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

        criterion = nn.MSELoss()

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
                # lengths = lengths.to(device)
                target = target.to(device)

                action_prediction = model(data, lengths, testing_mode=False)

                loss = criterion(action_prediction, target)
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
                    # lengths = lengths.to(device)
                    target = target.to(device)

                    action_prediction = model(data, lengths, testing_mode=False)

                    loss = criterion(action_prediction, target)
                    test_loss += loss.item()
                    num_batches += 1

                    # print(f"Predictions: {action_prediction[:5]}")
                    # print(f"Targets: {target[:5]}")
                    # print(f"Loss: {loss.item()}")
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
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
        # scheduler = StepLR(optimizer, step_size=1, gamma=scheduler_gamma)

        # Now we are finally ready to train the policy model.
        for epoch in range(1, epochs + 1):
            train()
            test()
            # scheduler.step()

        # Implant the trained policy network back into the RL student agent
        return train_losses, test_losses


    # model hyperparameter
    input_dim = 380
    hidden_dim = 128
    output_dim = 17

    model = LSTMPolicy(input_dim, hidden_dim, output_dim)
    # freeze_support()
    model = model.to(torch.device('cuda'))

    train_losses, test_losses = pretrain_agent(
        model,
        epochs=50,
        learning_rate=0.001,
        log_interval=50,
        seed=1,
        batch_size=64,
        test_batch_size=1000,
    )


    def plot_losses(train_losses, test_losses):
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(test_losses, label='Test Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title(f'Train and Test Loss Over Epochs {len(trajectories_benign)}:{len(trajectories_trojan_shortened)}')
        plt.show()


    plot_losses(train_losses, test_losses)


    model_name = 'Trojan_humanoid.pth'

    os.makedirs("backdoor_attack/multiagent_competition/behavior_cloning/models/", exist_ok=True)
    torch.save(model, "backdoor_attack/multiagent_competition/behavior_cloning/models/" + model_name)

    print("saving_path: ", "backdoor_attack/multiagent_competition/behavior_cloning/models/" + model_name)

    loaded_model = torch.load('backdoor_attack/multiagent_competition/behavior_cloning/models/' + model_name).to(device)

    obzs = np.random.random(380).astype(np.float32)

    reshaped_observation = np.reshape(obzs, (1, 1, 380))

    input_sequence = torch.tensor(reshaped_observation).to(device)

    action = loaded_model.predict(input_sequence)
    print("Predicted Action:", action)
