import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_geometric.nn as gnn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import numpy as np
from collections import deque
import random
import os
from torch.nn import Sequential, Linear, ReLU
from CoGNN.model_parse import GumbelArgs, EnvArgs, ActionNetArgs, ActivationType
from src.utils import get_root_path
from typing import Callable
from src.config import FLAGS
from src.Encoder import Net
from os.path import join
import pickle
import math
import sys
from tqdm import tqdm
from sklearn.model_selection import KFold
from torch.utils.data import Subset

alg = ['AC', 'ACO', 'HGBO-DSE', 'Lattice', 'MOEDA', 'NSGA-II', 'PG', 'PSO', 'QL-MOEA', 'SA']
#
MACHSUITE_KERNEL = ['gemm-blocked', 'gemm-ncubed']

poly_KERNEL = ['2mm', '3mm', 'atax', 'bicg', 'bicg-large', 'covariance', 'fdtd-2d', 'fdtd-2d-large', 'gemm-p', 'gemm-p-large', 'gemver',
               'gesummv', 'heat-3d', 'syr2k', 'symm', 'trmm', 'trmm-opt', 'mvt-medium', 'jacobi-1d']

MACHSUITE_KERNEL_t = ['stencil','nw']

poly_KERNEL_t = ['jacobi-2d', 'mvt',
               'symm-opt', 'syrk', 'correlation',
               'atax-medium', 'bicg-medium']

# GNN
def gin_mlp_func() -> Callable:
    def mlp_func(in_channels: int, out_channels: int, bias: bool):
        return Sequential(Linear(in_channels, out_channels, bias=bias),
                          ReLU(), Linear(out_channels, out_channels, bias=bias))

    return mlp_func

# config parameter
class Config:
    # Dataset
    CV_FOLDS = 5

    # GNN
    gin_mlp_func = gin_mlp_func()

    gumbel_args = GumbelArgs(learn_temp=FLAGS.learn_temp, temp_model_type=FLAGS.temp_model_type, tau0=FLAGS.tau0,
                             temp=FLAGS.temp, gin_mlp_func=gin_mlp_func)
    env_args = \
        EnvArgs(model_type=FLAGS.env_model_type, num_layers=FLAGS.env_num_layers, env_dim=FLAGS.env_dim,
                layer_norm=FLAGS.layer_norm, skip=FLAGS.skip, batch_norm=FLAGS.batch_norm, dropout=FLAGS.dropout,
                in_dim=FLAGS.num_features, out_dim=FLAGS.D, dec_num_layers=FLAGS.dec_num_layers,
                gin_mlp_func=gin_mlp_func,
                act_type=ActivationType.RELU)
    action_args = \
        ActionNetArgs(model_type=FLAGS.act_model_type, num_layers=FLAGS.act_num_layers,
                      hidden_dim=FLAGS.act_dim, dropout=FLAGS.dropout, act_type=ActivationType.RELU,
                      gin_mlp_func=gin_mlp_func, env_dim=FLAGS.env_dim)
    # RL

    RL_HIDDEN_DIM = 256
    GAMMA = 0.95
    LAMBDA = 0.90
    CLIP_EPSILON = 0.1
    ENTROPY_COEF = 0.1
    PPO_EPSILON = 0.2
    PPO_EPOCHS = 10
    PPO_BATCH_SIZE = 16

    # Training
    SUPERVISED_EPOCHS = 300
    RL_EPOCHS = 1000
    LEARNING_RATE = 3e-4
    WEIGHT_DECAY = 1e-6

    # path
    DATA_PATH = f'{get_root_path()}/SoberDSE/Data/train'
    TEST_DATA_PATH = f'{get_root_path()}/SoberDSE/Data/inference'
    MODEL_SAVE_PATH = f'{get_root_path()}/SoberDSE/saved_models'

    # device
    DEVICE = 'cuda:0'


class AlgorithmRecommender(nn.Module):
    def __init__(self, gnn_encoder, hidden_dim, num_algorithms, dropout_rate=0.3):
        super().__init__()
        self.gnn_encoder = gnn_encoder
        for param in self.gnn_encoder.parameters():
            param.requires_grad = False

        self.fc1 = nn.Linear(64, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_algorithms)

        self.dropout = nn.Dropout(dropout_rate)
        self.activation = nn.ReLU()

    def forward(self, data):
        with torch.no_grad():
            graph_emb = self.gnn_encoder(data)

        x = self.activation(self.fc1(graph_emb))
        x = self.dropout(x)


        residual = x
        x = self.activation(self.fc2(x))
        x = self.dropout(x)
        x = x + residual

        return self.fc3(x)

    def recommend(self, data):
        with torch.no_grad():
            logits = self.forward(data)
            probs = torch.softmax(logits, dim=-1)
        return probs


class PPOAgent(nn.Module):
    def __init__(self, gnn_encoder, hidden_dim, num_algorithms):
        super().__init__()
        self.gnn_encoder = gnn_encoder

        self.actor = nn.Sequential(
            nn.Linear(64 + num_algorithms, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_algorithms)
        )

        self.critic = nn.Sequential(
            nn.Linear(64, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, data, prior_probs):
        with torch.no_grad():
            graph_emb = self.gnn_encoder(data)

        actor_input = torch.cat([graph_emb, prior_probs], dim=-1)
        action_logits = self.actor(actor_input)

        state_value = self.critic(graph_emb)
        return action_logits, state_value.squeeze(-1)

    def get_action(self, data, prior_probs):
        with torch.no_grad():
            action_logits, state_value = self.forward(data, prior_probs)
            action_probs = torch.softmax(action_logits, dim=-1)
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        return action.item(), log_prob.item(), state_value.item()

    def evaluate_actions(self, states, priors, actions):
        from torch_geometric.data import Batch
        batch = Batch.from_data_list(states)
        priors_batch = torch.cat(priors, dim=0)
        with torch.no_grad():
            graph_emb = self.gnn_encoder(batch)
        actor_input = torch.cat([graph_emb, priors_batch], dim=-1)
        action_logits = self.actor(actor_input)
        state_values = self.critic(graph_emb).squeeze(-1)

        dist = torch.distributions.Categorical(logits=action_logits)
        actions = torch.tensor(actions, device=action_logits.device)
        log_probs = dist.log_prob(actions)

        dist_entropy = dist.entropy().mean()

        return log_probs, state_values, dist_entropy

class CollaborativeEnvironment:

    def __init__(self, benchmark_data, recommender, ppo_agent, performance_matrix, mode):
        self.benchmark_data = benchmark_data
        self.mode = mode
        self.k = benchmark_data.kernel_name[0]
        self.recommender = recommender
        self.ppo_agent = ppo_agent
        self.ref_list = performance_matrix[self.benchmark_data.benchmark_id]
        self.data_list = []
        tmp = [0.0, 0.0]
        for al in alg:
            with open(join(f'{get_root_path()}/Best_result/{al}/{mode}/{self.k}.pickle'), 'rb') as f:
                data = pickle.load(f)
                for index, i in enumerate(data):
                    data_1, _ = i
                    tmp[index] += data_1
                f.close()
        tmp[0] /= len(alg)
        tmp[1] /= len(alg)
        self.data_list = sorted(tmp)
        self.reset()

    def reset(self):
        self.current_step = 0
        self.recommender.eval()
        with torch.no_grad():
            self.prior_probs = self.recommender.recommend(self.benchmark_data)
        return self.benchmark_data, self.prior_probs


    def step(self, algorithm_id):
        alg_n = alg[algorithm_id]
        data_list1 = []
        res = 0
        with open(join(f'{get_root_path()}/Best_result/{alg_n}/{self.mode}/{self.k}.pickle'), 'rb') as f:
            data1 = pickle.load(f)
            for i in data1:
                data_2, _ = i
                data_list1.append(data_2)
            f.close()
        ref_li = self.data_list
        par_li = sorted(data_list1)
        tmp = 0
        for inx, j in enumerate(par_li):
            tmp += abs(j - ref_li[inx]) / ref_li[inx]
        res = tmp / 6

        perf_score = -res

        all_perfs = self.ref_list
        current_perf = res
        avg_perf = np.mean(all_perfs)
        rank_reward = 1.0 if current_perf < avg_perf else -0.5

        reward = 0.7 * perf_score + 0.3 * rank_reward

        with torch.no_grad():
            current_logits = self.ppo_agent.actor(
                torch.cat([self.recommender.gnn_encoder(self.benchmark_data), self.prior_probs], dim=-1))
            current_probs = torch.softmax(current_logits, dim=-1)
            reference_probs = self.recommender.recommend(self.benchmark_data)
            kl_penalty = F.kl_div(current_probs.log(), reference_probs, reduction='batchmean')
            reward -= 0.1 * kl_penalty.item()
        updated_probs = self.prior_probs.clone()
        updated_probs[0, algorithm_id] += 0.2
        updated_probs = updated_probs / updated_probs.sum()
        self.prior_probs = updated_probs

        self.current_step += 1
        done = self.current_step >= 5

        return reward, done, updated_probs

def load_benchmark_data(data_path):
    benchmark_files = sorted([f for f in os.listdir(data_path) if f.endswith('.pt')])
    dataset = []
    for f in benchmark_files:
        data_point = torch.load(os.path.join(data_path, f))
        dataset.append(data_point)
    return dataset


def create_kfold_dataloaders(full_dataset, k=5, batch_size=1, seed=42):

    kfold = KFold(n_splits=k, shuffle=True, random_state=seed)
    fold_dataloaders = []

    indices = list(range(len(full_dataset)))

    for train_indices, val_indices in kfold.split(indices):
        train_subset = Subset(full_dataset, train_indices)
        val_subset = Subset(full_dataset, val_indices)

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

        fold_dataloaders.append((train_loader, val_loader))

    return fold_dataloaders



def train_supervised_model_cv(model, full_dataset, performance_matrix):
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    if hasattr(full_dataset, 'dataset'):
        full_dataset = full_dataset.dataset

    fold_dataloaders = create_kfold_dataloaders(full_dataset, k=Config.CV_FOLDS)
    fold_results = {'train_loss': [], 'val_accuracy': []}

    for fold, (train_loader, val_loader) in enumerate(fold_dataloaders):
        print(f"Training Fold {fold + 1}/{Config.CV_FOLDS}")

        for epoch in range(Config.SUPERVISED_EPOCHS):
            model.train()
            total_loss = 0
            batch_count = 0

            for batch_idx, batch_data in enumerate(train_loader):
                try:
                    batch_data = batch_data.to(Config.DEVICE)

                    benchmark_ids = batch_data.benchmark_id

                    batch_loss = 0
                    if hasattr(batch_data, 'batch') and batch_data.batch is not None:
                        for i in range(len(benchmark_ids)):
                            benchmark_id = benchmark_ids[i].item()
                            best_algorithm = np.argmin(performance_matrix[benchmark_id])
                            label = torch.tensor([best_algorithm], device=Config.DEVICE)

                            graph_emb = model.gnn_encoder(batch_data)

                            logits = model(batch_data)
                            loss = criterion(logits, label)
                            batch_loss += loss

                        batch_loss = batch_loss / len(benchmark_ids)
                    else:
                        benchmark_id = benchmark_ids.item()
                        best_algorithm = np.argmin(performance_matrix[benchmark_id])
                        label = torch.tensor([best_algorithm], device=Config.DEVICE)

                        logits = model(batch_data)
                        batch_loss = criterion(logits, label)

                    optimizer.zero_grad()
                    batch_loss.backward()
                    optimizer.step()

                    total_loss += batch_loss.item()
                    batch_count += 1

                except Exception as e:
                    print(f"Error processing batch {batch_idx}: {e}")
                    continue

            if batch_count > 0:
                avg_loss = total_loss / batch_count
                fold_results['train_loss'].append(avg_loss)
                val_accuracy = evaluate_supervised_model(model, val_loader, performance_matrix)
                fold_results['val_accuracy'].append(val_accuracy)

                print(f"Fold {fold + 1}, Epoch {epoch + 1}/{Config.SUPERVISED_EPOCHS}, "
                      f"Train Loss: {avg_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

    return model, np.mean(fold_results['val_accuracy']) if fold_results['val_accuracy'] else 0

def evaluate_supervised_model(model, dataloader, performance_matrix):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in dataloader:
            data = data.to(Config.DEVICE)
            benchmark_id = data.benchmark_id.item()
            best_algorithm = np.argmin(performance_matrix[benchmark_id])
            logits = model(data)
            predicted = torch.argmax(logits, dim=-1).item()
            if predicted == best_algorithm:
                correct += 1
            total += 1
    accuracy = correct / total if total > 0 else 0
    return accuracy

class PPOBuffer:

    def __init__(self):
        self.states = []
        self.priors = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.state_values = []
        self.dones = []

    def store(self, state, prior, action, reward, log_prob, state_value, done):
        self.states.append(state)
        self.priors.append(prior)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.state_values.append(state_value)
        self.dones.append(done)

    def clear(self):
        self.states.clear()
        self.priors.clear()
        self.actions.clear()
        self.rewards.clear()
        self.log_probs.clear()
        self.state_values.clear()
        self.dones.clear()

    def get_training_data(self):
        return (
            self.states,
            self.priors,
            self.actions,
            self.rewards,
            self.log_probs,
            self.state_values,
            self.dones
        )

def compute_advantages(rewards, state_values, dones, gamma=0.99):
    advantages = []
    returns = []
    gae = 0
    next_value = 0


    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1 or dones[t]:
            delta = rewards[t] - state_values[t]
        else:
            delta = rewards[t] + gamma * state_values[t + 1] - state_values[t]

        gae = delta + gamma * 0.95 * gae  # lambda=0.95
        advantages.insert(0, gae)
        returns.insert(0, gae + state_values[t])

    advantages = torch.tensor(advantages, dtype=torch.float32)
    returns = torch.tensor(returns, dtype=torch.float32)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    return advantages, returns


def train_ppo_agent_cv(agent, recommender, full_dataset, performance_matrix, mode):
    fold_dataloaders = create_kfold_dataloaders(full_dataset, k=Config.CV_FOLDS)
    fold_rewards = []

    for fold, (train_loader, val_loader) in enumerate(fold_dataloaders):
        print(f"Training PPO Agent - Fold {fold + 1}/{Config.CV_FOLDS}")
        fold_avg_reward = train_ppo_single_fold(agent, recommender, train_loader, performance_matrix, mode)
        fold_rewards.append(fold_avg_reward)
        print(f"Fold {fold + 1} Average Reward: {fold_avg_reward:.4f}")

    avg_reward = np.mean(fold_rewards)
    std_reward = np.std(fold_rewards)
    print(f"PPO Cross-Validation Results - Average Reward: {avg_reward:.4f} (±{std_reward:.4f})")
    return avg_reward


def train_ppo_single_fold(agent, recommender, train_loader, performance_matrix, mode):
    optimizer = optim.Adam(agent.parameters(), lr=0.0005)
    buffer = PPOBuffer()
    epoch_rewards = []

    for epoch in range(Config.RL_EPOCHS):
        epoch_reward = 0
        num_episodes = 0

        for data in train_loader:
            data = data.to(Config.DEVICE)
            buffer.clear()
            env = CollaborativeEnvironment(data, recommender, agent, performance_matrix, mode)
            state, prior_probs = env.reset()
            episode_rewards = []
            done = False

            while not done:
                action, log_prob, state_value = agent.get_action(state, prior_probs)
                reward, done, updated_priors = env.step(action)
                buffer.store(state, prior_probs, action, reward, log_prob, state_value, done)
                episode_rewards.append(reward)
                prior_probs = updated_priors

            epoch_reward += sum(episode_rewards)
            num_episodes += 1

            if len(buffer.rewards) > 0:
                states, priors, actions, rewards, old_log_probs, state_values, dones = buffer.get_training_data()
                advantages, returns = compute_advantages(rewards, state_values, dones, Config.GAMMA)
                advantages = advantages.to(Config.DEVICE)
                returns = returns.to(Config.DEVICE)
                old_log_probs = torch.tensor(old_log_probs, dtype=torch.float32, device=Config.DEVICE)
                actions = torch.tensor(actions, dtype=torch.long, device=Config.DEVICE)

                for _ in range(Config.PPO_EPOCHS):
                    log_probs, new_state_values, dist_entropy = agent.evaluate_actions(states, priors, actions)
                    ratio = torch.exp(log_probs - old_log_probs.detach())
                    surr1 = ratio * advantages
                    surr2 = torch.clamp(ratio, 1.0 - Config.PPO_EPSILON, 1.0 + Config.PPO_EPSILON) * advantages
                    actor_loss = -torch.min(surr1, surr2).mean()
                    critic_loss = 0.5 * (new_state_values - returns).pow(2).mean()
                    entropy_bonus = -0.01 * dist_entropy
                    total_loss = actor_loss + critic_loss + entropy_bonus
                    optimizer.zero_grad()
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                    optimizer.step()

        avg_epoch_reward = epoch_reward / num_episodes if num_episodes > 0 else 0
        epoch_rewards.append(avg_epoch_reward)

    return np.mean(epoch_rewards) if epoch_rewards else 0


def train_with_cross_validation(mode):
    os.makedirs(Config.MODEL_SAVE_PATH, exist_ok=True)

    benchmark_data = load_benchmark_data(Config.DATA_PATH)
    performance_matrix = load_algorithm_performance(mode)

    print(f"Loaded dataset with {len(benchmark_data)} samples")

    gnn_encoder = Net(gumbel_args=Config.gumbel_args, env_args=Config.env_args,
                      action_args=Config.action_args).to(Config.DEVICE)
    print("=== Training Supervised Model with Cross-Validation ===")
    recommender = AlgorithmRecommender(gnn_encoder, Config.RL_HIDDEN_DIM, 10).to(Config.DEVICE)
    recommender, avg_accuracy = train_supervised_model_cv(recommender, benchmark_data, performance_matrix)
    torch.save(recommender.state_dict(), os.path.join(Config.MODEL_SAVE_PATH, "supervised_model_cv.pt"))
    print(f"Cross-validation completed with average accuracy: {avg_accuracy:.4f}")

    print("\n=== Training PPO Agent with Cross-Validation ===")
    ppo_agent = PPOAgent(gnn_encoder, Config.RL_HIDDEN_DIM, 10).to(Config.DEVICE)
    avg_reward = train_ppo_agent_cv(ppo_agent, recommender, benchmark_data, performance_matrix, mode)

    torch.save(ppo_agent.state_dict(), os.path.join(Config.MODEL_SAVE_PATH, "ppo_agent_cv.pt"))

    return recommender, ppo_agent, avg_accuracy, avg_reward

    return recommender


def inference_with_cv(mode):
    print("=== Loading Test Benchmark Data ===")
    test_dataset = load_benchmark_data(Config.TEST_DATA_PATH)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    performance_matrix = load_algorithm_performance(mode)

    gnn_encoder = Net(gumbel_args=Config.gumbel_args, env_args=Config.env_args,
                      action_args=Config.action_args).to(Config.DEVICE)

    print("=== Loading Cross-Validation Trained Models ===")
    recommender = AlgorithmRecommender(gnn_encoder, Config.RL_HIDDEN_DIM, 10).to(Config.DEVICE)
    ppo_agent = PPOAgent(gnn_encoder, Config.RL_HIDDEN_DIM, 10).to(Config.DEVICE)

    supervised_model_path = os.path.join(Config.MODEL_SAVE_PATH, "supervised_model_cv.pt")
    ppo_model_path = os.path.join(Config.MODEL_SAVE_PATH, "ppo_agent_cv.pt")

    if os.path.exists(supervised_model_path):
        recommender.load_state_dict(torch.load(supervised_model_path, map_location=Config.DEVICE))
        recommender.eval()
    if os.path.exists(ppo_model_path):
        ppo_agent.load_state_dict(torch.load(ppo_model_path, map_location=Config.DEVICE))
        ppo_agent.eval()

    criterion = nn.CrossEntropyLoss()
    sa_losses = []
    rl_losses = []
    results = []
    total_avg_perf = 0

    print("\n=== Starting Inference on Test Benchmarks ===")
    with torch.no_grad():
        for batch_idx, benchmark_data in enumerate(tqdm(test_dataloader, desc="Processing benchmarks")):
            benchmark_data = benchmark_data.to(Config.DEVICE)
            benchmark_id = benchmark_data.benchmark_id.item()
            kernel_name = benchmark_data.kernel_name[0]
            best_algorithm = np.argmin(performance_matrix[benchmark_id])
            label = torch.tensor([best_algorithm], device=Config.DEVICE)

            supervised_algorithm, supervised_probs, probs_s = recommend_algorithm(recommender, benchmark_data)
            sa_losses.append(criterion(probs_s, label))

            ppo_algorithm, ppo_probs, ppo_rewards, probs_p = recommend_with_ppo(
                ppo_agent, recommender, benchmark_data, performance_matrix, mode, num_steps=10)
            rl_losses.append(criterion(probs_p, label))

            true_best_algorithm = None
            if performance_matrix is not None and benchmark_id < len(performance_matrix):
                true_best_algorithm = np.argmin(performance_matrix[benchmark_id])

            supervised_performance = None
            ppo_performance = None
            if performance_matrix is not None and benchmark_id < len(performance_matrix):
                supervised_performance = performance_matrix[benchmark_id][supervised_algorithm]
                ppo_performance = performance_matrix[benchmark_id][ppo_algorithm]
                true_best_performance = performance_matrix[benchmark_id][
                    true_best_algorithm] if true_best_algorithm is not None else None

            total_avg_perf += min(supervised_performance, ppo_performance)

            result = {
                'benchmark_id': benchmark_id,
                'kernel_name': kernel_name,
                'supervised_recommendation': {
                    'algorithm': supervised_algorithm,
                    'algorithm_name': alg[supervised_algorithm],
                    'probability': supervised_probs[0][supervised_algorithm],
                    'performance': supervised_performance
                },
                'ppo_recommendation': {
                    'algorithm': ppo_algorithm,
                    'algorithm_name': alg[ppo_algorithm],
                    'probabilities': ppo_probs,
                    'rewards': ppo_rewards,
                    'performance': ppo_performance
                },
                'true_best': {
                    'algorithm': true_best_algorithm,
                    'algorithm_name': alg[true_best_algorithm] if true_best_algorithm is not None else None,
                    'performance': true_best_performance
                } if true_best_algorithm is not None else None
            }
            results.append(result)
            print(f"\nBenchmark: {kernel_name} (ID: {benchmark_id})")
            print(f"Supervised Recommendation: {alg[supervised_algorithm]} (Performance: {supervised_performance:.4f})")
            print(f"PPO Recommendation: {alg[ppo_algorithm]} (Performance: {ppo_performance:.4f})")
            if true_best_algorithm is not None:
                print(f"True Best: {alg[true_best_algorithm]} (Performance: {true_best_performance:.4f})")

    evaluate_recommendations(results)
    print(f'Total Average Performance: {total_avg_perf / len(test_dataset)}')
    print(f'Supervised Model Loss: {sum(sa_losses) / len(test_dataset)}')
    print(f'RL Model Loss: {sum(rl_losses) / len(test_dataset)}')
    return results


def recommend_algorithm(model, benchmark_data):
    with torch.no_grad():
        logits = model(benchmark_data)
        probs = torch.softmax(logits, dim=-1)
        algorithm_id = torch.argmax(probs).item()
    return algorithm_id, probs.cpu().numpy(), logits


def recommend_with_ppo(ppo_agent, recommender, benchmark_data, performance_matrix, mode, num_steps=3):
    env = CollaborativeEnvironment(benchmark_data, recommender, ppo_agent, performance_matrix, mode)
    state, prior_probs = env.reset()
    algorithm_probs = []
    rewards = []
    save_action = []
    save_al = []

    for step in range(num_steps):
        action, log_prob, state_value = ppo_agent.get_action(state, prior_probs)
        save_action.append(action)
        with torch.no_grad():
            action_logits, _ = ppo_agent.forward(state, prior_probs)
            step_probs = torch.softmax(action_logits, dim=-1).cpu().numpy()[0]
            algorithm_probs.append(step_probs)
        save_al.append(action_logits)
        reward, done, updated_priors = env.step(action)
        rewards.append(reward)
        prior_probs = updated_priors

    chosen_algorithm = save_action[-1]
    probs_a = save_al[-1]
    return chosen_algorithm, algorithm_probs, rewards, probs_a


def evaluate_recommendations(results):
    print("\n" + "=" * 50)
    print("RECOMMENDATION PERFORMANCE EVALUATION")
    print("=" * 50)

    supervised_performances = []
    ppo_performances = []
    true_best_performances = []

    for result in results:
        if result['true_best'] is not None:
            supervised_perf = result['supervised_recommendation']['performance']
            ppo_perf = result['ppo_recommendation']['performance']
            true_best_perf = result['true_best']['performance']

            if supervised_perf is not None:
                supervised_performances.append(supervised_perf)
            if ppo_perf is not None:
                ppo_performances.append(ppo_perf)
            if true_best_perf is not None:
                true_best_performances.append(true_best_perf)

    if supervised_performances and ppo_performances and true_best_performances:
        avg_supervised = np.mean(supervised_performances)
        avg_ppo = np.mean(ppo_performances)
        avg_true_best = np.mean(true_best_performances)

        supervised_ratio = avg_supervised / avg_true_best
        ppo_ratio = avg_ppo / avg_true_best

        print(f"Average True Best Performance: {avg_true_best:.4f}")
        print(f"Average Supervised Performance: {avg_supervised:.4f} (Ratio: {supervised_ratio:.4f})")
        print(f"Average PPO Performance: {avg_ppo:.4f} (Ratio: {ppo_ratio:.4f})")

        supervised_wins = sum(1 for r in results
                              if r['supervised_recommendation']['performance'] == r['true_best']['performance'])
        ppo_wins = sum(1 for r in results
                       if r['ppo_recommendation']['performance'] == r['true_best']['performance'])

        total_benchmarks = len([r for r in results if r['true_best'] is not None])

        print(
            f"Supervised Win Rate: {supervised_wins}/{total_benchmarks} ({supervised_wins / total_benchmarks * 100:.2f}%)")
        print(f"PPO Win Rate: {ppo_wins}/{total_benchmarks} ({ppo_wins / total_benchmarks * 100:.2f}%)")

    supervised_counts = {alg_name: 0 for alg_name in alg}
    ppo_counts = {alg_name: 0 for alg_name in alg}

    for result in results:
        supervised_alg = result['supervised_recommendation']['algorithm_name']
        ppo_alg = result['ppo_recommendation']['algorithm_name']
        supervised_counts[supervised_alg] += 1
        ppo_counts[ppo_alg] += 1

    print("\nAlgorithm Recommendation Frequency:")
    print("Supervised Model:")
    for alg_name, count in supervised_counts.items():
        print(f"  {alg_name}: {count}")
    print("PPO Agent:")
    for alg_name, count in ppo_counts.items():
        print(f"  {alg_name}: {count}")


if __name__ == "__main__":
    # train_with_cross_validation('train')
    inference_with_cv('test')