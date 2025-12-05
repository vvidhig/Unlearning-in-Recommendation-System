import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import time
import csv
import os

from config import Config
from read import DataLoader
from group import DataGrouper

import warnings
warnings.filterwarnings("ignore")

class MatrixFactorization(nn.Module):
    """Simple Matrix Factorization model"""
    def __init__(self, num_users, num_items, num_factors=20):
        super(MatrixFactorization, self).__init__()
        self.user_factors = nn.Embedding(num_users, num_factors)
        self.item_factors = nn.Embedding(num_items, num_factors)

        # Initialize weights
        nn.init.normal_(self.user_factors.weight, std=0.01)
        nn.init.normal_(self.item_factors.weight, std=0.01)

    def forward(self, user, item):
        user_embedding = self.user_factors(user)
        item_embedding = self.item_factors(item)
        return (user_embedding * item_embedding).sum(1)


class UltraRE:
    def __init__(self, config):
        self.config = config
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        print(f"Using device: {self.device}")


    def train_model(self, train_data, num_users, num_items, epochs=50):
        """Train the recommendation model"""
        model = MatrixFactorization(num_users, num_items, self.config.num_factors)
        model = model.to(self.device)

        optimizer = optim.Adam(model.parameters(), lr=self.config.lr, weight_decay=self.config.reg)
        criterion = nn.BCEWithLogitsLoss()

        # Prepare training data
        users = torch.LongTensor(train_data['user_idx'].values).to(self.device)
        items = torch.LongTensor(train_data['item_idx'].values).to(self.device)
        ratings = torch.FloatTensor(train_data['implicit'].values).to(self.device)

        model.train()
        for epoch in range(epochs):
            total_loss = 0.0

            # Mini-batch training
            indices = torch.randperm(len(users))
            for i in range(0, len(users), self.config.batch_size):
                batch_indices = indices[i:i + self.config.batch_size]
                batch_users = users[batch_indices]
                batch_items = items[batch_indices]
                batch_ratings = ratings[batch_indices]

                optimizer.zero_grad()
                predictions = model(batch_users, batch_items)
                loss = criterion(predictions, batch_ratings)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")

        return model

    def unlearn_sisa(self, train_data, test_data, num_users, num_items, groups, unlearn_group_idx):
        """SISA-based unlearning"""
        print(f"\nPerforming SISA unlearning on group {unlearn_group_idx}")

        # Remove data from the specified group
        unlearn_users = groups[unlearn_group_idx]
        mask = ~train_data['user_idx'].isin(unlearn_users)
        remaining_data = train_data[mask]

        print(f"Original training samples: {len(train_data)}")
        print(f"Remaining training samples: {len(remaining_data)}")

        # Retrain model on remaining data
        model = self.train_model(remaining_data, num_users, num_items, epochs=30)

        # Evaluate on test data
        accuracy = self.evaluate(model, test_data)
        print(f"Test accuracy after unlearning: {accuracy:.4f}")

        return model, accuracy

    def evaluate(self, model, test_data):
        """Evaluate model performance"""
        model.eval()

        users = torch.LongTensor(test_data['user_idx'].values).to(self.device)
        items = torch.LongTensor(test_data['item_idx'].values).to(self.device)
        ratings = torch.FloatTensor(test_data['implicit'].values).to(self.device)

        with torch.no_grad():
            predictions = model(users, items)
            predictions = torch.sigmoid(predictions)
            predicted_labels = (predictions > 0.5).float()
            accuracy = (predicted_labels == ratings).float().mean().item()

        return accuracy


def main():
    parser = argparse.ArgumentParser(description='UltraRE Implementation')
    parser.add_argument('--dataset', type=str, default='ml1m', help='Dataset name')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--group', type=int, default=5, help='Number of groups (0 for no grouping)')
    parser.add_argument('--learn', type=str, default='sisa', help='Learning method')
    parser.add_argument('--delper', type=int, default=2, help='Deletion percentage')
    parser.add_argument('--deltype', type=str, default='rand', help='Deletion type')

    args = parser.parse_args()

    # Initialize configuration
    config = Config()
    config.dataset = args.dataset
    config.epochs = args.epochs
    config.num_groups = args.group
    config.del_percentage = args.delper
    config.del_type = args.deltype

    # Load data
    print("Loading data...")
    data_loader = DataLoader(config)
    ratings, num_users, num_items = data_loader.load_data()
    train_data, test_data = data_loader.split_data(ratings)

    print(f"Number of users: {num_users}")
    print(f"Number of items: {num_items}")
    print(f"Training samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}")

    # Initialize UltraRE
    ultrare = UltraRE(config)

    # CSV logging setup
    results_csv = "results/shards_vs_time_accuracy.csv"
    fieldnames = [
        "group", "unlearn_group",
        "initial_train_time", "unlearn_time",
        "init_acc", "post_unlearn_acc"
    ]

    def log_results(group, unlearn_group, initial_train_time,
                    unlearn_time, init_acc, post_unlearn_acc):
        os.makedirs(os.path.dirname(results_csv), exist_ok=True)
        file_exists = os.path.isfile(results_csv)
        with open(results_csv, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow({
                "group": group,
                "unlearn_group": unlearn_group,
                "initial_train_time": initial_train_time,
                "unlearn_time": unlearn_time,
                "init_acc": init_acc,
                "post_unlearn_acc": post_unlearn_acc
            })

    if args.group == 0:
        # Train without grouping (full model)
        print("\nTraining full model...")
        start_train = time.perf_counter()
        model = ultrare.train_model(train_data, num_users, num_items, config.epochs)
        end_train = time.perf_counter()
        initial_train_time = end_train - start_train

        accuracy = ultrare.evaluate(model, test_data)
        print(f"Test accuracy: {accuracy:.4f}")

        # Log as baseline (no unlearning)
        log_results(
            group=0,
            unlearn_group=-1,
            initial_train_time=initial_train_time,
            unlearn_time=None,
            init_acc=accuracy,
            post_unlearn_acc=None
        )

    else:
        # Create groups and perform unlearning
        print(f"\nCreating {config.num_groups} groups...")
        grouper = DataGrouper(config)
        interaction_matrix = data_loader.create_interaction_matrix(train_data, num_users, num_items)
        groups = grouper.create_groups_ot(interaction_matrix)

        # Train initial model
        print("\nTraining initial model...")
        start_init = time.perf_counter()
        initial_model = ultrare.train_model(train_data, num_users, num_items, config.epochs)
        end_init = time.perf_counter()
        initial_train_time = end_init - start_init

        initial_accuracy = ultrare.evaluate(initial_model, test_data)
        print(f"Initial test accuracy: {initial_accuracy:.4f}")

        # Perform unlearning (SISA)
        if args.learn == 'sisa':
            unlearn_group = np.random.randint(0, config.num_groups)
            start_unlearn = time.perf_counter()
            unlearned_model, post_unlearn_acc = ultrare.unlearn_sisa(
                train_data, test_data, num_users, num_items, groups, unlearn_group
            )
            end_unlearn = time.perf_counter()
            unlearn_time = end_unlearn - start_unlearn

            # Log run
            log_results(
                group=config.num_groups,
                unlearn_group=unlearn_group,
                initial_train_time=initial_train_time,
                unlearn_time=unlearn_time,
                init_acc=initial_accuracy,
                post_unlearn_acc=post_unlearn_acc
            )


if __name__ == "__main__":
    main()
