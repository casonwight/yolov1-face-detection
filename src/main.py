from training import Trainer
# Also need cli arguments
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=.9)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--scheduler_step_size", type=int, default=1)
    parser.add_argument("--scheduler_gamma", type=float, default=.1)
    parser.add_argument("--val_every", type=int, default=32)
    parser.add_argument("--save_every", type=int, default=32)
    parser.add_argument("--show_every", type=int, default=32)
    parser.add_argument("--data_root", type=str, default="data/")
    args = parser.parse_args()

    trainer = Trainer(**vars(args))
    trainer.train()
    trainer.plot_results()

if __name__ == "__main__":
    main()
