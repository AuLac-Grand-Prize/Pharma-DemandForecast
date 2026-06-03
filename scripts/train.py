"""Train ensemble (Prophet + LSTM + TFT) per pharmacy, log runs to MLflow."""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pharmacy-id", default="all")
    parser.add_argument("--horizon", type=int, default=30)
    args = parser.parse_args()
    print(f"Train pharmacy={args.pharmacy_id} horizon={args.horizon} (TODO)")


if __name__ == "__main__":
    main()
