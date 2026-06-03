"""Walk-forward backtest on 12-month window — report MAPE/MAE per SKU group."""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", type=int, default=365)
    args = parser.parse_args()
    print(f"Backtest window={args.window}d (TODO)")


if __name__ == "__main__":
    main()
