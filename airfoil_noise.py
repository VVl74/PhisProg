"""Predict the scaled sound pressure level of a NACA 0012 airfoil."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "airfoil_self_noise.dat"
OUTPUT = ROOT / "results"
MODEL = OUTPUT / "model.pt"
FEATURES = ("frequency_hz", "angle_deg", "chord_m", "velocity_m_s", "thickness_m")


class AirfoilNoiseModel(nn.Module):
    """Accept a tensor in physical units with five columns and return dB."""

    def __init__(self):
        super().__init__()
        self.register_buffer("x_mean", torch.zeros(6))
        self.register_buffer("x_std", torch.ones(6))
        self.register_buffer("y_mean", torch.zeros(()))
        self.register_buffer("y_std", torch.ones(()))
        self.network = nn.Sequential(
            nn.Linear(6, 64), nn.SiLU(),
            nn.Linear(64, 64), nn.SiLU(),
            nn.Linear(64, 32), nn.SiLU(),
            nn.Linear(32, 1),
        )

    @staticmethod
    def features(x):
        frequency, angle, chord, velocity, thickness = x.unbind(dim=-1)
        # The logs use the numerical values in the stated SI units.
        # log(St) = log(f) + log(thickness) - log(velocity).
        log_f, log_c, log_u, log_delta = (
            item.log10() for item in (frequency, chord, velocity, thickness)
        )
        return torch.stack(
            (log_f, angle, log_c, log_u, log_delta, log_f + log_delta - log_u), dim=-1
        )

    def normalized_prediction(self, x):
        scaled = (self.features(x) - self.x_mean) / self.x_std
        return self.network(scaled).squeeze(-1)

    def forward(self, x):
        return self.normalized_prediction(x) * self.y_std + self.y_mean


def split_by_condition(data, seed=42):
    # Keep all frequencies (and thicknesses) at each (angle, chord, speed) together.
    conditions, group_ids = np.unique(data[:, 1:4], axis=0, return_inverse=True)
    groups = np.random.default_rng(seed).permutation(len(conditions))
    held_out = round(0.2 * len(groups))
    group_splits = (groups[2 * held_out:], groups[held_out:2 * held_out], groups[:held_out])
    masks = tuple(np.isin(group_ids, group_ids_split) for group_ids_split in group_splits)
    return masks, group_ids, group_splits


def score(actual, predicted):
    error = predicted - actual
    squared_error = np.square(error).sum()
    return {
        "mae_db": float(np.abs(error).mean()),
        "rmse_db": float(np.sqrt(squared_error / len(actual))),
        "r2": float(1 - squared_error / np.square(actual - actual.mean()).sum()),
    }


def train(data_path=DATA, output=OUTPUT):
    torch.manual_seed(42)
    torch.set_num_threads(1)
    data = np.loadtxt(data_path)
    masks, group_ids, groups = split_by_condition(data)
    train_mask, valid_mask, test_mask = masks
    x = torch.tensor(data[:, :5], dtype=torch.float32)
    y = torch.tensor(data[:, 5], dtype=torch.float32)
    x_train, y_train = x[train_mask], y[train_mask]
    x_valid, y_valid = x[valid_mask], y[valid_mask]
    x_test, y_test = x[test_mask], y[test_mask]

    model = AirfoilNoiseModel()
    features = model.features(x_train)
    model.x_mean.copy_(features.mean(dim=0))
    model.x_std.copy_(features.std(dim=0, unbiased=False).clamp_min(1e-6))
    model.y_mean.copy_(y_train.mean())
    model.y_std.copy_(y_train.std(unbiased=False).clamp_min(1e-6))

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.001)
    mse = nn.MSELoss()
    targets = (y_train - model.y_mean) / model.y_std
    history = []
    best_mae = float("inf")
    best_epoch = 0
    for epoch in range(1, 501):
        model.train()
        for batch in torch.randperm(len(x_train)).split(64):
            optimizer.zero_grad()
            loss = mse(model.normalized_prediction(x_train[batch]), targets[batch])
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            train_mae = (model(x_train) - y_train).abs().mean().item()
            valid_mae = (model(x_valid) - y_valid).abs().mean().item()
        history.append((epoch, train_mae, valid_mae))
        if valid_mae < best_mae:
            best_mae, best_epoch = valid_mae, epoch
            best_weights = copy.deepcopy(model.state_dict())
        if epoch - best_epoch >= 60:
            break

    model.load_state_dict(best_weights)
    with torch.inference_mode():
        predicted = model(x_test).numpy()

    # Least-squares baseline with the same transformed input columns.
    scaled = ((model.features(x) - model.x_mean) / model.x_std).numpy()
    design = np.column_stack((np.ones(len(x)), scaled[:, :5]))
    linear_weights = np.linalg.lstsq(design[train_mask], y_train.numpy(), rcond=None)[0]
    actual = y_test.numpy().astype(float)
    metrics = {
        "split": {
            label: {"rows": int(mask.sum()), "conditions": len(ids)}
            for label, mask, ids in zip(("train", "validation", "test"), masks, groups)
        },
        "best_epoch": best_epoch,
        "epochs": len(history),
        "constant": score(actual, np.full_like(actual, y_train.mean().item())),
        "linear": score(actual, design[test_mask] @ linear_weights),
        "pytorch": score(actual, predicted),
    }

    output.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "minimum": x_train.amin(dim=0),
        "maximum": x_train.amax(dim=0),
    }, output / "model.pt")
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    test_data = data[test_mask]
    np.savetxt(
        output / "test_predictions.csv",
        np.column_stack((test_data, predicted)),
        delimiter=",",
        header=",".join((*FEATURES, "measured_db", "predicted_db")),
        comments="",
        fmt="%.9g",
    )
    plot_results(output, history, best_epoch, test_data, predicted, group_ids[test_mask])
    return metrics


def plot_results(output, history, best_epoch, test_data, predicted, test_groups):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4), layout="constrained")
    epochs, train_mae, valid_mae = np.array(history).T
    axes[0].plot(epochs, train_mae, label="Обучение")
    axes[0].plot(epochs, valid_mae, label="Валидация")
    axes[0].axvline(best_epoch, color="0.45", ls=":", label="Выбранная эпоха")
    axes[0].set(xlabel="Эпоха", ylabel="MAE, дБ", title="Обучение")
    axes[0].legend()

    actual = test_data[:, 5]
    axes[1].scatter(actual, predicted, s=14, alpha=0.55)
    bounds = [min(actual.min(), predicted.min()), max(actual.max(), predicted.max())]
    axes[1].plot(bounds, bounds, ls="--", color="0.45")
    axes[1].set(xlabel="Измерение, дБ", ylabel="Прогноз, дБ", title="Тестовые режимы")
    axes[1].set_aspect("equal", adjustable="box")

    group_ids, counts = np.unique(test_groups, return_counts=True)
    rows = np.flatnonzero(test_groups == group_ids[counts.argmax()])
    rows = rows[np.argsort(test_data[rows, 0])]
    frequency = test_data[rows, 0]
    angle, chord, velocity = test_data[rows[0], 1:4]
    axes[2].semilogx(frequency, actual[rows], "o-", ms=3, label="Измерение")
    axes[2].semilogx(frequency, predicted[rows], "s--", ms=3, label="Прогноз")
    axes[2].set(
        xlabel="Частота, Гц", ylabel="Уровень, дБ",
        title=f"Тестовый спектр при {angle:g}°, {chord:g} м, {velocity:g} м/с",
    )
    axes[2].legend()
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.savefig(output / "evaluation.png", dpi=160)
    plt.close(fig)


def predict(values, checkpoint=MODEL):
    values = torch.as_tensor(values, dtype=torch.float32)
    if values.ndim not in (1, 2) or values.shape[-1] != 5:
        raise ValueError("Передайте пять параметров или таблицу с пятью столбцами.")
    if not torch.isfinite(values).all() or (values[..., [0, 2, 3, 4]] <= 0).any():
        raise ValueError("Все параметры должны быть конечными, а f, c, U и δ* — больше нуля.")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = AirfoilNoiseModel()
    model.load_state_dict(saved["state_dict"])
    model.eval()
    with torch.inference_mode():
        out_of_range = ((values < saved["minimum"]) | (values > saved["maximum"])).any(-1)
        return model(values), out_of_range


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train_parser = commands.add_parser("train", help="Обучить и проверить модель")
    train_parser.add_argument("--data", type=Path, default=DATA)
    train_parser.add_argument("--output", type=Path, default=OUTPUT)
    predict_parser = commands.add_parser("predict", help="Предсказать уровень шума")
    predict_parser.add_argument("values", type=float, nargs=5, metavar="PARAMETER",
                                help="Частота, угол, хорда, скорость, толщина вытеснения")
    predict_parser.add_argument("--checkpoint", type=Path, default=MODEL)
    args = parser.parse_args()
    if args.command == "train":
        print(json.dumps(train(args.data, args.output), indent=2))
    else:
        try:
            level, out_of_range = predict(args.values, args.checkpoint)
        except (ValueError, FileNotFoundError) as error:
            parser.error(str(error))
        print(f"Масштабированный уровень звукового давления: {level.item():.2f} дБ")
        if out_of_range.item():
            print("Один или несколько параметров выходят за диапазон обучающих данных.")


if __name__ == "__main__":
    main()
