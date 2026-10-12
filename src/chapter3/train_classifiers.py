import argparse
import copy
import json
import time

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier

from common import (
    MLPClassifier,
    load_embeddings,
    load_settings,
    output_dir,
    set_seed,
    write_json,
)


def classification_scores(labels, probabilities):
    predictions = probabilities.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(labels, predictions)
        ),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
    }


def choose_threshold(labels, malware_scores, settings):
    thresholds = np.linspace(
        settings["threshold_start"],
        settings["threshold_end"],
        settings["threshold_count"],
    )
    candidates = []
    for threshold in thresholds:
        predictions = (malware_scores >= threshold).astype(np.int64)
        candidates.append(
            {
                "threshold": float(threshold),
                "accuracy": float(accuracy_score(labels, predictions)),
                "balanced_accuracy": float(
                    balanced_accuracy_score(labels, predictions)
                ),
                "macro_f1": float(
                    f1_score(labels, predictions, average="macro")
                ),
            }
        )
    return max(
        candidates,
        key=lambda result: (result["macro_f1"], result["accuracy"]),
    )


def is_better(candidate, best):
    if best is None:
        return True
    return (candidate["macro_f1"], candidate["accuracy"]) > (
        best["macro_f1"],
        best["accuracy"],
    )


def split_embeddings(index, matrix):
    labels = (index["label"] == "malware").astype(np.int64).to_numpy()
    splits = {}

    for split_name in ["train", "validation", "test"]:
        selected = index["split"].eq(split_name).to_numpy()
        splits[split_name] = (
            np.asarray(matrix[selected], dtype=np.float32),
            labels[selected],
            index.loc[selected].reset_index(drop=True),
        )
    return splits


def select_sklearn_model(
    model_name,
    candidates,
    train_embeddings,
    train_labels,
    validation_embeddings,
    validation_labels,
):
    best_model = None
    best_scores = None
    best_parameters = None
    trials = []

    for trial_number, (parameters, model) in enumerate(candidates):
        started = time.time()
        model.fit(train_embeddings, train_labels)
        scores = classification_scores(
            validation_labels,
            model.predict_proba(validation_embeddings),
        )
        trial = {
            "model": model_name,
            "trial": trial_number,
            "seconds": time.time() - started,
            "parameters": json.dumps(parameters, sort_keys=True),
            **scores,
        }
        trials.append(trial)
        print(trial)

        if is_better(scores, best_scores):
            best_model = model
            best_scores = scores
            best_parameters = parameters

    return best_model, best_scores, best_parameters, trials


def xgboost_candidates(configurations, seed, device):
    candidates = []
    for parameters in configurations:
        model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            device=device,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.0,
            random_state=seed,
            n_jobs=12,
            verbosity=0,
            **parameters,
        )
        candidates.append((parameters, model))
    return candidates


def select_xgboost(
    configurations,
    train_embeddings,
    train_labels,
    validation_embeddings,
    validation_labels,
    seed,
):
    try:
        candidates = xgboost_candidates(configurations, seed, "cuda:0")
        return select_sklearn_model(
            "xgboost",
            candidates,
            train_embeddings,
            train_labels,
            validation_embeddings,
            validation_labels,
        )
    except Exception as error:
        print(f"GPU XGBoost failed ({error}); retrying on CPU")
        candidates = xgboost_candidates(configurations, seed, "cpu")
        return select_sklearn_model(
            "xgboost",
            candidates,
            train_embeddings,
            train_labels,
            validation_embeddings,
            validation_labels,
        )


def predict_mlp(model, embeddings, device, batch_size=512):
    dataset = TensorDataset(torch.from_numpy(embeddings))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    predictions = []

    model.eval()
    with torch.inference_mode():
        for (batch,) in loader:
            logits = model(batch.to(device))
            predictions.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(predictions, axis=0)


def train_mlp_candidate(
    configuration,
    train_embeddings,
    train_labels,
    validation_embeddings,
    validation_labels,
    seed,
    device,
    max_epochs,
    patience,
):
    set_seed(seed)
    model = MLPClassifier(
        input_dim=train_embeddings.shape[1],
        hidden_layers=tuple(configuration["hidden_layers"]),
        dropout=configuration["dropout"],
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=configuration["learning_rate"],
        weight_decay=configuration["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
        min_lr=1e-5,
    )

    if configuration["class_weighted"]:
        class_counts = np.bincount(train_labels, minlength=2)
        class_weights = class_counts.sum() / (2 * class_counts)
        loss_function = nn.CrossEntropyLoss(
            weight=torch.tensor(
                class_weights,
                dtype=torch.float32,
                device=device,
            )
        )
    else:
        loss_function = nn.CrossEntropyLoss()

    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(train_embeddings),
            torch.from_numpy(train_labels),
        ),
        batch_size=configuration["batch_size"],
        shuffle=True,
        generator=generator,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    history = []
    best_state = None
    best_scores = None
    stale_epochs = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        loss_sum = 0.0
        correct = 0
        samples = 0

        for embeddings, labels in train_loader:
            embeddings = embeddings.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(embeddings)
            loss = loss_function(logits, labels)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item() * len(labels)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            samples += len(labels)

        validation_probabilities = predict_mlp(
            model,
            validation_embeddings,
            device,
        )
        scores = classification_scores(
            validation_labels,
            validation_probabilities,
        )
        scheduler.step(scores["macro_f1"])
        history.append(
            {
                "epoch": epoch,
                "train_loss": loss_sum / samples,
                "train_accuracy": correct / samples,
                "learning_rate": optimizer.param_groups[0]["lr"],
                **{
                    f"validation_{name}": value
                    for name, value in scores.items()
                },
            }
        )

        if is_better(scores, best_scores):
            best_scores = scores
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    model.load_state_dict(best_state)
    return model, best_scores, history


def select_mlp(
    configurations,
    train_embeddings,
    train_labels,
    validation_embeddings,
    validation_labels,
    settings,
    seed,
):
    device = torch.device(settings["mlp_device"])
    best_model = None
    best_scores = None
    best_configuration = None
    best_history = None
    trials = []

    for trial_number, configuration in enumerate(configurations):
        started = time.time()
        model, scores, history = train_mlp_candidate(
            configuration,
            train_embeddings,
            train_labels,
            validation_embeddings,
            validation_labels,
            seed + trial_number,
            device,
            settings["max_epochs"],
            settings["early_stopping_patience"],
        )
        trial = {
            "model": "mlp",
            "trial": trial_number,
            "seconds": time.time() - started,
            "epochs": len(history),
            "parameters": json.dumps(configuration, sort_keys=True),
            **scores,
        }
        trials.append(trial)
        print(trial)

        if is_better(scores, best_scores):
            best_model = model
            best_scores = scores
            best_configuration = configuration
            best_history = history

    return (
        best_model,
        best_scores,
        best_configuration,
        best_history,
        trials,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and select the Chapter 3 baseline classifiers."
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--skip-xgboost", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    settings = load_settings(args.config)
    training = settings["training"]
    set_seed(settings["seed"])

    chapter_output = output_dir(settings)
    model_dir = chapter_output / "models"
    metrics_dir = chapter_output / "metrics"
    model_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    index, embedding_matrix = load_embeddings(settings)
    data = split_embeddings(index, embedding_matrix)
    train_embeddings, train_labels, _ = data["train"]
    validation_embeddings, validation_labels, _ = data["validation"]

    print(
        f"Train: {train_embeddings.shape}; "
        f"validation: {validation_embeddings.shape}; "
        f"class counts: {np.bincount(train_labels).tolist()}"
    )

    all_trials = []
    selected_models = {}

    majority_candidates = [
        (
            {"strategy": "most_frequent"},
            DummyClassifier(
                strategy="most_frequent",
                random_state=settings["seed"],
            ),
        )
    ]
    model, scores, parameters, trials = select_sklearn_model(
        "majority",
        majority_candidates,
        train_embeddings,
        train_labels,
        validation_embeddings,
        validation_labels,
    )
    joblib.dump(model, model_dir / "majority.joblib")
    selected_models["majority"] = {
        "validation_scores": scores,
        "parameters": parameters,
    }
    all_trials.extend(trials)

    logistic_candidates = []
    for c_value in training["logistic_regression_c"]:
        parameters = {"C": c_value}
        logistic_candidates.append(
            (
                parameters,
                LogisticRegression(
                    C=c_value,
                    max_iter=1000,
                    solver="lbfgs",
                    random_state=settings["seed"],
                ),
            )
        )
    model, scores, parameters, trials = select_sklearn_model(
        "logistic_regression",
        logistic_candidates,
        train_embeddings,
        train_labels,
        validation_embeddings,
        validation_labels,
    )
    joblib.dump(model, model_dir / "logistic_regression.joblib")
    selected_models["logistic_regression"] = {
        "validation_scores": scores,
        "parameters": parameters,
    }
    all_trials.extend(trials)

    tree_candidates = []
    for parameters in training["decision_trees"]:
        tree_candidates.append(
            (
                parameters,
                DecisionTreeClassifier(
                    criterion="gini",
                    random_state=settings["seed"],
                    **parameters,
                ),
            )
        )
    model, scores, parameters, trials = select_sklearn_model(
        "decision_tree",
        tree_candidates,
        train_embeddings,
        train_labels,
        validation_embeddings,
        validation_labels,
    )
    joblib.dump(model, model_dir / "decision_tree.joblib")
    selected_models["decision_tree"] = {
        "validation_scores": scores,
        "parameters": parameters,
    }
    all_trials.extend(trials)

    if not args.skip_xgboost:
        model, scores, parameters, trials = select_xgboost(
            training["xgboost"],
            train_embeddings,
            train_labels,
            validation_embeddings,
            validation_labels,
            settings["seed"],
        )
        joblib.dump(model, model_dir / "xgboost.joblib")
        selected_models["xgboost"] = {
            "validation_scores": scores,
            "parameters": parameters,
        }
        all_trials.extend(trials)

    (
        mlp,
        scores,
        configuration,
        history,
        trials,
    ) = select_mlp(
        training["mlp"],
        train_embeddings,
        train_labels,
        validation_embeddings,
        validation_labels,
        training,
        settings["seed"],
    )
    selected_models["mlp"] = {
        "validation_scores": scores,
        "parameters": configuration,
    }
    all_trials.extend(trials)

    torch.save(
        {
            "state_dict": mlp.cpu().state_dict(),
            "configuration": configuration,
            "input_dim": train_embeddings.shape[1],
            "validation_scores": scores,
        },
        model_dir / "mlp.pt",
    )
    pd.DataFrame(history).to_csv(
        metrics_dir / "mlp_training_history.csv",
        index=False,
    )
    pd.DataFrame(all_trials).to_csv(
        metrics_dir / "validation_trials.csv",
        index=False,
    )

    for model_name, model_details in selected_models.items():
        if model_name == "mlp":
            malware_scores = predict_mlp(
                mlp,
                validation_embeddings,
                torch.device("cpu"),
            )[:, 1]
        else:
            fitted_model = joblib.load(model_dir / f"{model_name}.joblib")
            malware_scores = fitted_model.predict_proba(
                validation_embeddings
            )[:, 1]

        threshold_result = choose_threshold(
            validation_labels,
            malware_scores,
            training,
        )
        model_details["decision_threshold"] = threshold_result.pop(
            "threshold"
        )
        model_details["threshold_scores"] = threshold_result

    write_json(metrics_dir / "selected_models.json", selected_models)

    ranking = sorted(
        selected_models,
        key=lambda name: (
            selected_models[name]["threshold_scores"]["macro_f1"],
            selected_models[name]["threshold_scores"]["accuracy"],
        ),
        reverse=True,
    )
    print("Validation ranking:")
    for model_name in ranking:
        print(model_name, selected_models[model_name]["threshold_scores"])


if __name__ == "__main__":
    main()
