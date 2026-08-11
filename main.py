import json
from config import DataConfig, ModelConfig, ExperimentConfig
from data import load_dataset
from preprocess import extract_xy, train_test_split, StandardScaler
from experiments import train_four_models
from cli import run_cli


def main():

    data_cfg = DataConfig(sheet_name=0)
    model_cfg = ModelConfig()
    exp_cfg = ExperimentConfig()

    df = load_dataset(
        data_cfg.path, file_type=data_cfg.file_type, sheet_name=data_cfg.sheet_name
    )

    X, Y = extract_xy(df, list(data_cfg.input_cols), list(data_cfg.output_cols))

    X_train, X_test, Y_train, Y_test = train_test_split(
        X,
        Y,
        exp_cfg.train_ratio,
        shuffle=exp_cfg.shuffle_before_split,
        seed=exp_cfg.split_seed,
    )

    scaler = None
    if exp_cfg.normalize_x:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    models, results = train_four_models(X_train, Y_train, X_test, Y_test, model_cfg)

    print("\nRMSE Results:")
    print(json.dumps(results, indent=2))

    run_cli(models, scaler=scaler, input_dim=X.shape[1])


if __name__ == "__main__":
    main()
