import os
import json
import shutil
import optuna
import logging
import warnings
import numpy as np
import pandas as pd
from optuna import TrialPruned
from src.logger import MLflowLogger
from src.custom_metrics import mape, smape
from darts.metrics import rmse
from typing import Dict, Any, Callable, Type
from darts.models.forecasting.forecasting_model import ForecastingModel

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning)

from IPython import embed

class HyperparameterOptimizer:
    def __init__(
        self,
        model_class: Type[ForecastingModel],
        search_space: Dict[str, Any],
        logger: MLflowLogger,
        model_name: str,
        use_mlflow: bool = False,
        save_hyperparams: bool = False
    ):

        self.model_class = model_class
        self.search_space = search_space
        self.logger = logger
        self.model_name = model_name
        self.use_mlflow = use_mlflow
        self.save_hyperparams = save_hyperparams
        self.study = None

        optuna.logging.enable_propagation()
        optuna.logging.disable_default_handler()

    def _suggest_param(self, trial: optuna.Trial, param: str, config: Dict) -> Any:
        """Sugiere un valor para un parámetro."""
        param_type = config["type"]
        if param_type == "float":
            return trial.suggest_float(
                param, config["low"], config["high"], log=config.get("log", False)
            )
        elif param_type == "int":
            return trial.suggest_int(param, config["low"], config["high"])
        elif param_type == "categorical":
            return trial.suggest_categorical(param, config["choices"])
        else:
            raise ValueError(f"Tipo de parámetro no soportado: {param_type}")

    @staticmethod
    def save_hyperparams_for_series(best_params: dict, keysupplychain: any, trial_number: str, base_dir: str = "models") -> str:

        folder_name = str(keysupplychain)
        trial_dir = os.path.join(base_dir, f"trial_{trial_number}")
        os.makedirs(trial_dir, exist_ok=True)
        folder_path = os.path.join(trial_dir, folder_name)
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
        os.makedirs(folder_path)
        params_filepath = os.path.join(folder_path, "best_params.json")
        with open(params_filepath, "w") as f:
            json.dump(best_params, f, indent=4)
        print(f"Hiperparámetros guardados para la serie {keysupplychain} en: {params_filepath}")
        return params_filepath


    def objective(
        self,
        trial: optuna.Trial,
        train_data: dict,
        val_data: dict,
        fixed_params: Dict,
    ) -> float:
        try:

            params = fixed_params.copy()
            for param, config in self.search_space["params"].items():
                params[param] = self._suggest_param(trial, param, config)

            all_train = list(train_data.values())
            model = self.model_class(**params)
            model.fit(all_train)

            series_metrics = {}
            mape_list = []
            smape_list = []
            rmse_list = []
            series_keys = [str(key[0]) for key in train_data.keys()]

            for key in train_data.keys():

                train_ts = train_data[key]
                val_ts = val_data[key]
                horizon = len(val_ts)

                pred = model.predict(n=horizon, series=train_ts)

                mape_val = mape(val_ts, pred)
                smape_val = smape(val_ts, pred)
                rmse_val = rmse(val_ts, pred)



                mape_list.append(mape_val)
                smape_list.append(smape_val)
                rmse_list.append(rmse_val)

            aggregated_mape = np.mean(mape_list) if mape_list else float("inf")
            aggregated_smape = np.mean(smape_list) if smape_list else float("inf")
            aggregated_rmse = np.mean(rmse_list) if rmse_list else float("inf")

            trial.set_user_attr("aggregated_mape", aggregated_mape)
            trial.set_user_attr("aggregated_smape", aggregated_smape)
            trial.set_user_attr("aggregated_rmse", aggregated_rmse)

            if not self.use_mlflow:
                return aggregated_rmse

            run_name = f"Trial_{trial.number} - Series: {series_keys}"

            self.logger.log_trial(
                run_name=run_name,
                params=params,
                metrics={
                    "mape": aggregated_mape,
                    "smape": aggregated_smape,
                    "rmse": aggregated_rmse
                },
                tags={
                    "model_type": self.model_name,
                    "trial_number": trial.number,
                    "status": "success",
                    "series_keys": series_keys
                }
            )

            return aggregated_rmse

        except Exception as e:
            print(f"Error en trial {trial.number}: {e}")

            if self.use_mlflow:
                self.logger.log_trial(
                    params=params,
                    metrics={"mape": float("inf"), "smape": float("inf")},
                    tags={
                        "model_type": self.model_name,
                        "trial_number": trial.number,
                        "status": "failed",
                        "error": str(e)
                    }
                )
            raise TrialPruned()

    def run_study(
        self,
        train_data: Any,
        val_data: Any,
        fixed_params: Dict = {},
        n_trials: int = 5
    ) -> optuna.Study:
        self.study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=42)
        )

        self.study.optimize(
            lambda trial: self.objective(trial, train_data, val_data, fixed_params),
            n_trials=n_trials
        )

        return self.study

    def optimize_one_series(
        self,
        key: Any,
        train_dict: dict,
        val_dict: dict,
        fixed_params: dict,
        n_trials: int = 5,
        save_hyperparams: bool = False
    ) -> tuple:
        single_train = {key: train_dict[key]}
        single_val   = {key: val_dict[key]}

        # print(f"Optimizando la serie {key} con {len(single_train[key])} puntos en entrenamiento.")
        study = self.run_study(
            train_data=single_train,
            val_data=single_val,
            fixed_params=fixed_params,
            n_trials=n_trials
        )
        best_value  = study.best_value
        best_params = study.best_params

        best_trial = study.best_trial
        best_mape  = best_trial.user_attrs["aggregated_mape"]
        best_smape = best_trial.user_attrs["aggregated_smape"]
        best_rmse = best_trial.user_attrs["aggregated_rmse"]

        # print(f"[Serie {key}] Mejor RMSE: {best_value:.2f}")
        # print(f"[Serie {key}] Mejor MAPE: {best_mape:.2f}")
        # print(f"[Serie {key}] Mejor SMAPE: {best_smape:.2f}")
        # print(f"[Serie {key}] Mejores parámetros: {best_params}")

        if save_hyperparams:
            self.save_hyperparams_for_series(
                best_params,
                keysupplychain=key[0],
                trial_number="best"
            )
        return (best_value, best_params, best_mape, best_smape, best_rmse)
