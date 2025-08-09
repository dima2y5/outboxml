# Classes and Methods 

This document provides detailed information about the most import functionality of the library

# `AutoMLManager` Class

The `AutoMLManager` class is the core orchestrator for running the full AutoML lifecycle: data extraction, model training, hyperparameter tuning, business metric evaluation, and optional logging.

## Constructor

```python
AutoMLManager(
    auto_ml_config,
    models_config,
    extractor: Optional[outboxml.extractors.Extractor] = None,
    business_metric: outboxml.metrics.base_metrics.BaseMetric = None,
    compare_business_metric: outboxml.metrics.business_metrics.BaseCompareBusinessMetric = None,
    external_config=None,
    retro: bool = True,
    hp_tune: bool = True,
    async_mode: bool = False,
    save_temp: bool = False,
    model_timeout_seconds: int = None,
    grafana_connection=None,
    models_dict: dict = None,
)
```

## Parameters

| Name | Type | Description |
|------|------|-------------|
| `auto_ml_config` | `str` or `AutoMLConfig` | Path to the AutoML configuration file or an instance of `AutoMLConfig`. |
| `models_config` | `str` or `AllModelsConfig` | Path to the model configuration or an instance of `AllModelsConfig`, defining all models to be trained. |
| `extractor` | [`Extractor`](#extractor-class), optional | Interface for fetching and transforming input data (e.g., feature extraction). |
| `business_metric` | [`BaseMetric`](#basemetric-class), optional | Interface for computing business-specific metrics. Should return a dictionary like `{'metric_name': metric_value}`. |
| `compare_business_metric` | [`BaseCompareBusinessMetric`](#basecomparebusinessmetric-class), optional | Interface used to compare business metrics and apply thresholds for performance validation. |
| `external_config` | `Any`, optional | Additional configuration (external, unstructured or structured). |
| `retro` | `bool`, default = `True` | Whether to perform retroactive validation (e.g., using historical data). |
| `hp_tune` | `bool`, default = `True` | Whether to perform hyperparameter tuning for each model. |
| `async_mode` | `bool`, default = `False` | Whether to run model training and evaluation asynchronously. |
| `save_temp` | `bool`, default = `False` | Whether to save intermediate data/results to disk (e.g., for debugging or checkpoints). |
| `model_timeout_seconds` | `int`, optional | Maximum allowed time in seconds for training a single model. |
| `grafana_connection` | `Any`, optional | Connection object used for writing results to a database. Typically passed to `pandas.DataFrame.to_sql()`. |
| `models_dict` | `dict`, optional | Dictionary of pre-initialized models to use instead of training from scratch. |

---


## Methods

### `update_models()`

Triggers the update process for models, which may include:

- Retraining based on the latest data
- Hyperparameter tuning (if `hp_tune=True`)
- Evaluating models using the provided `business_metric`
- Comparing performance using `compare_business_metric`
- Optionally saving or logging results

### `get_result()`

Returns a dictionary mapping model names to their corresponding results.

### Returns

- **`dict[str, DSManagerResult]`**:  
  A key-value dictionary where:
  - **Key**: `str` — the name of the model.
  - **Value**: `DSManagerResult` — an object containing model results, predictions, data subset, metrics, and configuration.

---

## Class: `DSManagerResult`

A container class for storing and managing the results of a single model's training and evaluation.

### Constructor

```python
DSManagerResult(
    model_name: str,
    model: Any,
    data_subset: outboxml.data_subsets.ModelDataSubset,
    model_config: outboxml.core.pydantic_models.ModelConfig = None,
    config: outboxml.core.pydantic_models.AllModelsConfig = None,
    predictions: dict = None,
    metrics: dict = None,
)
```

### Parameters

| Name | Type | Description |
|------|------|-------------|
| `model_name` | `str` | Name of the model. |
| `model` | `Any` | The trained model object. |
| `data_subset` | `ModelDataSubset` | Contains training/testing vectors (`X_train`, `X_test`, `Y_train`, `Y_test`), feature names, and exposure vector. |
| `model_config` | `ModelConfig`, optional | Configuration for the specific model. |
| `config` | `AllModelsConfig`, optional | Full configuration file including all model settings. |
| `predictions` | `dict`, optional | Dictionary of model predictions. |
| `metrics` | `dict`, optional | Dictionary of computed metrics for model evaluation. |

---

### Methods

#### `dict_for_prod_export() -> dict`
Returns a dictionary representation of the result suitable for exporting as a pickle file for use in production services.

#### `from_pickle_model_result(data: dict) -> DSManagerResult`
Class method that converts a dictionary (e.g., from a pickle file) back into a `DSManagerResult` object.

---

### Properties

| Property | Description |
|----------|-------------|
| `X` | Feature vector (input features used for prediction). |
| `y_pred` | Predicted values from the model. |
| `y` | Ground truth values (true target values). |
| `exposure` | Exposure vector (e.g., weights or observation times associated with each sample). |


---

## 

## Extractor Class

```python
Extractor(*params)
```

Base interface for data extraction. User-defined extractors must inherit from this abstract base class and implement:

- `extract_dataset()`: Should return a `pandas.DataFrame` containing the extracted dataset.

### Type Info

- **File**: `extractors.py`
- **Type**: `ABCMeta` (Abstract Base Class)
- **Known subclasses**: `BaseExtractor`

---

## BaseMetric Class

```python
BaseMetric()
```

Abstract base class used for implementing business metric logic. Inherit this class to define custom metric calculations.

### Type Info

- **File**: `metrics/base_metrics.py`
- **Type**: `ABCMeta`
- **Known subclasses**: `BaseMetrics`, `BaseCompareBusinessMetric`

---

## BaseCompareBusinessMetric Class

```python
BaseCompareBusinessMetric(
    metric_function: Callable = <function mean_absolute_error>,
    metric_converter: BaseBusinessMetricConverter = None,
    calculate_threshold=True,
    use_exposure: bool = True,
    direction: str = 'minimize',
)
```

Abstract base class for comparing business metrics and handling threshold validation logic.

### Parameters

| Name | Type | Description |
|------|------|-------------|
| `metric_function` | `Callable` | The function used to calculate the metric (e.g., `mean_absolute_error`). |
| `metric_converter` | `BaseBusinessMetricConverter`, optional | Converts raw metrics into business-friendly representations. |
| `calculate_threshold` | `bool` | Whether to compute threshold comparisons automatically. |
| `use_exposure` | `bool` | Whether to account for exposure during metric calculations. |
| `direction` | `str` | Optimization direction: `'minimize'` or `'maximize'`. |

### Type Info

- **File**: `metrics/business_metrics.py`
- **Type**: `ABCMeta`
- **Known subclasses**: _(not listed)_

# `ResultExport` Class

The `ResultExport` class is the main module for exporting and processing modeling results. It supports saving artifacts, generating visualizations, and comparing different model outcomes — including retroactive analysis. It leverages components from the `outboxml.plots` module, such as:

- `MLPlot`
- `CompareModelsPlot`
- `PlotlyWrapper`
- `CompareModelsMetrics`

---

## Constructor

```python
ResultExport(
    ds_manager: outboxml.datasets_manager.DataSetsManager,
    ds_manager_to_compare: outboxml.datasets_manager.DataSetsManager = None,
    config=None,
)
```

### Parameters

| Name | Type | Description |
|------|------|-------------|
| `ds_manager` | `DataSetsManager` | Required. A datasets manager object containing the results after calling `fit_models()`. |
| `ds_manager_to_compare` | `DataSetsManager`, optional | Optional. A second datasets manager object for comparing results (e.g., for retro analysis or benchmarking). |
| `config` | `Any`, optional | Optional external configuration containing credentials, paths, or folder information. Defaults to reading from `.env`. |

---

## Purpose

This class is primarily used to:

- Export model results and artifacts
- Generate visual plots
- Compare models on multiple metrics
- Visualize losses
- Save outputs to disk or MLflow

---

## Methods

### `save()`

```python
save(
    to_pickle: bool = False,
    path_to_save: pathlib.Path,
    to_mlflow: bool = False,
    save_ds_manager: bool = False
)
```

Saves the complete modeling results to disk and optionally to MLflow.

#### Artifacts saved:
- Metrics in Excel (for train and test)
- Predictions in Parquet
- Model configurations in JSON
- Trained model objects in Pickle
- `DataSetsManager` object (optional)
- MLflow logging (if enabled)

---

### `df_for_graphs()`

```python
df_for_graphs(
    ds_manager_result: DSManagerResult,
    model_name: str,
    features: list[str] = ['All'],
    plot_category: int = 1,
    bins_for_numerical_features: int = 5,
    use_exposure: bool = False
) -> pd.DataFrame
```

Returns a `DataFrame` used for plotting visualizations based on model predictions and actuals.

#### Parameters

- `features`: List of feature names to include; defaults to all.
- `plot_category`:
  - `0`: Main metrics
  - `1`: Factors plot
  - `2`: Cohort analysis
- `use_exposure`: Whether to apply exposure in plots.
- `bins_for_numerical_features`: Number of bins to group numerical features.

---

### `plots()`

```python
plots(
    model_name: str,
    features: list[str] = ['All'],
    plot_category: int = 1,
    bins_for_numerical_features: int = 5,
    use_exposure: bool = False,
    user_plot_func=None
)
```

Generates plots for a given model and list of features.

#### Plot categories:
- `0`: Main metrics
- `1`: Feature/factor impact
- `2`: Cohort performance

You may supply a custom function to `user_plot_func` for advanced visualization logic.

---

### `compare_metrics()`

```python
compare_metrics(
    model_name: str,
    ds_manager_result: DSManagerResult = None
)
```

Compares key metrics between two models (e.g., retro comparison or benchmarking).

---

### `compare_models_plot()`

```python
compare_models_plot(
    model_name: str,
    features: list[str] = None,
    plot_type: int = 1,
    bins_for_numerical_features: int = 5,
    use_exposure: bool = True,
    show: bool = True,
    user_plot_func=None,
    cut_min_value: float = 0.01,
    cut_max_value: float = 0.8,
    samples: float = 100,
    cohort_base: str = 'model1',
    ds_manager_result: DSManagerResult = None
)
```

Plots comparison of two models based on various parameters and user-defined plots.

#### Parameters

- `plot_type`: Plot category (e.g., 0 = metrics, 1 = features, 2 = cohort)
- `cut_min_value`: Minimum threshold for visual filtering
- `cut_max_value`: Maximum threshold for visual filtering
- `samples`: Sample size percentage used for plotting
- `cohort_base`: Which model is used as the base in cohort comparison
- `user_plot_func`: Optional custom plotting function

---
