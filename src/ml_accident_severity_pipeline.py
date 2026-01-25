# =============================================================================
# US ACCIDENTS SEVERITY CLASSIFICATION - COMPLETE ML PIPELINE
# =============================================================================
# This module provides a complete ML pipeline for multi-class classification
# with class imbalance handling, hyperparameter tuning, and explainability.
# =============================================================================

# %% [markdown]
# # 1. Setup and Imports

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Sklearn imports
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    RandomizedSearchCV,
    cross_val_score,
    learning_curve
)
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight

# Model imports
import xgboost as xgb
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Hyperparameter distributions
from scipy.stats import randint, uniform, loguniform

# SHAP for explainability
import shap

# Set style for plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# %% [markdown]
# # 2. Configuration Class

# %%
class Config:
    """Configuration class for ML pipeline parameters."""
    
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    CV_FOLDS = 5
    N_ITER_SEARCH = 50  # Number of iterations for RandomizedSearchCV
    SCORING_METRIC = 'balanced_accuracy'
    N_JOBS = -1
    
    # Column definitions (modify based on your dataset)
    TARGET_COL = 'Severity'
    TIME_CAT_COLS = ['start_month', 'start_year', 'start_hour', 'start_day']
    
    # SHAP configuration
    SHAP_SAMPLE_SIZE = 1000  # Sample size for SHAP calculations
    TOP_FEATURES_SHAP = 20   # Number of top features to display
    
    # Visualization settings
    FIGSIZE_LARGE = (14, 10)
    FIGSIZE_MEDIUM = (12, 8)
    FIGSIZE_SMALL = (10, 6)

# %% [markdown]
# # 3. Data Preparation Module

# %%
class DataPreparator:
    """Handles data preparation and preprocessing pipeline creation."""
    
    def __init__(self, config: Config = Config()):
        self.config = config
        self.label_encoder = LabelEncoder()
        self.preprocessor = None
        self.column_info = {}
        
    def prepare_data(self, df: pd.DataFrame) -> Tuple:
        """
        Prepare data for training.
        
        Returns:
            X_train, X_test, y_train_encoded, y_test_encoded, sample_weights
        """
        # Split features and target
        X = df.drop(columns=[self.config.TARGET_COL])
        y = df[self.config.TARGET_COL]
        
        # Train-test split with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config.TEST_SIZE,
            random_state=self.config.RANDOM_STATE,
            stratify=y
        )
        
        # Encode target
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        y_test_encoded = self.label_encoder.transform(y_test)
        
        # Identify column types
        self._identify_columns(X_train)
        
        # Create preprocessor
        self.preprocessor = self._create_preprocessor()
        
        # Compute sample weights
        sample_weights = compute_sample_weight('balanced', y_train_encoded)
        
        # Print class distribution
        self._print_class_distribution(y_train_encoded)
        
        return X_train, X_test, y_train_encoded, y_test_encoded, sample_weights
    
    def _identify_columns(self, X: pd.DataFrame) -> None:
        """Identify different column types in the dataset."""
        self.column_info = {
            'num_cols': X.select_dtypes(include=['int64', 'float64']).columns.tolist(),
            'cat_cols': X.select_dtypes(include=['category', 'object']).columns.tolist(),
            'bool_cols': X.select_dtypes(include=['bool']).columns.tolist(),
            'time_cat_cols': [col for col in self.config.TIME_CAT_COLS if col in X.columns]
        }
        
        # Columns to encode (excluding time-based)
        self.column_info['encode_cat_cols'] = [
            col for col in self.column_info['cat_cols'] 
            if col not in self.column_info['time_cat_cols']
        ]
        
        print("Column Types Identified:")
        print(f"  Numerical: {len(self.column_info['num_cols'])}")
        print(f"  Categorical (to encode): {len(self.column_info['encode_cat_cols'])}")
        print(f"  Time categorical (passthrough): {len(self.column_info['time_cat_cols'])}")
        print(f"  Boolean: {len(self.column_info['bool_cols'])}")
    
    def _create_preprocessor(self) -> ColumnTransformer:
        """Create preprocessing pipeline."""
        transformers = []
        
        # Numerical pipeline
        if self.column_info['num_cols']:
            transformers.append((
                'num',
                Pipeline([
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', MinMaxScaler())
                ]),
                self.column_info['num_cols']
            ))
        
        # Categorical pipeline
        if self.column_info['encode_cat_cols']:
            transformers.append((
                'cat',
                Pipeline([
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('encoder', OrdinalEncoder(
                        handle_unknown='use_encoded_value',
                        unknown_value=-1
                    ))
                ]),
                self.column_info['encode_cat_cols']
            ))
        
        # Passthrough for time categorical
        if self.column_info['time_cat_cols']:
            transformers.append((
                'time', 'passthrough', self.column_info['time_cat_cols']
            ))
        
        # Passthrough for boolean
        if self.column_info['bool_cols']:
            transformers.append((
                'bool', 'passthrough', self.column_info['bool_cols']
            ))
        
        return ColumnTransformer(
            transformers=transformers,
            remainder='drop'
        )
    
    def _print_class_distribution(self, y: np.ndarray) -> None:
        """Print class distribution information."""
        unique, counts = np.unique(y, return_counts=True)
        total = len(y)
        
        print("\nClass Distribution:")
        for cls, count in zip(unique, counts):
            original_label = self.label_encoder.inverse_transform([cls])[0]
            pct = (count / total) * 100
            print(f"  Class {original_label}: {count:,} ({pct:.2f}%)")
    
    def get_feature_names(self) -> List[str]:
        """Get feature names after preprocessing."""
        feature_names = []
        
        if self.column_info['num_cols']:
            feature_names.extend(self.column_info['num_cols'])
        if self.column_info['encode_cat_cols']:
            feature_names.extend(self.column_info['encode_cat_cols'])
        if self.column_info['time_cat_cols']:
            feature_names.extend(self.column_info['time_cat_cols'])
        if self.column_info['bool_cols']:
            feature_names.extend(self.column_info['bool_cols'])
            
        return feature_names

# %% [markdown]
# # 4. Model Factory Module

# %%
class ModelFactory:
    """Factory class for creating model pipelines and parameter distributions."""
    
    def __init__(self, preprocessor: ColumnTransformer, config: Config = Config()):
        self.preprocessor = preprocessor
        self.config = config
    
    def get_model_configs(self) -> Dict[str, Dict]:
        """
        Get all model configurations with pipelines and parameter distributions.
        
        Returns:
            Dictionary containing model name -> {pipeline, param_dist}
        """
        return {
            'XGBoost': {
                'pipeline': self._create_xgb_pipeline(),
                'param_dist': self._get_xgb_params(),
                'fit_params': lambda sw: {'classifier__sample_weight': sw}
            },
            'LightGBM': {
                'pipeline': self._create_lgb_pipeline(),
                'param_dist': self._get_lgb_params(),
                'fit_params': lambda sw: {}  # Uses class_weight='balanced'
            },
            'CatBoost': {
                'pipeline': self._create_catboost_pipeline(),
                'param_dist': self._get_catboost_params(),
                'fit_params': lambda sw: {}  # Uses auto_class_weights
            }
        }
    
    def _create_xgb_pipeline(self) -> Pipeline:
        """Create XGBoost pipeline."""
        return Pipeline([
            ('preprocessor', self.preprocessor),
            ('classifier', xgb.XGBClassifier(
                objective='multi:softprob',
                eval_metric='mlogloss',
                random_state=self.config.RANDOM_STATE,
                n_jobs=self.config.N_JOBS,
                use_label_encoder=False
            ))
        ])
    
    def _create_lgb_pipeline(self) -> Pipeline:
        """Create LightGBM pipeline."""
        return Pipeline([
            ('preprocessor', self.preprocessor),
            ('classifier', LGBMClassifier(
                random_state=self.config.RANDOM_STATE,
                n_jobs=self.config.N_JOBS,
                class_weight='balanced',
                verbose=-1
            ))
        ])
    
    def _create_catboost_pipeline(self) -> Pipeline:
        """Create CatBoost pipeline."""
        return Pipeline([
            ('preprocessor', self.preprocessor),
            ('classifier', CatBoostClassifier(
                random_state=self.config.RANDOM_STATE,
                auto_class_weights='Balanced',
                verbose=0
            ))
        ])
    
    def _get_xgb_params(self) -> Dict:
        """Get XGBoost parameter distribution for RandomizedSearchCV."""
        return {
            'classifier__n_estimators': randint(100, 500),
            'classifier__max_depth': randint(3, 12),
            'classifier__learning_rate': loguniform(0.01, 0.3),
            'classifier__min_child_weight': randint(1, 10),
            'classifier__subsample': uniform(0.6, 0.4),
            'classifier__colsample_bytree': uniform(0.6, 0.4),
            'classifier__gamma': uniform(0, 0.5),
            'classifier__reg_alpha': loguniform(1e-3, 10),
            'classifier__reg_lambda': loguniform(1e-3, 10)
        }
    
    def _get_lgb_params(self) -> Dict:
        """Get LightGBM parameter distribution for RandomizedSearchCV."""
        return {
            'classifier__n_estimators': randint(100, 500),
            'classifier__max_depth': randint(3, 12),
            'classifier__learning_rate': loguniform(0.01, 0.3),
            'classifier__num_leaves': randint(20, 150),
            'classifier__min_child_samples': randint(10, 100),
            'classifier__subsample': uniform(0.6, 0.4),
            'classifier__colsample_bytree': uniform(0.6, 0.4),
            'classifier__reg_alpha': loguniform(1e-3, 10),
            'classifier__reg_lambda': loguniform(1e-3, 10)
        }
    
    def _get_catboost_params(self) -> Dict:
        """Get CatBoost parameter distribution for RandomizedSearchCV."""
        return {
            'classifier__iterations': randint(100, 500),
            'classifier__depth': randint(4, 10),
            'classifier__learning_rate': loguniform(0.01, 0.3),
            'classifier__l2_leaf_reg': loguniform(1, 10),
            'classifier__border_count': randint(32, 255),
            'classifier__bagging_temperature': uniform(0, 1)
        }

# %% [markdown]
# # 5. Model Trainer Module

# %%
class ModelTrainer:
    """Handles model training with RandomizedSearchCV."""
    
    def __init__(self, config: Config = Config()):
        self.config = config
        self.cv = StratifiedKFold(
            n_splits=config.CV_FOLDS,
            shuffle=True,
            random_state=config.RANDOM_STATE
        )
        self.trained_models = {}
        self.search_results = {}
    
    def train_all_models(
        self,
        model_configs: Dict,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        sample_weights: np.ndarray
    ) -> Dict:
        """
        Train all models using RandomizedSearchCV.
        
        Returns:
            Dictionary of trained models with their search results
        """
        print("=" * 60)
        print("STARTING RANDOMIZED SEARCH CV FOR ALL MODELS")
        print("=" * 60)
        print(f"CV Folds: {self.config.CV_FOLDS}")
        print(f"N Iterations: {self.config.N_ITER_SEARCH}")
        print(f"Scoring Metric: {self.config.SCORING_METRIC}")
        print("=" * 60)
        
        for model_name, model_config in model_configs.items():
            print(f"\n{'='*60}")
            print(f"Training: {model_name}")
            print('='*60)
            
            # Create RandomizedSearchCV
            random_search = RandomizedSearchCV(
                estimator=model_config['pipeline'],
                param_distributions=model_config['param_dist'],
                n_iter=self.config.N_ITER_SEARCH,
                cv=self.cv,
                scoring=self.config.SCORING_METRIC,
                n_jobs=self.config.N_JOBS,
                verbose=1,
                random_state=self.config.RANDOM_STATE,
                refit=True,
                return_train_score=True
            )
            
            # Get fit parameters
            fit_params = model_config['fit_params'](sample_weights)
            
            # Fit
            random_search.fit(X_train, y_train, **fit_params)
            
            # Store results
            self.trained_models[model_name] = random_search.best_estimator_
            self.search_results[model_name] = {
                'best_params': random_search.best_params_,
                'best_score': random_search.best_score_,
                'cv_results': pd.DataFrame(random_search.cv_results_),
                'search_object': random_search
            }
            
            print(f"\n{model_name} Best CV Score: {random_search.best_score_:.4f}")
            print(f"Best Parameters:")
            for param, value in random_search.best_params_.items():
                param_name = param.replace('classifier__', '')
                if isinstance(value, float):
                    print(f"  {param_name}: {value:.6f}")
                else:
                    print(f"  {param_name}: {value}")
        
        return self.trained_models
    
    def get_cv_results_summary(self) -> pd.DataFrame:
        """Get summary of CV results for all models."""
        summary_data = []
        
        for model_name, results in self.search_results.items():
            cv_df = results['cv_results']
            summary_data.append({
                'Model': model_name,
                'Best CV Score': results['best_score'],
                'Mean Train Score': cv_df.loc[cv_df['rank_test_score'] == 1, 'mean_train_score'].values[0],
                'Std CV Score': cv_df.loc[cv_df['rank_test_score'] == 1, 'std_test_score'].values[0],
                'Mean Fit Time (s)': cv_df.loc[cv_df['rank_test_score'] == 1, 'mean_fit_time'].values[0]
            })
        
        return pd.DataFrame(summary_data).sort_values('Best CV Score', ascending=False)

# %% [markdown]
# # 6. Model Evaluator Module

# %%
class ModelEvaluator:
    """Handles comprehensive model evaluation."""
    
    def __init__(self, label_encoder: LabelEncoder, config: Config = Config()):
        self.label_encoder = label_encoder
        self.config = config
        self.evaluation_results = {}
    
    def evaluate_all_models(
        self,
        models: Dict,
        X_test: pd.DataFrame,
        y_test: np.ndarray
    ) -> pd.DataFrame:
        """
        Evaluate all models on test set.
        
        Returns:
            DataFrame with evaluation metrics for all models
        """
        print("\n" + "=" * 60)
        print("MODEL EVALUATION ON TEST SET")
        print("=" * 60)
        
        for model_name, model in models.items():
            print(f"\nEvaluating: {model_name}")
            self.evaluation_results[model_name] = self._evaluate_single_model(
                model, X_test, y_test, model_name
            )
        
        # Create comparison DataFrame
        metrics_df = pd.DataFrame(self.evaluation_results).T
        metrics_df = metrics_df.sort_values('Balanced Accuracy', ascending=False)
        
        return metrics_df
    
    def _evaluate_single_model(
        self,
        model,
        X_test: pd.DataFrame,
        y_test: np.ndarray,
        model_name: str
    ) -> Dict:
        """Evaluate a single model and return metrics."""
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
        
        metrics = {
            'Balanced Accuracy': balanced_accuracy_score(y_test, y_pred),
            'Macro Precision': precision_score(y_test, y_pred, average='macro', zero_division=0),
            'Macro Recall': recall_score(y_test, y_pred, average='macro', zero_division=0),
            'Macro F1': f1_score(y_test, y_pred, average='macro', zero_division=0),
            'Weighted F1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
            'MCC': matthews_corrcoef(y_test, y_pred)
        }
        
        # ROC-AUC if probabilities available
        if y_proba is not None:
            try:
                metrics['ROC-AUC (OvR)'] = roc_auc_score(
                    y_test, y_proba, multi_class='ovr', average='macro'
                )
                metrics['ROC-AUC (OvO)'] = roc_auc_score(
                    y_test, y_proba, multi_class='ovo', average='macro'
                )
            except Exception as e:
                print(f"  Warning: Could not compute ROC-AUC: {e}")
        
        # Store predictions for later use
        metrics['_y_pred'] = y_pred
        metrics['_y_proba'] = y_proba
        
        return metrics
    
    def get_best_model(self, models: Dict, metric: str = 'Balanced Accuracy') -> Tuple[str, Any]:
        """
        Get the best performing model based on specified metric.
        
        Returns:
            Tuple of (model_name, model_object)
        """
        if not self.evaluation_results:
            raise ValueError("No evaluation results. Run evaluate_all_models first.")
        
        best_model_name = max(
            self.evaluation_results.keys(),
            key=lambda x: self.evaluation_results[x].get(metric, 0)
        )
        
        print(f"\nBest Model based on {metric}: {best_model_name}")
        print(f"Score: {self.evaluation_results[best_model_name][metric]:.4f}")
        
        return best_model_name, models[best_model_name]
    
    def print_classification_reports(
        self,
        models: Dict,
        X_test: pd.DataFrame,
        y_test: np.ndarray
    ) -> None:
        """Print classification reports for all models."""
        for model_name, model in models.items():
            y_pred = model.predict(X_test)
            
            print(f"\n{'='*60}")
            print(f"Classification Report: {model_name}")
            print('='*60)
            print(classification_report(
                y_test,
                y_pred,
                target_names=[str(c) for c in self.label_encoder.classes_],
                zero_division=0
            ))

# %% [markdown]
# # 7. Visualization Module

# %%
class Visualizer:
    """Handles all visualizations for the ML pipeline."""
    
    def __init__(self, label_encoder: LabelEncoder, config: Config = Config()):
        self.label_encoder = label_encoder
        self.config = config
        self.class_names = [str(c) for c in label_encoder.classes_]
    
    def plot_all_visualizations(
        self,
        models: Dict,
        X_test: pd.DataFrame,
        y_test: np.ndarray,
        evaluation_results: Dict,
        search_results: Dict
    ) -> None:
        """Generate all visualizations."""
        print("\n" + "=" * 60)
        print("GENERATING VISUALIZATIONS")
        print("=" * 60)
        
        # 1. Model Comparison Bar Chart
        self.plot_model_comparison(evaluation_results)
        
        # 2. Confusion Matrices
        self.plot_confusion_matrices(models, X_test, y_test)
        
        # 3. ROC Curves
        self.plot_roc_curves(models, X_test, y_test)
        
        # 4. Precision-Recall Curves
        self.plot_precision_recall_curves(models, X_test, y_test)
        
        # 5. Hyperparameter Search Results
        self.plot_search_results(search_results)
        
        # 6. Per-Class Performance
        self.plot_per_class_performance(models, X_test, y_test)
    
    def plot_model_comparison(self, evaluation_results: Dict) -> None:
        """Plot model comparison bar chart."""
        # Filter out internal keys
        metrics_to_plot = ['Balanced Accuracy', 'Macro F1', 'MCC', 'ROC-AUC (OvR)']
        
        data = []
        for model_name, metrics in evaluation_results.items():
            for metric in metrics_to_plot:
                if metric in metrics and not metric.startswith('_'):
                    data.append({
                        'Model': model_name,
                        'Metric': metric,
                        'Score': metrics[metric]
                    })
        
        df = pd.DataFrame(data)
        
        fig, ax = plt.subplots(figsize=self.config.FIGSIZE_MEDIUM)
        
        x = np.arange(len(metrics_to_plot))
        width = 0.25
        models = list(evaluation_results.keys())
        
        for i, model in enumerate(models):
            model_data = df[df['Model'] == model]
            scores = [model_data[model_data['Metric'] == m]['Score'].values[0] 
                     if len(model_data[model_data['Metric'] == m]) > 0 else 0 
                     for m in metrics_to_plot]
            ax.bar(x + i * width, scores, width, label=model)
        
        ax.set_xlabel('Metric')
        ax.set_ylabel('Score')
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(x + width)
        ax.set_xticklabels(metrics_to_plot, rotation=15)
        ax.legend()
        ax.set_ylim(0, 1)
        
        # Add value labels
        for i, model in enumerate(models):
            model_data = df[df['Model'] == model]
            scores = [model_data[model_data['Metric'] == m]['Score'].values[0] 
                     if len(model_data[model_data['Metric'] == m]) > 0 else 0 
                     for m in metrics_to_plot]
            for j, score in enumerate(scores):
                ax.annotate(f'{score:.3f}', xy=(x[j] + i * width, score),
                           ha='center', va='bottom', fontsize=8, rotation=90)
        
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def plot_confusion_matrices(
        self,
        models: Dict,
        X_test: pd.DataFrame,
        y_test: np.ndarray
    ) -> None:
        """Plot confusion matrices for all models."""
        n_models = len(models)
        fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
        
        if n_models == 1:
            axes = [axes]
        
        for ax, (model_name, model) in zip(axes, models.items()):
            y_pred = model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            
            # Normalize
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            sns.heatmap(
                cm_normalized,
                annot=True,
                fmt='.2%',
                cmap='Blues',
                xticklabels=self.class_names,
                yticklabels=self.class_names,
                ax=ax
            )
            ax.set_title(f'{model_name}\nConfusion Matrix (Normalized)')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig('confusion_matrices.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        # Also plot raw counts
        fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
        
        if n_models == 1:
            axes = [axes]
        
        for ax, (model_name, model) in zip(axes, models.items()):
            y_pred = model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            
            sns.heatmap(
                cm,
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=self.class_names,
                yticklabels=self.class_names,
                ax=ax
            )
            ax.set_title(f'{model_name}\nConfusion Matrix (Counts)')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig('confusion_matrices_counts.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def plot_roc_curves(
        self,
        models: Dict,
        X_test: pd.DataFrame,
        y_test: np.ndarray
    ) -> None:
        """Plot ROC curves for all models and classes."""
        n_classes = len(self.class_names)
        
        fig, axes = plt.subplots(2, 2, figsize=self.config.FIGSIZE_LARGE)
        axes = axes.flatten()
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(models)))
        
        # Plot ROC for each class
        for class_idx in range(min(n_classes, 4)):
            ax = axes[class_idx]
            
            for (model_name, model), color in zip(models.items(), colors):
                if hasattr(model, 'predict_proba'):
                    y_proba = model.predict_proba(X_test)
                    y_test_binary = (y_test == class_idx).astype(int)
                    
                    fpr, tpr, _ = roc_curve(y_test_binary, y_proba[:, class_idx])
                    roc_auc = auc(fpr, tpr)
                    
                    ax.plot(fpr, tpr, color=color, lw=2,
                           label=f'{model_name} (AUC = {roc_auc:.3f})')
            
            ax.plot([0, 1], [0, 1], 'k--', lw=1)
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title(f'ROC Curve - Class {self.class_names[class_idx]} (Severity)')
            ax.legend(loc='lower right', fontsize=8)
        
        plt.tight_layout()
        plt.savefig('roc_curves.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def plot_precision_recall_curves(
        self,
        models: Dict,
        X_test: pd.DataFrame,
        y_test: np.ndarray
    ) -> None:
        """Plot Precision-Recall curves (especially important for imbalanced classes)."""
        n_classes = len(self.class_names)
        
        fig, axes = plt.subplots(2, 2, figsize=self.config.FIGSIZE_LARGE)
        axes = axes.flatten()
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(models)))
        
        for class_idx in range(min(n_classes, 4)):
            ax = axes[class_idx]
            
            for (model_name, model), color in zip(models.items(), colors):
                if hasattr(model, 'predict_proba'):
                    y_proba = model.predict_proba(X_test)
                    y_test_binary = (y_test == class_idx).astype(int)
                    
                    precision, recall, _ = precision_recall_curve(
                        y_test_binary, y_proba[:, class_idx]
                    )
                    avg_precision = average_precision_score(
                        y_test_binary, y_proba[:, class_idx]
                    )
                    
                    ax.plot(recall, precision, color=color, lw=2,
                           label=f'{model_name} (AP = {avg_precision:.3f})')
            
            # Add baseline
            baseline = (y_test == class_idx).mean()
            ax.axhline(y=baseline, color='k', linestyle='--', lw=1, 
                      label=f'Baseline ({baseline:.3f})')
            
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            ax.set_title(f'Precision-Recall - Class {self.class_names[class_idx]} (Severity)')
            ax.legend(loc='upper right', fontsize=8)
        
        plt.tight_layout()
        plt.savefig('precision_recall_curves.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def plot_search_results(self, search_results: Dict) -> None:
        """Plot hyperparameter search results."""
        fig, axes = plt.subplots(1, len(search_results), figsize=(6 * len(search_results), 5))
        
        if len(search_results) == 1:
            axes = [axes]
        
        for ax, (model_name, results) in zip(axes, search_results.items()):
            cv_df = results['cv_results']
            
            # Sort by mean test score
            sorted_df = cv_df.sort_values('mean_test_score', ascending=True).tail(20)
            
            ax.barh(range(len(sorted_df)), sorted_df['mean_test_score'])
            ax.set_yticks(range(len(sorted_df)))
            ax.set_yticklabels([f"Config {i+1}" for i in range(len(sorted_df))])
            ax.set_xlabel('Mean CV Score')
            ax.set_title(f'{model_name}\nTop 20 Configurations')
            ax.axvline(x=results['best_score'], color='r', linestyle='--', 
                      label=f"Best: {results['best_score']:.4f}")
            ax.legend()
        
        plt.tight_layout()
        plt.savefig('search_results.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def plot_per_class_performance(
        self,
        models: Dict,
        X_test: pd.DataFrame,
        y_test: np.ndarray
    ) -> None:
        """Plot per-class recall and precision for all models."""
        data = []
        
        for model_name, model in models.items():
            y_pred = model.predict(X_test)
            
            for class_idx, class_name in enumerate(self.class_names):
                mask = y_test == class_idx
                if mask.sum() > 0:
                    recall = recall_score(y_test, y_pred, labels=[class_idx], average=None, zero_division=0)[0]
                    precision = precision_score(y_test, y_pred, labels=[class_idx], average=None, zero_division=0)[0]
                    f1 = f1_score(y_test, y_pred, labels=[class_idx], average=None, zero_division=0)[0]
                    
                    data.append({
                        'Model': model_name,
                        'Class': class_name,
                        'Recall': recall,
                        'Precision': precision,
                        'F1': f1,
                        'Support': mask.sum()
                    })
        
        df = pd.DataFrame(data)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        metrics = ['Recall', 'Precision', 'F1']
        for ax, metric in zip(axes, metrics):
            pivot_df = df.pivot(index='Class', columns='Model', values=metric)
            pivot_df.plot(kind='bar', ax=ax)
            ax.set_title(f'Per-Class {metric}')
            ax.set_xlabel('Severity Class')
            ax.set_ylabel(metric)
            ax.legend(title='Model')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
            ax.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig('per_class_performance.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        # Print support for each class
        print("\nClass Support (Test Set):")
        for class_name in self.class_names:
            support = df[df['Class'] == class_name]['Support'].iloc[0]
            print(f"  Class {class_name}: {support:,}")

# %% [markdown]
# # 8. SHAP Explainer Module

# %%
class SHAPExplainer:
    """Handles SHAP-based model explainability, focusing on high-severity classes."""
    
    def __init__(
        self,
        label_encoder: LabelEncoder,
        feature_names: List[str],
        config: Config = Config()
    ):
        self.label_encoder = label_encoder
        self.feature_names = feature_names
        self.config = config
        self.class_names = [str(c) for c in label_encoder.classes_]
        self.shap_values = None
        self.explainer = None
    
    def explain_model(
        self,
        model,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        model_name: str = "Model",
        focus_classes: List[int] = [2, 3]  # Classes 3 and 4 in original labels (indices 2, 3)
    ) -> None:
        """
        Generate SHAP explanations with focus on high-severity classes.
        
        Args:
            model: Trained model pipeline
            X_train: Training data for background
            X_test: Test data for explanations
            model_name: Name of the model for titles
            focus_classes: List of class indices to focus on (for Severity 3 and 4)
        """
        print("\n" + "=" * 60)
        print(f"SHAP ANALYSIS FOR {model_name}")
        print("=" * 60)
        
        # Get preprocessed data
        preprocessor = model.named_steps['preprocessor']
        classifier = model.named_steps['classifier']
        
        X_train_processed = preprocessor.transform(X_train)
        X_test_processed = preprocessor.transform(X_test)
        
        # Sample for efficiency
        sample_size = min(self.config.SHAP_SAMPLE_SIZE, len(X_test_processed))
        indices = np.random.choice(len(X_test_processed), sample_size, replace=False)
        X_sample = X_test_processed[indices]
        
        print(f"Computing SHAP values for {sample_size} samples...")
        
        # Create explainer based on model type
        if isinstance(classifier, (xgb.XGBClassifier, LGBMClassifier)):
            self.explainer = shap.TreeExplainer(classifier)
            self.shap_values = self.explainer.shap_values(X_sample)
        elif isinstance(classifier, CatBoostClassifier):
            self.explainer = shap.TreeExplainer(classifier)
            self.shap_values = self.explainer.shap_values(X_sample)
        else:
            # Fallback to KernelExplainer
            background = shap.sample(X_train_processed, 100)
            self.explainer = shap.KernelExplainer(classifier.predict_proba, background)
            self.shap_values = self.explainer.shap_values(X_sample)
        
        # Generate all SHAP visualizations
        self._plot_summary_all_classes(X_sample, model_name)
        self._plot_summary_focused_classes(X_sample, focus_classes, model_name)
        self._plot_feature_importance_comparison(focus_classes, model_name)
        self._plot_dependence_plots(X_sample, focus_classes, model_name)
        self._plot_decision_plots(X_sample, focus_classes, model_name)
        
        print("\nSHAP analysis complete!")
    
    def _plot_summary_all_classes(self, X_sample: np.ndarray, model_name: str) -> None:
        """Plot SHAP summary for all classes."""
        print("\nGenerating summary plot for all classes...")
        
        fig, axes = plt.subplots(2, 2, figsize=self.config.FIGSIZE_LARGE)
        axes = axes.flatten()
        
        for idx, (ax, class_name) in enumerate(zip(axes, self.class_names)):
            plt.sca(ax)
            
            if isinstance(self.shap_values, list):
                shap_vals = self.shap_values[idx]
            else:
                shap_vals = self.shap_values[:, :, idx] if len(self.shap_values.shape) == 3 else self.shap_values
            
            shap.summary_plot(
                shap_vals,
                X_sample,
                feature_names=self.feature_names,
                max_display=10,
                show=False,
                plot_size=None
            )
            ax.set_title(f'Severity {class_name}')
        
        plt.suptitle(f'{model_name} - SHAP Summary (All Classes)', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(f'shap_summary_all_{model_name.lower().replace(" ", "_")}.png', 
                   dpi=150, bbox_inches='tight')
        plt.show()
    
    def _plot_summary_focused_classes(
        self,
        X_sample: np.ndarray,
        focus_classes: List[int],
        model_name: str
    ) -> None:
        """Plot detailed SHAP summary for high-severity classes (3 and 4)."""
        print(f"\nGenerating detailed summary for high-severity classes...")
        
        n_focus = len(focus_classes)
        fig, axes = plt.subplots(1, n_focus, figsize=(8 * n_focus, 8))
        
        if n_focus == 1:
            axes = [axes]
        
        for ax, class_idx in zip(axes, focus_classes):
            plt.sca(ax)
            
            if class_idx < len(self.class_names):
                if isinstance(self.shap_values, list):
                    shap_vals = self.shap_values[class_idx]
                else:
                    shap_vals = self.shap_values[:, :, class_idx] if len(self.shap_values.shape) == 3 else self.shap_values
                
                shap.summary_plot(
                    shap_vals,
                    X_sample,
                    feature_names=self.feature_names,
                    max_display=self.config.TOP_FEATURES_SHAP,
                    show=False,
                    plot_size=None
                )
                ax.set_title(f'Severity {self.class_names[class_idx]}\n(High Severity)', fontsize=12)
        
        plt.suptitle(f'{model_name} - SHAP Analysis for High-Severity Classes', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(f'shap_high_severity_{model_name.lower().replace(" ", "_")}.png', 
                   dpi=150, bbox_inches='tight')
        plt.show()
    
    def _plot_feature_importance_comparison(
        self,
        focus_classes: List[int],
        model_name: str
    ) -> None:
        """Plot feature importance comparison between high-severity classes."""
        print("\nGenerating feature importance comparison...")
        
        importance_data = []
        
        for class_idx in focus_classes:
            if class_idx < len(self.class_names):
                if isinstance(self.shap_values, list):
                    shap_vals = self.shap_values[class_idx]
                else:
                    shap_vals = self.shap_values[:, :, class_idx] if len(self.shap_values.shape) == 3 else self.shap_values
                
                # Mean absolute SHAP values
                mean_abs_shap = np.abs(shap_vals).mean(axis=0)
                
                for feat_idx, feat_name in enumerate(self.feature_names):
                    if feat_idx < len(mean_abs_shap):
                        importance_data.append({
                            'Feature': feat_name,
                            'Class': f'Severity {self.class_names[class_idx]}',
                            'Mean |SHAP|': mean_abs_shap[feat_idx]
                        })
        
        df = pd.DataFrame(importance_data)
        
        # Get top features
        top_features = df.groupby('Feature')['Mean |SHAP|'].mean().nlargest(15).index.tolist()
        df_top = df[df['Feature'].isin(top_features)]
        
        fig, ax = plt.subplots(figsize=self.config.FIGSIZE_MEDIUM)
        
        pivot_df = df_top.pivot(index='Feature', columns='Class', values='Mean |SHAP|')
        pivot_df = pivot_df.reindex(top_features)
        
        pivot_df.plot(kind='barh', ax=ax)
        ax.set_xlabel('Mean |SHAP Value|')
        ax.set_title(f'{model_name}\nFeature Importance for High-Severity Classes')
        ax.legend(title='Severity')
        
        plt.tight_layout()
        plt.savefig(f'shap_importance_comparison_{model_name.lower().replace(" ", "_")}.png', 
                   dpi=150, bbox_inches='tight')
        plt.show()
    
    def _plot_dependence_plots(
        self,
        X_sample: np.ndarray,
        focus_classes: List[int],
        model_name: str
    ) -> None:
        """Plot SHAP dependence plots for top features in high-severity classes."""
        print("\nGenerating dependence plots for high-severity classes...")
        
        for class_idx in focus_classes:
            if class_idx >= len(self.class_names):
                continue
                
            if isinstance(self.shap_values, list):
                shap_vals = self.shap_values[class_idx]
            else:
                shap_vals = self.shap_values[:, :, class_idx] if len(self.shap_values.shape) == 3 else self.shap_values
            
            # Get top 4 features
            mean_abs_shap = np.abs(shap_vals).mean(axis=0)
            top_indices = np.argsort(mean_abs_shap)[-4:][::-1]
            
            fig, axes = plt.subplots(2, 2, figsize=self.config.FIGSIZE_LARGE)
            axes = axes.flatten()
            
            for ax, feat_idx in zip(axes, top_indices):
                if feat_idx < len(self.feature_names):
                    plt.sca(ax)
                    shap.dependence_plot(
                        feat_idx,
                        shap_vals,
                        X_sample,
                        feature_names=self.feature_names,
                        show=False,
                        ax=ax
                    )
            
            plt.suptitle(f'{model_name} - Dependence Plots for Severity {self.class_names[class_idx]}', 
                        fontsize=14, y=1.02)
            plt.tight_layout()
            plt.savefig(f'shap_dependence_class{class_idx}_{model_name.lower().replace(" ", "_")}.png', 
                       dpi=150, bbox_inches='tight')
            plt.show()
    
    def _plot_decision_plots(
        self,
        X_sample: np.ndarray,
        focus_classes: List[int],
        model_name: str
    ) -> None:
        """Plot SHAP decision plots for high-severity predictions."""
        print("\nGenerating decision plots for high-severity predictions...")
        
        for class_idx in focus_classes:
            if class_idx >= len(self.class_names):
                continue
                
            if isinstance(self.shap_values, list):
                shap_vals = self.shap_values[class_idx]
                expected_value = self.explainer.expected_value[class_idx] if isinstance(
                    self.explainer.expected_value, (list, np.ndarray)
                ) else self.explainer.expected_value
            else:
                shap_vals = self.shap_values[:, :, class_idx] if len(self.shap_values.shape) == 3 else self.shap_values
                expected_value = self.explainer.expected_value[class_idx] if isinstance(
                    self.explainer.expected_value, (list, np.ndarray)
                ) else self.explainer.expected_value
            
            # Find samples with highest SHAP contribution for this class
            class_contributions = shap_vals.sum(axis=1)
            top_sample_indices = np.argsort(class_contributions)[-10:]
            
            fig, ax = plt.subplots(figsize=(12, 8))
            
            try:
                shap.decision_plot(
                    expected_value,
                    shap_vals[top_sample_indices],
                    feature_names=self.feature_names,
                    show=False
                )
                plt.title(f'{model_name} - Decision Plot for Severity {self.class_names[class_idx]}\n(Top 10 High-Risk Predictions)')
                plt.tight_layout()
                plt.savefig(f'shap_decision_class{class_idx}_{model_name.lower().replace(" ", "_")}.png', 
                           dpi=150, bbox_inches='tight')
                plt.show()
            except Exception as e:
                print(f"  Warning: Could not generate decision plot for class {class_idx}: {e}")
    
    def get_top_features_for_severity(
        self,
        focus_classes: List[int] = [2, 3],
        n_features: int = 10
    ) -> pd.DataFrame:
        """
        Get top features contributing to high-severity predictions.
        
        Returns:
            DataFrame with top features for each severity class
        """
        results = []
        
        for class_idx in focus_classes:
            if class_idx >= len(self.class_names):
                continue
                
            if isinstance(self.shap_values, list):
                shap_vals = self.shap_values[class_idx]
            else:
                shap_vals = self.shap_values[:, :, class_idx] if len(self.shap_values.shape) == 3 else self.shap_values
            
            mean_abs_shap = np.abs(shap_vals).mean(axis=0)
            mean_shap = shap_vals.mean(axis=0)
            
            for feat_idx in np.argsort(mean_abs_shap)[-n_features:][::-1]:
                if feat_idx < len(self.feature_names):
                    results.append({
                        'Severity Class': self.class_names[class_idx],
                        'Feature': self.feature_names[feat_idx],
                        'Mean |SHAP|': mean_abs_shap[feat_idx],
                        'Mean SHAP': mean_shap[feat_idx],
                        'Direction': 'Increases Risk' if mean_shap[feat_idx] > 0 else 'Decreases Risk'
                    })
        
        return pd.DataFrame(results)

# %% [markdown]
# # 9. Main Pipeline Orchestrator

# %%
class MLPipeline:
    """Main orchestrator for the complete ML pipeline."""
    
    def __init__(self, config: Config = Config()):
        self.config = config
        self.data_preparator = None
        self.model_factory = None
        self.model_trainer = None
        self.model_evaluator = None
        self.visualizer = None
        self.shap_explainer = None
        
        # Data
        self.X_train = None
        self.X_test = None
        self.y_train_encoded = None
        self.y_test_encoded = None
        self.sample_weights = None
        
        # Models
        self.trained_models = None
        self.best_model_name = None
        self.best_model = None
    
    def run_complete_pipeline(
        self,
        df: pd.DataFrame,
        run_shap: bool = True,
        focus_severity_classes: List[int] = None
    ) -> Dict:
        """
        Run the complete ML pipeline.
        
        Args:
            df: Input DataFrame with features and target
            run_shap: Whether to run SHAP analysis
            focus_severity_classes: Classes to focus SHAP analysis on (default: [2, 3] for Severity 3 & 4)
        
        Returns:
            Dictionary with all results
        """
        # Default focus classes (indices for Severity 3 and 4)
        if focus_severity_classes is None:
            focus_severity_classes = [2, 3]
        
        print("=" * 60)
        print("STARTING COMPLETE ML PIPELINE")
        print("=" * 60)
        
        # Step 1: Data Preparation
        print("\n📊 STEP 1: Data Preparation")
        self.data_preparator = DataPreparator(self.config)
        (self.X_train, self.X_test, self.y_train_encoded, 
         self.y_test_encoded, self.sample_weights) = self.data_preparator.prepare_data(df)
        
        # Step 2: Create Model Factory
        print("\n🏭 STEP 2: Creating Model Configurations")
        self.model_factory = ModelFactory(
            self.data_preparator.preprocessor,
            self.config
        )
        model_configs = self.model_factory.get_model_configs()
        
        # Step 3: Train Models with RandomizedSearchCV
        print("\n🎯 STEP 3: Training Models with RandomizedSearchCV")
        self.model_trainer = ModelTrainer(self.config)
        self.trained_models = self.model_trainer.train_all_models(
            model_configs,
            self.X_train,
            self.y_train_encoded,
            self.sample_weights
        )
        
        # Print CV Results Summary
        cv_summary = self.model_trainer.get_cv_results_summary()
        print("\n📈 Cross-Validation Results Summary:")
        print(cv_summary.to_string(index=False))
        
        # Step 4: Evaluate Models
        print("\n📊 STEP 4: Evaluating Models on Test Set")
        self.model_evaluator = ModelEvaluator(
            self.data_preparator.label_encoder,
            self.config
        )
        evaluation_df = self.model_evaluator.evaluate_all_models(
            self.trained_models,
            self.X_test,
            self.y_test_encoded
        )
        
        # Print evaluation results
        print("\n📊 Test Set Evaluation Results:")
        display_cols = [col for col in evaluation_df.columns if not col.startswith('_')]
        print(evaluation_df[display_cols].to_string())
        
        # Step 5: Select Best Model
        print("\n🏆 STEP 5: Selecting Best Model")
        self.best_model_name, self.best_model = self.model_evaluator.get_best_model(
            self.trained_models,
            metric='Balanced Accuracy'
        )
        
        # Print classification reports
        self.model_evaluator.print_classification_reports(
            self.trained_models,
            self.X_test,
            self.y_test_encoded
        )
        
        # Step 6: Visualizations
        print("\n📊 STEP 6: Generating Visualizations")
        self.visualizer = Visualizer(
            self.data_preparator.label_encoder,
            self.config
        )
        self.visualizer.plot_all_visualizations(
            self.trained_models,
            self.X_test,
            self.y_test_encoded,
            self.model_evaluator.evaluation_results,
            self.model_trainer.search_results
        )
        
        # Step 7: SHAP Analysis
        if run_shap:
            print("\n🔍 STEP 7: SHAP Explainability Analysis")
            self._run_shap_analysis(focus_severity_classes)
        
        # Compile results
        results = {
            'trained_models': self.trained_models,
            'best_model_name': self.best_model_name,
            'best_model': self.best_model,
            'evaluation_results': evaluation_df,
            'cv_summary': cv_summary,
            'search_results': self.model_trainer.search_results,
            'label_encoder': self.data_preparator.label_encoder,
            'preprocessor': self.data_preparator.preprocessor,
            'feature_names': self.data_preparator.get_feature_names()
        }
        
        print("\n" + "=" * 60)
        print("PIPELINE COMPLETE!")
        print("=" * 60)
        print(f"\n🏆 Best Model: {self.best_model_name}")
        print(f"   Balanced Accuracy: {evaluation_df.loc[self.best_model_name, 'Balanced Accuracy']:.4f}")
        print(f"   Macro F1: {evaluation_df.loc[self.best_model_name, 'Macro F1']:.4f}")
        
        return results
    
    def _run_shap_analysis(self, focus_classes: List[int]) -> None:
        """Run SHAP analysis on all models with focus on high-severity classes."""
        feature_names = self.data_preparator.get_feature_names()
        
        self.shap_explainer = SHAPExplainer(
            self.data_preparator.label_encoder,
            feature_names,
            self.config
        )
        
        # Run SHAP for best model first
        print(f"\n🔍 Running SHAP analysis for best model: {self.best_model_name}")
        self.shap_explainer.explain_model(
            self.best_model,
            self.X_train,
            self.X_test,
            model_name=self.best_model_name,
            focus_classes=focus_classes
        )
        
        # Get and print top features for high severity
        print("\n📊 Top Features Contributing to High Severity (Classes 3 & 4):")
        top_features_df = self.shap_explainer.get_top_features_for_severity(
            focus_classes=focus_classes,
            n_features=10
        )
        print(top_features_df.to_string(index=False))
        
        # Optionally run SHAP for other models
        for model_name, model in self.trained_models.items():
            if model_name != self.best_model_name:
                print(f"\n🔍 Running SHAP analysis for: {model_name}")
                other_explainer = SHAPExplainer(
                    self.data_preparator.label_encoder,
                    feature_names,
                    self.config
                )
                other_explainer.explain_model(
                    model,
                    self.X_train,
                    self.X_test,
                    model_name=model_name,
                    focus_classes=focus_classes
                )

# %% [markdown]
# # 10. Usage Example

# %%
def main():
    """Main function demonstrating pipeline usage."""
    
    # Example usage (uncomment and modify path as needed):
    # 
    # # Load data
    # df = pd.read_csv('US_Accidents_March23.csv')
    # 
    # # Preprocess if needed (example)
    # df = df.dropna(subset=['Severity'])
    # 
    # # Initialize and run pipeline
    # config = Config()
    # config.N_ITER_SEARCH = 30  # Adjust based on computational resources
    # 
    # pipeline = MLPipeline(config)
    # results = pipeline.run_complete_pipeline(
    #     df,
    #     run_shap=True,
    #     focus_severity_classes=[2, 3]  # Indices for Severity 3 and 4
    # )
    # 
    # # Access results
    # best_model = results['best_model']
    # evaluation_df = results['evaluation_results']
    
    print("Pipeline module loaded successfully!")
    print("\nUsage:")
    print("  from ml_accident_severity_pipeline import MLPipeline, Config")
    print("  ")
    print("  config = Config()")
    print("  config.N_ITER_SEARCH = 50")
    print("  ")
    print("  pipeline = MLPipeline(config)")
    print("  results = pipeline.run_complete_pipeline(df, run_shap=True)")
    print("  ")
    print("  # Access best model")
    print("  best_model = results['best_model']")

if __name__ == "__main__":
    main()
