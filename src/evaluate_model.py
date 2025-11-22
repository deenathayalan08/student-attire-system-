"""
Comprehensive model evaluation utilities for the Student Attire Verification System.
Provides detailed metrics, visualizations, and comparison functions.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
from sklearn.model_selection import cross_val_score, cross_validate
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional
import streamlit as st
from pathlib import Path
import joblib
import os


class ModelEvaluator:
    """Comprehensive model evaluation class for attire classification."""

    def __init__(self, model_path: str = None):
        self.model_path = model_path
        self.model = None
        self.last_evaluation = {}

    def load_model(self, model_path: Optional[str] = None):
        """Load the trained model."""
        if model_path:
            self.model_path = model_path

        if self.model_path and os.path.exists(self.model_path):
            try:
                self.model = joblib.load(self.model_path)
                return True
            except Exception as e:
                st.error(f"Failed to load model: {e}")
                return False
        return False

    def evaluate_model_comprehensive(self, X_train: np.ndarray, y_train: np.ndarray,
                                   X_test: np.ndarray = None, y_test: np.ndarray = None,
                                   cv_folds: int = 5) -> Dict[str, Any]:
        """
        Perform comprehensive model evaluation including cross-validation and test set metrics.

        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features (optional)
            y_test: Test labels (optional)
            cv_folds: Number of cross-validation folds

        Returns:
            Dictionary containing all evaluation metrics
        """
        if self.model is None:
            return {"error": "Model not loaded"}

        results = {
            "cross_validation": {},
            "test_set_metrics": {},
            "per_class_metrics": {},
            "confusion_matrix": {},
            "feature_importance": {},
            "model_info": {}
        }

        # Cross-validation metrics
        try:
            cv_metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
            cv_results = cross_validate(
                self.model, X_train, y_train,
                cv=cv_folds,
                scoring=cv_metrics,
                return_train_score=False
            )

            results["cross_validation"] = {
                "accuracy": {
                    "mean": cv_results['test_accuracy'].mean(),
                    "std": cv_results['test_accuracy'].std(),
                    "scores": cv_results['test_accuracy'].tolist()
                },
                "precision_macro": {
                    "mean": cv_results['test_precision_macro'].mean(),
                    "std": cv_results['test_precision_macro'].std(),
                    "scores": cv_results['test_precision_macro'].tolist()
                },
                "recall_macro": {
                    "mean": cv_results['test_recall_macro'].mean(),
                    "std": cv_results['test_recall_macro'].std(),
                    "scores": cv_results['test_recall_macro'].tolist()
                },
                "f1_macro": {
                    "mean": cv_results['test_f1_macro'].mean(),
                    "std": cv_results['test_f1_macro'].std(),
                    "scores": cv_results['test_f1_macro'].tolist()
                }
            }
        except Exception as e:
            results["cross_validation"]["error"] = str(e)

        # Test set evaluation if provided
        if X_test is not None and y_test is not None:
            try:
                y_pred = self.model.predict(X_test)
                y_proba = self.model.predict_proba(X_test) if hasattr(self.model, 'predict_proba') else None

                results["test_set_metrics"] = {
                    "accuracy": accuracy_score(y_test, y_pred),
                    "precision_macro": precision_score(y_test, y_pred, average='macro', zero_division=0),
                    "precision_micro": precision_score(y_test, y_pred, average='micro', zero_division=0),
                    "recall_macro": recall_score(y_test, y_pred, average='macro', zero_division=0),
                    "recall_micro": recall_score(y_test, y_pred, average='micro', zero_division=0),
                    "f1_macro": f1_score(y_test, y_pred, average='macro', zero_division=0),
                    "f1_micro": f1_score(y_test, y_pred, average='micro', zero_division=0),
                }

                # Confusion matrix
                cm = confusion_matrix(y_test, y_pred)
                results["confusion_matrix"] = {
                    "matrix": cm.tolist(),
                    "labels": np.unique(np.concatenate([y_test, y_pred])).tolist()
                }

                # Per-class metrics
                class_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
                results["per_class_metrics"] = class_report

                # ROC-AUC if binary or multi-class with probabilities
                if y_proba is not None:
                    try:
                        if len(np.unique(y_test)) == 2:
                            results["test_set_metrics"]["roc_auc"] = roc_auc_score(y_test, y_proba[:, 1])
                        else:
                            results["test_set_metrics"]["roc_auc_ovr"] = roc_auc_score(y_test, y_proba, multi_class='ovr')
                    except:
                        pass

            except Exception as e:
                results["test_set_metrics"]["error"] = str(e)

        # Feature importance if available
        if hasattr(self.model, 'feature_importances_'):
            results["feature_importance"] = {
                "importances": self.model.feature_importances_.tolist(),
                "available": True
            }
        elif hasattr(self.model, 'coef_'):
            # For linear models
            coef = self.model.coef_
            if coef.ndim > 1:
                # Multi-class case
                importances = np.mean(np.abs(coef), axis=0)
            else:
                # Binary case
                importances = np.abs(coef)
            results["feature_importance"] = {
                "importances": importances.tolist(),
                "available": True
            }
        else:
            results["feature_importance"] = {"available": False}

        # Model info
        results["model_info"] = {
            "type": type(self.model).__name__,
            "parameters": self.model.get_params() if hasattr(self.model, 'get_params') else {},
            "num_features": X_train.shape[1] if X_train is not None else 0,
            "num_samples": len(X_train) if X_train is not None else 0
        }

        self.last_evaluation = results
        return results

    def compare_prediction_to_training(self, prediction_score: float,
                                     training_scores: List[float]) -> Dict[str, Any]:
        """
        Compare a single prediction score to the training dataset distribution.

        Args:
            prediction_score: The score from a single prediction (0-1)
            training_scores: List of scores from training/validation data

        Returns:
            Dictionary with comparison metrics
        """
        if not training_scores:
            return {"error": "No training scores provided"}

        training_array = np.array(training_scores)
        percentile = np.sum(training_array <= prediction_score) / len(training_array) * 100

        # Calculate statistics
        mean_score = np.mean(training_array)
        std_score = np.std(training_array)
        median_score = np.median(training_array)
        min_score = np.min(training_array)
        max_score = np.max(training_array)

        # Determine performance category
        if percentile >= 75:
            category = "excellent"
            color = "green"
        elif percentile >= 50:
            category = "good"
            color = "blue"
        elif percentile >= 25:
            category = "average"
            color = "orange"
        else:
            category = "needs_improvement"
            color = "red"

        # Calculate z-score
        z_score = (prediction_score - mean_score) / std_score if std_score > 0 else 0

        return {
            "prediction_score": prediction_score,
            "percentile_rank": percentile,
            "performance_category": category,
            "category_color": color,
            "training_stats": {
                "mean": mean_score,
                "std": std_score,
                "median": median_score,
                "min": min_score,
                "max": max_score,
                "count": len(training_scores)
            },
            "z_score": z_score,
            "deviation_from_mean": prediction_score - mean_score,
            "relative_performance": "above_average" if z_score > 0 else "below_average" if z_score < 0 else "average"
        }


def display_evaluation_results(results: Dict[str, Any], title: str = "Model Evaluation Results"):
    """Display comprehensive evaluation results in Streamlit."""
    st.subheader(f"📊 {title}")

    if "error" in results:
        st.error(f"Evaluation failed: {results['error']}")
        return

    # Cross-validation results
    if "cross_validation" in results and results["cross_validation"]:
        st.markdown("### 🔄 Cross-Validation Results")
        cv_data = []
        for metric, values in results["cross_validation"].items():
            if isinstance(values, dict) and "mean" in values:
                cv_data.append({
                    "Metric": metric.replace("_", " ").title(),
                    "Mean": ".3f",
                    "Std": ".3f",
                    "Range": ".3f"
                })

        if cv_data:
            cv_df = pd.DataFrame(cv_data)
            st.dataframe(cv_df, use_container_width=True)

            # Visual indicators
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                acc = results["cross_validation"].get("accuracy", {}).get("mean", 0)
                st.metric("CV Accuracy", ".1%", delta=None)
            with col2:
                prec = results["cross_validation"].get("precision_macro", {}).get("mean", 0)
                st.metric("CV Precision", ".1%", delta=None)
            with col3:
                rec = results["cross_validation"].get("recall_macro", {}).get("mean", 0)
                st.metric("CV Recall", ".1%", delta=None)
            with col4:
                f1 = results["cross_validation"].get("f1_macro", {}).get("mean", 0)
                st.metric("CV F1-Score", ".1%", delta=None)

    # Test set metrics
    if "test_set_metrics" in results and results["test_set_metrics"]:
        st.markdown("### 🧪 Test Set Performance")
        metrics = results["test_set_metrics"]

        # Main metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Accuracy", ".1%", delta=None)
        with col2:
            st.metric("Precision (Macro)", ".1%", delta=None)
        with col3:
            st.metric("Recall (Macro)", ".1%", delta=None)
        with col4:
            st.metric("F1-Score (Macro)", ".1%", delta=None)

        # Additional metrics
        if "roc_auc" in metrics:
            st.metric("ROC-AUC", ".3f")
        elif "roc_auc_ovr" in metrics:
            st.metric("ROC-AUC (OVR)", ".3f")

    # Confusion matrix
    if "confusion_matrix" in results and results["confusion_matrix"]:
        st.markdown("### 📈 Confusion Matrix")
        cm_data = results["confusion_matrix"]["matrix"]
        labels = results["confusion_matrix"].get("labels", [])

        if cm_data and len(cm_data) > 0:
            cm_df = pd.DataFrame(cm_data, index=labels, columns=labels)
            st.dataframe(cm_df, use_container_width=True)

    # Per-class metrics
    if "per_class_metrics" in results and results["per_class_metrics"]:
        st.markdown("### 📋 Per-Class Performance")
        class_metrics = results["per_class_metrics"]

        # Convert to displayable format
        class_data = []
        for class_name, metrics in class_metrics.items():
            if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
                class_data.append({
                    "Class": class_name,
                    "Precision": ".3f",
                    "Recall": ".3f",
                    "F1-Score": ".3f",
                    "Support": metrics.get('support', 0)
                })

        if class_data:
            class_df = pd.DataFrame(class_data)
            st.dataframe(class_df, use_container_width=True)

    # Model info
    if "model_info" in results and results["model_info"]:
        with st.expander("ℹ️ Model Information", expanded=False):
            info = results["model_info"]
            st.write(f"**Model Type:** {info.get('type', 'Unknown')}")
            st.write(f"**Training Samples:** {info.get('num_samples', 0)}")
            st.write(f"**Features:** {info.get('num_features', 0)}")

            if info.get('parameters'):
                st.markdown("**Parameters:**")
                st.json(info['parameters'])


def display_prediction_comparison(comparison: Dict[str, Any], title: str = "Prediction vs Training Comparison"):
    """Display prediction comparison results in Streamlit."""
    st.subheader(f"📊 {title}")

    if "error" in comparison:
        st.error(f"Comparison failed: {comparison['error']}")
        return

    # Main comparison metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        score = comparison.get("prediction_score", 0)
        st.metric("Prediction Score", ".1%")
    with col2:
        percentile = comparison.get("percentile_rank", 0)
        st.metric("Percentile Rank", ".1f")
    with col3:
        category = comparison.get("performance_category", "unknown").replace("_", " ").title()
        color = comparison.get("category_color", "gray")
        st.markdown(f"**Performance:** <span style='color:{color}'>{category}</span>", unsafe_allow_html=True)

    # Performance indicator
    st.markdown("### 🎯 Performance Indicator")

    # Create a visual gauge
    percentile = comparison.get("percentile_rank", 0)
    if percentile >= 75:
        gauge_color = "🟢"
        gauge_text = "Excellent"
    elif percentile >= 50:
        gauge_color = "🔵"
        gauge_text = "Good"
    elif percentile >= 25:
        gauge_color = "🟡"
        gauge_text = "Average"
    else:
        gauge_color = "🔴"
        gauge_text = "Needs Improvement"

    # Visual gauge using progress bar
    st.progress(percentile / 100)
    st.markdown(f"<h3 style='text-align: center; color: {comparison.get('category_color', 'gray')};'>{gauge_color} {gauge_text} ({percentile:.1f}th percentile)</h3>", unsafe_allow_html=True)

    # Training statistics comparison
    st.markdown("### 📈 Training Data Statistics")
    stats = comparison.get("training_stats", {})

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Training Mean", ".1%")
    with col2:
        st.metric("Training Std", ".1%")
    with col3:
        st.metric("Training Median", ".1%")
    with col4:
        st.metric("Training Range", ".1%")

    # Deviation analysis
    st.markdown("### 📊 Deviation Analysis")
    deviation = comparison.get("deviation_from_mean", 0)
    z_score = comparison.get("z_score", 0)

    col1, col2 = st.columns(2)
    with col1:
        deviation_color = "green" if deviation >= 0 else "red"
        st.metric("Deviation from Mean", "+.1%" if deviation >= 0 else ".1%", delta=None)
    with col2:
        st.metric("Z-Score", "+.2f" if z_score >= 0 else ".2f", delta=None)

    # Interpretation
    st.markdown("### 💡 Interpretation")
    relative_perf = comparison.get("relative_performance", "average")

    if relative_perf == "above_average":
        st.success("✅ This prediction performs **above average** compared to the training data.")
    elif relative_perf == "below_average":
        st.warning("⚠️ This prediction performs **below average** compared to the training data.")
    else:
        st.info("ℹ️ This prediction performs **at average** level compared to the training data.")

    # Additional context
    st.markdown("**What this means:**")
    st.write(f"- Your prediction score of {comparison.get('prediction_score', 0):.1%} ranks at the {percentile:.1f}th percentile of training samples.")
    st.write(f"- The average training score was {stats.get('mean', 0):.1%} with a standard deviation of {stats.get('std', 0):.1%}.")
    st.write(f"- This prediction is {abs(deviation):.1%} {'above' if deviation >= 0 else 'below'} the training mean.")


# Utility functions for easy access
def evaluate_model(model, X_train, y_train, X_test=None, y_test=None, cv_folds=5):
    """Convenience function to evaluate a model."""
    evaluator = ModelEvaluator()
    evaluator.model = model
    return evaluator.evaluate_model_comprehensive(X_train, y_train, X_test, y_test, cv_folds)


def compare_to_training(prediction_score, training_scores):
    """Convenience function to compare prediction to training data."""
    evaluator = ModelEvaluator()
    return evaluator.compare_prediction_to_training(prediction_score, training_scores)
