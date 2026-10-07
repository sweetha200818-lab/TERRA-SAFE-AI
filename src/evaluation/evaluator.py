"""
Model evaluation for TERRA SAFE AI - Comprehensive metrics and visualizations
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, auc,
    brier_score_loss
)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

from src.utils.config import get_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelEvaluator:
    """Comprehensive model evaluation for landslide prediction"""
    
    def __init__(self, config: Optional[Any] = None):
        self.config = config or get_config()
        self.results = {}
        self.thresholds = {
            'low_max': self.config.risk_thresholds.low_max,
            'moderate_max': self.config.risk_thresholds.moderate_max,
            'high_min': self.config.risk_thresholds.high_min
        }
    
    def evaluate(self, 
                 y_true: np.ndarray, 
                 y_pred: np.ndarray, 
                 y_prob: np.ndarray,
                 model_name: str = "model") -> Dict[str, Any]:
        """
        Comprehensive evaluation of model predictions
        
        Args:
            y_true: True labels
            y_pred: Predicted labels (0/1)
            y_prob: Predicted probabilities
            model_name: Name for the model
            
        Returns:
            Dictionary with all metrics and curves
        """
        results = {'model_name': model_name}
        
        # Basic metrics
        results['accuracy'] = accuracy_score(y_true, y_pred)
        results['precision'] = precision_score(y_true, y_pred, zero_division=0)
        results['recall'] = recall_score(y_true, y_pred, zero_division=0)
        results['f1_score'] = f1_score(y_true, y_pred, zero_division=0)
        results['roc_auc'] = roc_auc_score(y_true, y_prob)
        results['brier_score'] = brier_score_loss(y_true, y_prob)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        results['confusion_matrix'] = cm.tolist()
        results['tn'], results['fp'], results['fn'], results['tp'] = cm.ravel()
        
        # Derived metrics
        results['specificity'] = results['tn'] / (results['tn'] + results['fp']) if (results['tn'] + results['fp']) > 0 else 0
        results['npv'] = results['tn'] / (results['tn'] + results['fn']) if (results['tn'] + results['fn']) > 0 else 0
        results['fpr'] = results['fp'] / (results['fp'] + results['tn']) if (results['fp'] + results['tn']) > 0 else 0
        results['fnr'] = results['fn'] / (results['fn'] + results['tp']) if (results['fn'] + results['tp']) > 0 else 0
        
        # Classification report
        results['classification_report'] = classification_report(y_true, y_pred, output_dict=True)
        
        # ROC curve
        fpr, tpr, roc_thresholds = roc_curve(y_true, y_prob)
        results['roc_curve'] = {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'thresholds': roc_thresholds.tolist()
        }
        
        # Precision-Recall curve
        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_prob)
        results['pr_curve'] = {
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'thresholds': pr_thresholds.tolist()
        }
        results['pr_auc'] = auc(recall, precision)
        
        # Calibration curve
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
        results['calibration_curve'] = {
            'prob_true': prob_true.tolist(),
            'prob_pred': prob_pred.tolist()
        }
        
        # Optimal threshold (Youden's J statistic)
        youden_j = tpr - fpr
        optimal_idx = np.argmax(youden_j)
        results['optimal_threshold'] = float(roc_thresholds[optimal_idx])
        results['optimal_youden_j'] = float(youden_j[optimal_idx])
        
        # Threshold analysis
        results['threshold_analysis'] = self._analyze_thresholds(y_true, y_prob)
        
        # Risk classification metrics
        results['risk_classification'] = self._evaluate_risk_classification(y_true, y_prob)
        
        self.results[model_name] = results
        logger.info(f"{model_name} evaluation complete - AUC: {results['roc_auc']:.4f}, F1: {results['f1_score']:.4f}")
        
        return results
    
    def _analyze_thresholds(self, y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, Any]:
        """Analyze performance across different thresholds"""
        thresholds = np.arange(0.1, 1.0, 0.05)
        analysis = []
        
        for thresh in thresholds:
            y_pred_thresh = (y_prob >= thresh).astype(int)
            
            cm = confusion_matrix(y_true, y_pred_thresh)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
            else:
                tn, fp, fn, tp = 0, 0, 0, 0
            
            analysis.append({
                'threshold': round(float(thresh), 2),
                'accuracy': accuracy_score(y_true, y_pred_thresh),
                'precision': precision_score(y_true, y_pred_thresh, zero_division=0),
                'recall': recall_score(y_true, y_pred_thresh, zero_division=0),
                'f1': f1_score(y_true, y_pred_thresh, zero_division=0),
                'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
                'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn)
            })
        
        return {'thresholds': analysis}
    
    def _evaluate_risk_classification(self, y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, Any]:
        """Evaluate risk level classification (LOW/MODERATE/HIGH)"""
        # Classify probabilities into risk levels
        risk_pred = pd.cut(y_prob, 
                          bins=[0, self.thresholds['low_max'], self.thresholds['moderate_max'], 1.0],
                          labels=['LOW', 'MODERATE', 'HIGH'])
        
        # For true labels, we need to create pseudo risk levels
        # Since y_true is binary, we can't directly compare
        # Instead, evaluate how well probabilities separate classes
        
        low_mask = y_prob <= self.thresholds['low_max']
        mod_mask = (y_prob > self.thresholds['low_max']) & (y_prob <= self.thresholds['moderate_max'])
        high_mask = y_prob > self.thresholds['moderate_max']
        
        return {
            'low_risk': {
                'count': int(low_mask.sum()),
                'positive_rate': float(y_true[low_mask].mean()) if low_mask.any() else 0
            },
            'moderate_risk': {
                'count': int(mod_mask.sum()),
                'positive_rate': float(y_true[mod_mask].mean()) if mod_mask.any() else 0
            },
            'high_risk': {
                'count': int(high_mask.sum()),
                'positive_rate': float(y_true[high_mask].mean()) if high_mask.any() else 0
            }
        }
    
    def compare_models(self, model_names: Optional[List[str]] = None) -> pd.DataFrame:
        """Compare multiple models"""
        if model_names is None:
            model_names = list(self.results.keys())
        
        comparison = []
        for name in model_names:
            if name in self.results:
                r = self.results[name]
                comparison.append({
                    'Model': name,
                    'Accuracy': r['accuracy'],
                    'Precision': r['precision'],
                    'Recall': r['recall'],
                    'F1-Score': r['f1_score'],
                    'ROC-AUC': r['roc_auc'],
                    'PR-AUC': r['pr_auc'],
                    'Brier Score': r['brier_score'],
                    'Optimal Threshold': r['optimal_threshold']
                })
        
        return pd.DataFrame(comparison).round(4)
    
    def plot_roc_curves(self, model_names: Optional[List[str]] = None, 
                        save_path: Optional[str] = None, show: bool = False):
        """Plot ROC curves for multiple models"""
        if model_names is None:
            model_names = list(self.results.keys())
        
        plt.figure(figsize=(10, 8))
        
        for name in model_names:
            if name in self.results:
                roc = self.results[name]['roc_curve']
                auc_score = self.results[name]['roc_auc']
                plt.plot(roc['fpr'], roc['tpr'], label=f'{name} (AUC = {auc_score:.3f})', linewidth=2)
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.500)', alpha=0.5)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves Comparison', fontsize=14)
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"ROC curves saved to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_precision_recall_curves(self, model_names: Optional[List[str]] = None,
                                     save_path: Optional[str] = None, show: bool = False):
        """Plot Precision-Recall curves for multiple models"""
        if model_names is None:
            model_names = list(self.results.keys())
        
        plt.figure(figsize=(10, 8))
        
        for name in model_names:
            if name in self.results:
                pr = self.results[name]['pr_curve']
                pr_auc = self.results[name]['pr_auc']
                plt.plot(pr['recall'], pr['precision'], label=f'{name} (PR-AUC = {pr_auc:.3f})', linewidth=2)
        
        # Baseline (random classifier)
        baseline = self.results[model_names[0]] if model_names else None
        if baseline:
            pos_rate = baseline['classification_report']['1']['support'] / sum(
                baseline['classification_report'][str(c)]['support'] for c in [0, 1] if str(c) in baseline['classification_report']
            )
            plt.axhline(y=pos_rate, color='k', linestyle='--', label=f'Baseline (PR-AUC = {pos_rate:.3f})', alpha=0.5)
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curves Comparison', fontsize=14)
        plt.legend(loc='lower left', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"PR curves saved to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_confusion_matrix(self, model_name: str, 
                              save_path: Optional[str] = None, show: bool = False,
                              normalize: bool = False):
        """Plot confusion matrix for a model"""
        if model_name not in self.results:
            raise ValueError(f"Model {model_name} not found in results")
        
        cm = np.array(self.results[model_name]['confusion_matrix'])
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
            title = f'Normalized Confusion Matrix - {model_name}'
        else:
            fmt = 'd'
            title = f'Confusion Matrix - {model_name}'
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
                    xticklabels=['No Landslide', 'Landslide'],
                    yticklabels=['No Landslide', 'Landslide'],
                    cbar_kws={'label': 'Count' if not normalize else 'Proportion'})
        plt.title(title, fontsize=14)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_calibration_curve(self, model_names: Optional[List[str]] = None,
                               save_path: Optional[str] = None, show: bool = False):
        """Plot calibration curves for multiple models"""
        if model_names is None:
            model_names = list(self.results.keys())
        
        plt.figure(figsize=(10, 8))
        
        # Perfect calibration line
        plt.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated', alpha=0.5)
        
        for name in model_names:
            if name in self.results:
                cal = self.results[name]['calibration_curve']
                plt.plot(cal['prob_pred'], cal['prob_true'], 's-', label=name, linewidth=2, markersize=8)
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('Mean Predicted Probability', fontsize=12)
        plt.ylabel('Fraction of Positives', fontsize=12)
        plt.title('Calibration Curves', fontsize=14)
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Calibration curves saved to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_feature_importance(self, feature_importance: pd.Series, 
                                model_name: str = "Model",
                                top_n: int = 20,
                                save_path: Optional[str] = None, show: bool = False):
        """Plot feature importance"""
        top_features = feature_importance.head(top_n).sort_values(ascending=True)
        
        plt.figure(figsize=(10, max(6, top_n * 0.3)))
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_features)))
        bars = plt.barh(range(len(top_features)), top_features.values, color=colors)
        plt.yticks(range(len(top_features)), top_features.index, fontsize=10)
        plt.xlabel('Feature Importance', fontsize=12)
        plt.title(f'Top {top_n} Feature Importance - {model_name}', fontsize=14)
        plt.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, top_features.values)):
            plt.text(val + 0.001, i, f'{val:.4f}', va='center', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Feature importance plot saved to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_risk_distribution(self, y_prob: np.ndarray, 
                               save_path: Optional[str] = None, show: bool = False):
        """Plot distribution of predicted probabilities by true class"""
        plt.figure(figsize=(10, 6))
        
        # Separate by true class
        pos_probs = y_prob[y_prob > 0]  # placeholder
        # We need y_true for this - use results if available
        
        plt.hist(y_prob, bins=50, alpha=0.7, edgecolor='black', label='All Predictions')
        plt.axvline(x=self.thresholds['low_max'], color='green', linestyle='--', 
                    label=f'LOW/Moderate Threshold ({self.thresholds["low_max"]})')
        plt.axvline(x=self.thresholds['moderate_max'], color='orange', linestyle='--',
                    label=f'Moderate/HIGH Threshold ({self.thresholds["moderate_max"]})')
        plt.axvline(x=self.thresholds['high_min'], color='red', linestyle='--',
                    label=f'HIGH Threshold ({self.thresholds["high_min"]})')
        
        plt.xlabel('Predicted Probability', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('Distribution of Predicted Probabilities', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Risk distribution plot saved to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def generate_report(self, model_name: str, output_path: Optional[str] = None) -> str:
        """Generate text evaluation report"""
        if model_name not in self.results:
            raise ValueError(f"Model {model_name} not found in results")
        
        r = self.results[model_name]
        
        report = f"""
{'='*60}
MODEL EVALUATION REPORT: {model_name.upper()}
{'='*60}

PERFORMANCE METRICS:
  Accuracy:        {r['accuracy']:.4f}
  Precision:       {r['precision']:.4f}
  Recall:          {r['recall']:.4f}
  F1-Score:        {r['f1_score']:.4f}
  ROC-AUC:         {r['roc_auc']:.4f}
  PR-AUC:          {r['pr_auc']:.4f}
  Brier Score:     {r['brier_score']:.4f}
  Specificity:     {r['specificity']:.4f}
  NPV:             {r['npv']:.4f}

CONFUSION MATRIX:
  True Negatives:  {r['tn']}
  False Positives: {r['fp']}
  False Negatives: {r['fn']}
  True Positives:  {r['tp']}

THRESHOLD ANALYSIS:
  Optimal Threshold (Youden's J): {r['optimal_threshold']:.4f}
  Optimal Youden's J:             {r['optimal_youden_j']:.4f}

RISK CLASSIFICATION DISTRIBUTION:
  LOW Risk:       {r['risk_classification']['low_risk']['count']} samples
    Positive Rate: {r['risk_classification']['low_risk']['positive_rate']:.4f}
  MODERATE Risk:  {r['risk_classification']['moderate_risk']['count']} samples
    Positive Rate: {r['risk_classification']['moderate_risk']['positive_rate']:.4f}
  HIGH Risk:      {r['risk_classification']['high_risk']['count']} samples
    Positive Rate: {r['risk_classification']['high_risk']['positive_rate']:.4f}

CLASSIFICATION REPORT:
"""
        for label, metrics in r['classification_report'].items():
            if isinstance(metrics, dict):
                report += f"  Class {label}: Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}, F1={metrics['f1-score']:.4f}, Support={metrics['support']}\n"
        
        report += f"\n{'='*60}\n"
        
        if output_path:
            Path(output_path).write_text(report)
            logger.info(f"Evaluation report saved to {output_path}")
        
        return report
    
    def save_results(self, output_path: str):
        """Save all evaluation results to JSON"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy types to native Python types
        def convert(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj
        
        serializable = convert(self.results)
        
        with open(output_path, 'w') as f:
            json.dump(serializable, f, indent=2)
        
        logger.info(f"All evaluation results saved to {output_path}")
    
    def create_evaluation_plots(self, output_dir: str, model_names: Optional[List[str]] = None):
        """Create all evaluation plots"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if model_names is None:
            model_names = list(self.results.keys())
        
        for name in model_names:
            if name in self.results:
                self.plot_confusion_matrix(name, output_dir / f"{name}_confusion_matrix.png")
                self.plot_confusion_matrix(name, output_dir / f"{name}_confusion_matrix_norm.png", normalize=True)
        
        self.plot_roc_curves(model_names, output_dir / "roc_curves.png")
        self.plot_precision_recall_curves(model_names, output_dir / "pr_curves.png")
        self.plot_calibration_curve(model_names, output_dir / "calibration_curves.png")
        
        logger.info(f"All evaluation plots saved to {output_dir}")


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series, 
                   model_name: str = "model", 
                   output_dir: Optional[str] = None) -> Dict[str, Any]:
    """Convenience function to evaluate a single model"""
    evaluator = ModelEvaluator()
    
    y_prob = model.predict_proba(X_test)
    y_pred = model.predict(X_test)
    
    results = evaluator.evaluate(y_test.values, y_pred, y_prob, model_name)
    
    if output_dir:
        evaluator.create_evaluation_plots(output_dir, [model_name])
        evaluator.generate_report(model_name, Path(output_dir) / f"{model_name}_report.txt")
    
    return results