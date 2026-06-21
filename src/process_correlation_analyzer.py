import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import pearsonr, spearmanr


CRITICAL_THRESHOLD = 0.3


class ProcessCorrelationAnalyzer:
    """공정 파라미터-불량 클래스 상관관계 분석기 — Step 10/12 공통 사용"""

    def __init__(self, process_df: pd.DataFrame, param_cols: list):
        self.df     = process_df.copy()
        self.params = param_cols

    def correlation_table(self, defect_class: str) -> pd.DataFrame:
        """특정 불량 클래스 vs 공정 파라미터 Pearson/Spearman 상관계수"""
        target = (self.df['defect_class'] == defect_class).astype(float)
        rows = []
        for p in self.params:
            r_p, pval_p = pearsonr(self.df[p], target)
            r_s, pval_s = spearmanr(self.df[p], target)
            rows.append({
                'parameter':  p,
                'pearson_r':  round(r_p, 4),
                'pearson_p':  round(pval_p, 4),
                'spearman_r': round(r_s, 4),
                'spearman_p': round(pval_s, 4),
                'abs_pearson': round(abs(r_p), 4),
                'significant': 'Yes' if pval_p < 0.05 else 'No',
            })
        return pd.DataFrame(rows).sort_values('abs_pearson', ascending=False)

    def critical_parameters(self, defect_class: str,
                             threshold: float = CRITICAL_THRESHOLD) -> pd.DataFrame:
        """임계값 이상 상관 + 유의미한 파라미터만 반환"""
        tbl = self.correlation_table(defect_class)
        return tbl[
            (tbl['abs_pearson'] >= threshold) &
            (tbl['significant'] == 'Yes')
        ].copy()

    def pearson_matrix(self) -> pd.DataFrame:
        """전체 불량 클래스 × 공정 파라미터 Pearson r 매트릭스"""
        defect_classes = [c for c in self.df['defect_class'].unique() if c != 'none']
        return pd.DataFrame(
            {cls: self.correlation_table(cls).set_index('parameter')['pearson_r']
             for cls in defect_classes}
        ).T

    def predict_defect_probability(self, param_values: dict,
                                    defect_class: str) -> float:
        """
        공정 파라미터 값 → 해당 불량 발생 상대 위험도 (단순 선형 스코어링).
        실제 배포 시 Step 12 Bayesian Opt 모델로 교체.
        """
        tbl = self.correlation_table(defect_class)
        crit = tbl[tbl['abs_pearson'] >= CRITICAL_THRESHOLD]
        if crit.empty:
            return 0.0

        score = 0.0
        for _, row in crit.iterrows():
            p = row['parameter']
            if p not in param_values:
                continue
            baseline = self.df[p].mean()
            sigma    = self.df[p].std() + 1e-10
            z = (param_values[p] - baseline) / sigma
            score += row['pearson_r'] * z

        # 시그모이드로 [0,1] 범위 변환
        return float(1 / (1 + np.exp(-score)))

    def generate_report(self, output_path: str = None) -> dict:
        """전체 불량 클래스 상관관계 요약 리포트"""
        defect_classes = [c for c in self.df['defect_class'].unique() if c != 'none']
        report = {}
        for cls in defect_classes:
            crit = self.critical_parameters(cls)
            report[cls] = {
                'n_critical': len(crit),
                'critical_params': crit['parameter'].tolist(),
                'top_pearson_r':   crit['pearson_r'].tolist(),
                'top_param': crit['parameter'].iloc[0] if len(crit) > 0 else None,
                'top_r': float(crit['pearson_r'].iloc[0]) if len(crit) > 0 else 0.0,
            }
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
        return report