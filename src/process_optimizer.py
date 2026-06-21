import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution


PARAMS = [
    "cmp_pressure", "polish_time", "slurry_ph", "annealing_temp",
    "temp_gradient", "etch_depth", "vacuum_pressure",
    "pr_thickness_cv", "particle_count"
]

PARAM_BOUNDS_DEFAULT = {
    "cmp_pressure":   (80.0,  130.0),
    "polish_time":    (18.0,  65.0),
    "slurry_ph":      (4.0,   8.5),
    "annealing_temp": (1050.0,1190.0),
    "temp_gradient":  (0.2,   2.8),
    "etch_depth":     (410.0, 600.0),
    "vacuum_pressure":(5e-7,  1.8e-6),
    "pr_thickness_cv":(0.8,   4.2),
    "particle_count": (1.0,   18.0),
}

COST_PER_UNIT = {
    "cmp_pressure":   400,
    "polish_time":    800,
    "slurry_ph":      2500,
    "annealing_temp": 3000,
    "temp_gradient":  6000,
    "etch_depth":     250,
    "vacuum_pressure":5e7,
    "pr_thickness_cv":7000,
    "particle_count": 9000,
}


class ProcessOptimizer:
    """
    공정 파라미터 최적화 — Differential Evolution 기반
    Step 10의 ProcessCorrelationAnalyzer와 연동
    """

    def __init__(self, pca, baseline: dict,
                 monthly_wafers: int = 50_000,
                 wafer_value: float = 500.0,
                 param_bounds: dict = None):
        self.pca            = pca
        self.baseline       = baseline
        self.monthly_revenue= monthly_wafers * wafer_value
        self.param_bounds   = param_bounds or PARAM_BOUNDS_DEFAULT

    def adjustment_cost(self, params: dict) -> float:
        return sum(abs(params.get(p, self.baseline[p]) - self.baseline[p])
                   * COST_PER_UNIT[p] for p in PARAMS)

    def objective(self, params_array, defect_class,
                  w_defect=0.7, w_cost=0.2, w_constraint=0.1):
        param_dict = dict(zip(PARAMS, params_array))
        defect_score = self.pca.predict_defect_probability(param_dict, defect_class)
        cost_score   = min(self.adjustment_cost(param_dict) / 1_000_000, 1.0)
        constr = sum(
            max(0, lo - v) / (hi - lo) + max(0, v - hi) / (hi - lo)
            for (p, v), (lo, hi) in zip(param_dict.items(),
                                        [self.param_bounds[p] for p in PARAMS])
        )
        return w_defect * defect_score + w_cost * cost_score + w_constraint * min(constr, 1.0)

    def optimize(self, defect_class, maxiter=150, popsize=12) -> dict:
        bounds  = [self.param_bounds[p] for p in PARAMS]
        result  = differential_evolution(
            lambda x: self.objective(x, defect_class),
            bounds, seed=42, maxiter=maxiter, popsize=popsize,
            mutation=(0.5, 1.5), recombination=0.7, workers=1,
        )
        opt_params  = dict(zip(PARAMS, result.x))
        base_prob   = self.pca.predict_defect_probability(self.baseline, defect_class)
        opt_prob    = self.pca.predict_defect_probability(opt_params, defect_class)
        adj_cost    = self.adjustment_cost(opt_params)
        reduction   = (base_prob - opt_prob) / (base_prob + 1e-8)
        monthly_gain= reduction * base_prob * self.monthly_revenue
        roi         = (monthly_gain * 12 - adj_cost) / (adj_cost + 1) * 100

        return {
            "defect_class":   defect_class,
            "baseline_prob":  round(base_prob, 4),
            "optimal_prob":   round(opt_prob,  4),
            "prob_reduction_pct": round(reduction * 100, 2),
            "optimal_params": {p: round(v, 4) for p, v in opt_params.items()},
            "param_changes":  {p: round(opt_params[p] - self.baseline[p], 4)
                               for p in PARAMS},
            "adj_cost_usd":   round(adj_cost, 0),
            "monthly_gain_usd": round(monthly_gain, 0),
            "annual_gain_usd":  round(monthly_gain * 12, 0),
            "roi_pct":        round(roi, 1),
            "payback_months": round(adj_cost / (monthly_gain + 1), 2),
            "converged":      result.success,
        }

    def scenario_analysis(self, defect_class, param, n_steps=50) -> pd.DataFrame:
        """단일 파라미터 What-If 스캔"""
        lo, hi = self.param_bounds[param]
        rows = []
        for v in np.linspace(lo, hi, n_steps):
            p_dict = {**self.baseline, param: v}
            rows.append({
                "value": v,
                "defect_prob": self.pca.predict_defect_probability(p_dict, defect_class),
                "adj_cost": self.adjustment_cost(p_dict),
            })
        return pd.DataFrame(rows)