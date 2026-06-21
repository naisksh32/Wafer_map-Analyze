import json
import numpy as np
from pathlib import Path


class DefectAnalyzer:
    """불량 메타데이터 기반 분석 클래스 — Step 10~12 공통 사용"""

    def __init__(self, metadata_path):
        path = Path(metadata_path)
        if not path.exists():
            raise FileNotFoundError(f"메타데이터 파일 없음: {path}")
        with open(path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        self.classes = list(self.metadata.keys())

    def get_mechanism(self, defect_class: str) -> str:
        self._validate(defect_class)
        return self.metadata[defect_class]['physical_mechanism']

    def get_process_stage(self, defect_class: str) -> str:
        self._validate(defect_class)
        return self.metadata[defect_class]['primary_process_stage']

    def get_critical_parameters(self, defect_class: str) -> list:
        self._validate(defect_class)
        return self.metadata[defect_class]['critical_parameters']

    def get_severity(self, defect_class: str) -> str:
        self._validate(defect_class)
        return self.metadata[defect_class]['severity_level']

    def get_severity_score(self, defect_class: str) -> int:
        self._validate(defect_class)
        return self.metadata[defect_class]['severity_score']

    def get_remediation(self, defect_class: str) -> str:
        self._validate(defect_class)
        return self.metadata[defect_class]['remediation']

    def predict_yield_loss(self, defect_class: str, defect_rate: float) -> float:
        """불량 발생률 × 수율 영향도 → 예상 수율 손실"""
        self._validate(defect_class)
        yield_impact = self.metadata[defect_class]['yield_impact']
        return float(defect_rate * yield_impact)

    def get_roi_estimate(self, defect_class: str) -> float:
        self._validate(defect_class)
        return self.metadata[defect_class]['estimated_roi_pct']

    def prioritize_defects(self, detected_classes: list) -> list:
        """검출된 불량 리스트를 심각도 기준으로 정렬"""
        scored = [
            (cls, self.get_severity_score(cls))
            for cls in detected_classes
            if cls in self.metadata and cls != 'none'
        ]
        return [cls for cls, _ in sorted(scored, key=lambda x: x[1], reverse=True)]

    def generate_report(self, defect_class: str, defect_rate: float = 0.0) -> dict:
        """단일 불량에 대한 분석 리포트 딕셔너리 반환"""
        self._validate(defect_class)
        meta = self.metadata[defect_class]
        return {
            'defect_class':       defect_class,
            'korean_name':        meta.get('korean_name', ''),
            'physical_mechanism': meta['physical_mechanism'],
            'process_stage':      meta['primary_process_stage'],
            'severity':           meta['severity_level'],
            'severity_score':     meta['severity_score'],
            'yield_loss_pct':     self.predict_yield_loss(defect_class, defect_rate) * 100,
            'estimated_roi_pct':  meta['estimated_roi_pct'],
            'critical_parameters': meta['critical_parameters'],
            'remediation':        meta['remediation'],
            'spatial_evidence':   meta.get('spatial_evidence', {})
        }

    def batch_report(self, predictions: list, defect_rates: dict = None) -> list:
        """배치 예측 결과 → 리포트 리스트 (우선순위 정렬)"""
        unique = list(set(predictions))
        prioritized = self.prioritize_defects(unique)
        reports = []
        for cls in prioritized:
            rate = defect_rates.get(cls, 0.0) if defect_rates else 0.0
            reports.append(self.generate_report(cls, rate))
        return reports

    def _validate(self, defect_class: str):
        if defect_class not in self.metadata:
            raise ValueError(f"알 수 없는 불량 클래스: {defect_class}. 유효: {self.classes}")