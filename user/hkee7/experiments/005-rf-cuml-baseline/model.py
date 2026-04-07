from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class RFBackend:
    name: str
    model: Any


def build_rf(
    *,
    use_cuml: bool,
    n_estimators: int,
    max_depth: int | None,
    max_features: str,
    random_state: int,
) -> RFBackend:
    if use_cuml:
        try:
            from cuml.ensemble import RandomForestClassifier as CumlRF

            model = CumlRF(
                n_estimators=n_estimators,
                max_depth=max_depth if max_depth is not None else 32,
                max_features=max_features,
                random_state=random_state,
            )
            return RFBackend(name="cuml", model=model)
        except Exception:
            pass

    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_features=max_features,
        random_state=random_state,
        n_jobs=-1,
        bootstrap=True,
        oob_score=True,
    )
    return RFBackend(name="sklearn", model=model)


