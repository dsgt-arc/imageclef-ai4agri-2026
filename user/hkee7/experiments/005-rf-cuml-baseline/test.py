from __future__ import annotations

import numpy as np

from metrics import exact_accuracy, mae, pm1_accuracy
from model import build_rf


def main() -> None:
    rng = np.random.default_rng(42)
    x = rng.normal(size=(300, 24)).astype(np.float32)
    y = np.clip((x[:, :3].sum(axis=1) + 3.0).round().astype(np.int32), 1, 5)

    backend = build_rf(
        use_cuml=False,
        n_estimators=40,
        max_depth=8,
        max_features="sqrt",
        random_state=42,
    )
    backend.model.fit(x, y)
    pred = np.asarray(backend.model.predict(x)).astype(np.int32)

    pm1 = pm1_accuracy(pred, y)
    exact = exact_accuracy(pred, y)
    err = mae(pred, y)

    print(f"backend={backend.name}")
    print(f"pm1_accuracy={pm1:.4f}")
    print(f"exact_accuracy={exact:.4f}")
    print(f"mae={err:.4f}")

    assert pm1 >= 0.95
    assert exact >= 0.60
    assert err <= 0.50


if __name__ == "__main__":
    main()

