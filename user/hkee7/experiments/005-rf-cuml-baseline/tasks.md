# Tasks: 005-rf-cuml-baseline

## 1. Setup
- [x] 1.1 Write experiment proposal
- [x] 1.2 Scaffold config, dataset, metrics, model, train, and test scripts
- [ ] 1.3 Sync dependencies for this experiment (`uv sync --package 005-rf-cuml-baseline`)
- [ ] 1.4 Run smoke test (`python test.py`)

## 2. Baseline Training
- [ ] 2.1 Train RF baseline on sampled train pixels (`python train.py`)
- [ ] 2.2 Record validation metrics (±1, exact, MAE)
- [ ] 2.3 Check OOB score (when backend supports it)

## 3. Feature Importance Review
- [ ] 3.1 Export ranked importances
- [ ] 3.2 Identify dominant timesteps and bands
- [ ] 3.3 Compare importance trends against expected agronomic seasonality

## 4. Follow-up
- [ ] 4.1 Compare RF baseline against experiment 002/003 validation results
- [ ] 4.2 Decide whether architecture changes are still justified
- [ ] 4.3 Document findings in `results.md`

