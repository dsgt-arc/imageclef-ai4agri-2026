# Tasks

## 1. Setup
- [x] 1.1 Vendor U-TAE source code (vendor/utae-paps)
- [x] 1.2 Create experiment structure and pyproject.toml
- [x] 1.3 Implement dataset adapter, model, metrics, training script
- [ ] 1.4 Install dependencies and verify imports on cluster

## 2. Data Validation
- [ ] 2.1 Verify band ordering matches constants (B2–B12, 10 bands)
- [ ] 2.2 Verify day-of-year parsing from metadata.csv
- [ ] 2.3 Check label distribution for class imbalance

## 3. Training
- [ ] 3.1 Run regression baseline (smooth L1, 50 epochs)
- [ ] 3.2 Run classification baseline (CE, 50 epochs) for comparison
- [ ] 3.3 Collect training curves and validation metrics

## 4. Analysis
- [ ] 4.1 Compare regression vs. classification on ±1 accuracy
- [ ] 4.2 Analyse per-class performance and confusion patterns
- [ ] 4.3 Document findings in results.md
