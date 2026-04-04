# Tasks: 003-presto-finetune

## 1. Setup
- [x] 1.1 Write experiment proposal
- [x] 1.2 Scaffold experiment files (pyproject.toml, config, dataset, model, train, test)
- [ ] 1.3 Install Presto from GitHub: `pip install git+https://github.com/nasaharvest/presto.git`
- [ ] 1.4 Run smoke-test: `python test.py --device cpu`
- [ ] 1.5 Verify Presto pretrained weights download successfully

## 2. Data Validation
- [ ] 2.1 Confirm band ordering in precomputed tensors (B2–B12, 10 bands)
- [ ] 2.2 Confirm day-of-year parsing from metadata.csv
- [ ] 2.3 Check that dataset shape matches model expectations `(T=34, C=10, H=128, W=128)`

## 3. Stage 1 — Frozen Encoder
- [ ] 3.1 Run Stage 1 (head-only, 10 epochs): `python train.py --stage 1`
- [ ] 3.2 Record ±1 accuracy, loss curve
- [ ] 3.3 Compare vs U-TAE ordinal baseline from 002

## 4. Stage 2 — Full Fine-tune
- [ ] 4.1 Launch Stage 2 from Stage 1 best checkpoint: `python train.py --stage 2 --ckpt artifacts/best_stage1.pt`
- [ ] 4.2 Record ±1 accuracy improvement over Stage 1
- [ ] 4.3 Compare total training time vs U-TAE

## 5. Analysis
- [ ] 5.1 Plot training curves (loss, ±1 acc) for both stages
- [ ] 5.2 Per-class confusion analysis
- [ ] 5.3 Document findings in results.md

## 6. Optional Extensions
- [ ] 6.1 Add lat/lon to dataset and test with Presto's geolocation encoding
- [ ] 6.2 Try larger spatial head (deeper / larger kernel)
- [ ] 6.3 Experiment with Presto band masking to handle temporal gaps
