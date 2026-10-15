# Model weights go here

Nothing in this folder is committed. Either copy the weights here or point
`NNUNET_MODEL_DIR` at wherever they live.

The study's weights are inside the validated result archives
(`MU_Glioma_no29_nnunet_fold1_seed2026_reports.zip` … `no33`), under `model/`:

```
model/checkpoint_best.pth
model/plans.json
model/dataset.json
```

Extract the `model/` folder of each run you want to use:

```
models/
  no29/model/{checkpoint_best.pth, plans.json, dataset.json}     one model
  no30/model/…                                                   add more for an ensemble
```

- One run folder → that single model is used.
- Several run folders → their softmax probabilities are averaged.

Each run is an independent nnU-Net with its own `plans.json`; they are not
folds of one model. See `docs/MODEL_CONTRACT.md` §4–§5 for what that means for
interpreting results.

Check what the app will see:

```
python backend/scripts/inspect_contract.py --model-dir models
```
