# UCP
Utility-constrained Policies

This implementation builds on research code from [meta-pytorch/LeanRL](https://github.com/meta-pytorch/LeanRL).

## Environment Setup
- `conda env create -f conda-recipe.yaml`
- `conda activate UCP`

## Running the Code
- Launch training with `python ucp.py` plus any CLI flags.
- Environment-specific YAML config files in `configs/` can provide defaults. CLI arguments override YAML values, and missing config files fall back to the dataclass defaults in `ucp.py`.

## `ucp.py`
- Implements the current UCP training loop with distributional reward and cost critics, SAC-style policy updates, and a Lagrangian safety constraint.
- Uses TorchRL's `MultiStepTransform` for configurable n-step returns.
- Supports cost-stock observation augmentation through `AugmentedObservation`.
- Supports sampled initial cost stocks with a binned contextual Lagrangian, where each cost-budget bin has its own learned multiplier.
- Supports TensorBoard/W&B logging, periodic checkpoint saving, evaluation, `torch.compile`, and CUDA graph capture.

Example:
```bash
python ucp.py --env-id SafetyCarGoal1-v0 # UCP
python ucp.py --env-id SafetyCarGoal1-v0 --no-add-cost-stock # UCP-NA 
```


## License

This project includes code derived from LeanRL:
https://github.com/meta-pytorch/LeanRL

Specifically:
- sac_continuous_action_torchcompile.py

LeanRL is licensed under the MIT License.

Modifications and additional code:
Copyright 2026 Mehrdad Moghimi
Licensed under the Apache License, Version 2.0.