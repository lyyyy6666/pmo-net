# Loss Audit

## Summary

The code implements the three manuscript loss components:

`L_total = L_pred + lambda_1 L_phy + lambda_2 L_smooth`

with:

- time-weighted prediction MSE,
- non-negativity penalty,
- temporal smoothness regularization.

The default hyperparameters match the manuscript values for `lambda_1` and `lambda_2`: `0.1` and `0.01`.

The training code also contains an optional observable dynamics consistency loss:

`+ lambda_observable_dyn * L_dyn`

This extra term is disabled by default (`lambda_observable_dyn=0.0`, `use_observable_dyn_loss=false`) and is not part of the manuscript main experiments.

## Actual Training Objective In Code

### GansuAir

Defined in `src/pmonet/experiments/train_wind_gansu.py:279-345`.

Actual total loss:

- `src/pmonet/experiments/train_wind_gansu.py:317`: `L_pred`
- `src/pmonet/experiments/train_wind_gansu.py:318`: `L_nonneg`
- `src/pmonet/experiments/train_wind_gansu.py:319`: `L_smooth`
- `src/pmonet/experiments/train_wind_gansu.py:321-330`: optional `L_dyn`
- `src/pmonet/experiments/train_wind_gansu.py:332`: total loss is `L_pred + lambda_nonnegative * L_nonneg + lambda_temporal_smooth * L_smooth + lambda_observable_dyn * L_dyn`

The training loop calls this criterion at `src/pmonet/experiments/train_wind_gansu.py:948-955`.

### KnowAir and Beijing

KnowAir defines `KnowAirWindLoss` at `src/pmonet/experiments/train_wind_knowair.py:385-460`.

Actual total loss:

- `src/pmonet/experiments/train_wind_knowair.py:419`: `L_pred`
- `src/pmonet/experiments/train_wind_knowair.py:420`: `L_nonneg`
- `src/pmonet/experiments/train_wind_knowair.py:421-424`: `L_smooth`
- `src/pmonet/experiments/train_wind_knowair.py:426-445`: optional `L_dyn`
- `src/pmonet/experiments/train_wind_knowair.py:447`: total loss includes optional `lambda_observable_dyn * L_dyn`

Beijing reuses `KnowAirWindLoss` through `src/pmonet/experiments/train_wind_beijing.py:377-411`.

## Component Details

### Time-Weighted Prediction Loss

Present.

- GansuAir: `src/pmonet/experiments/train_wind_gansu.py:301-302` creates `time_weights = exp(-0.05 * h)` and normalizes by `time_weights.sum()`.
- KnowAir/Beijing: `src/pmonet/experiments/train_wind_knowair.py:403-404` does the same.
- Prediction loss uses squared error in normalized target space:
  - GansuAir: `src/pmonet/experiments/train_wind_gansu.py:317`.
  - KnowAir/Beijing: `src/pmonet/experiments/train_wind_knowair.py:419`.

Implementation note: the manuscript states `w_h = exp(-beta h), beta = 0.05`. The code normalizes these exponential weights by their sum before applying them, so the relative horizon weighting matches the formula while keeping the aggregate loss scale stable.

### Non-Negativity Loss

Present.

- GansuAir: `F.relu(-output_real).mean()` at `src/pmonet/experiments/train_wind_gansu.py:318`.
- KnowAir/Beijing: `F.relu(-output_real).mean()` at `src/pmonet/experiments/train_wind_knowair.py:420`.

This matches the manuscript description of penalizing negative predictions.

### Temporal Smoothness Loss

Present.

- GansuAir: normalized-space adjacent-step squared difference at `src/pmonet/experiments/train_wind_gansu.py:319`.
- KnowAir/Beijing: normalized-space adjacent-step squared difference at `src/pmonet/experiments/train_wind_knowair.py:421-424`.

The manuscript should specify whether smoothness is computed in normalized or original scale. Current code uses normalized outputs.

## Hyperparameters

Defaults:

- `lambda_nonnegative = 0.1`
  - `src/pmonet/experiments/train_wind_gansu.py:386`
  - `src/pmonet/experiments/train_wind_knowair.py:511`
  - `src/pmonet/experiments/train_wind_beijing.py:167`
- `lambda_temporal_smooth = 0.01`
  - `src/pmonet/experiments/train_wind_gansu.py:387`
  - `src/pmonet/experiments/train_wind_knowair.py:512`
  - `src/pmonet/experiments/train_wind_beijing.py:168`
- `beta = 0.05`
  - hard-coded in `src/pmonet/experiments/train_wind_gansu.py:301`
  - hard-coded in `src/pmonet/experiments/train_wind_knowair.py:403`
- `lambda_observable_dyn = 0.0` by default
  - `src/pmonet/experiments/train_wind_gansu.py:385`
  - `src/pmonet/experiments/train_wind_knowair.py:498`
  - `src/pmonet/experiments/train_wind_beijing.py:156`

The YAML config files explicitly list these loss hyperparameters under `loss`.

## Extra Components Not Described In Manuscript

Optional extra component:

- Observable dynamics consistency loss, `L_dyn`.
- GansuAir class: `ObservableDynamicsConsistency`, `src/pmonet/experiments/train_wind_gansu.py:118-276`.
- KnowAir/Beijing class: `ObservableDynamicsConsistency`, `src/pmonet/experiments/train_wind_knowair.py:205-382`.
- Enabled only if `use_observable_dyn_loss=true` and `lambda_observable_dyn > 0.0`.
- It may include diffusion-decay or advection-diffusion residuals depending on `observable_dyn_mode`.

No evidence found in release code for:

- adversarial loss,
- spectral loss,
- VMD-related loss,
- graph regularization loss,
- separate wind consistency loss beyond optional observable dynamics residual,
- auxiliary task loss.

## Match To Manuscript

Matches the confirmed manuscript training objective under the official release scripts:

- `use_observable_dyn_loss=false`,
- `lambda_observable_dyn=0.0`,
- `lambda_nonnegative=0.1`,
- `lambda_temporal_smooth=0.01`,
- `beta=0.05`.

Implementation detail: the code normalizes the exponential time weights by their sum. Smoothness is computed in normalized output space.

Not fully identical if:

- the manuscript formula implies unnormalized `w_h = exp(-beta h)`;
- a run enables `L_dyn`, because the manuscript formula does not include `lambda_observable_dyn L_dyn`;
- the manuscript describes `L_phy` only as non-negativity, while the code also contains an optional physics residual loss under a separate flag.

## Recommended Action

Recommended action: keep the release code and official scripts as-is for the confirmed paper objective.

- Keep `L_dyn` as optional disabled code for diagnostics/experimentation.
- Keep official scripts with `--use_observable_dyn_loss false --lambda_observable_dyn 0.0`.
- In the manuscript or reproducibility notes, mention that `w_h = exp(-beta h)` is normalized across the prediction horizon in the implementation.
