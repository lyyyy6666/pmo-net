# Model Audit

## Summary

The release code contains a real wind-aware branch in the final model implementation. Wind-aware transport is active when `use_wind_advection=True` and `meteo_future` is passed into the model.

Current release status: the official GansuAir and KnowAir shell wrappers explicitly pass `--use_wind_advection true`. Beijing remains disabled by default because the release does not assume a confirmed u/v wind input for that workflow.

## Wind-Aware Branch

Wind-aware branch: present.

Key implementation:

- `src/pmonet/models/pmonet.py:28`: `WindAwarePhysicalODEFunc`.
- `src/pmonet/models/pmonet.py:53-55`: constructor accepts `wind_u_idx`, `wind_v_idx`, and `use_wind_advection`.
- `src/pmonet/models/pmonet.py:176-203`: `build_wind_graph()` builds a directed wind graph from `meteo_t`.
- `src/pmonet/models/pmonet.py:195-200`: wind vector is formed from selected u/v channels, projected onto pairwise station directions, distance-weighted, and masked by graph topology.
- `src/pmonet/models/pmonet.py:215-219`: advection is computed as `k_adv * (inflow - outflow)`.
- `src/pmonet/models/pmonet.py:290-297`: physical derivative combines diffusion, wind advection, reaction MLP, and decay.

The wind graph and advection term participate in the computation graph with respect to model state and learnable `k_adv`. The meteorology tensor is input data, not a learnable parameter.

## Forward Interface

The final model accepts meteorology through:

- `src/pmonet/models/pmonet.py:581-587`: `solve_fused_ode(..., meteo_future=None, ...)`.
- `src/pmonet/models/pmonet.py:593-607`: validates `meteo_future` shape when wind advection is enabled.
- `src/pmonet/models/pmonet.py:623-625`: passes `meteo_future[:, step_idx - 1]` into the physical ODE at each prediction step.
- `src/pmonet/models/pmonet.py:670-677`: `forecast(..., meteo_future=None, ...)`.
- `src/pmonet/models/pmonet.py:782-804`: `forward(..., meteo_future=None, ...)` forwards it into `forecast`.

Expected shape is `(B, pred_len, N, meteo_dim)`.

## Data Flow

### GansuAir

- `src/pmonet/data/gansu.py:142-146`: dataloader accepts meteorology sidecar files.
- `src/pmonet/data/gansu.py:195-221`: loads raw meteorology, validates time/station alignment, and creates normalized meteorology.
- `src/pmonet/data/gansu.py:268-281`: returns future normalized and raw meteorology when `return_meteo_pair=True`.
- `src/pmonet/data/gansu.py:338-351`: collate function returns both `meteo_future_batch` and `meteo_future_raw_batch`.
- `src/pmonet/experiments/train_wind_gansu.py:463-475`: resolves `u10` and `v10` indices from meteo column metadata.
- `src/pmonet/experiments/train_wind_gansu.py:510-529`: enables `return_meteo_pair` when wind advection or observable dynamics loss is requested.
- `src/pmonet/experiments/train_wind_gansu.py:694-698`: passes wind indices and `use_wind_advection` into the model.
- `src/pmonet/experiments/train_wind_gansu.py:829-834` and `941-946`: passes `meteo_future_raw` to `model.forecast()` when `args.use_wind_advection` is true.

### KnowAir

- `src/pmonet/data/knowair.py:139-140`: dataloader accepts raw wind feature indices.
- `src/pmonet/data/knowair.py:203-214`: removes target from meteorology columns and maps raw wind indices to meteorology positions.
- `src/pmonet/data/knowair.py:300-316`: returns normalized and raw future meteorology.
- `src/pmonet/experiments/train_wind_knowair.py:602-630`: prepares dataset with resolved wind indices and metadata.
- `src/pmonet/experiments/train_wind_knowair.py:701-743`: builds model with wind indices and `use_wind_advection`.
- `src/pmonet/experiments/train_wind_knowair.py:875-891`: passes `meteo_future_raw` to `model.forecast()` when wind advection is true.

KnowAir includes additional safety checks: wind usage requires local metadata confirmation unless columns can be resolved safely.

### Beijing

- `src/pmonet/data/beijing.py` reads meteorology columns and maps `wind_u_idx`/`wind_v_idx`.
- `src/pmonet/experiments/train_wind_beijing.py:154-158`: CLI exposes wind advection and wind indices.
- `src/pmonet/experiments/train_wind_beijing.py:331-373`: builds the same wind-capable model.
- `src/pmonet/experiments/train_wind_beijing.py:508-523`: passes raw future meteorology to the model when wind advection is true.

`configs/beijing.yaml` currently sets `use_wind_advection: false`.

## Physical Branch Composition

Actual physical branch in `WindAwarePhysicalODEFunc`:

- Diffusion: normalized graph Laplacian, `src/pmonet/models/pmonet.py:135-141` and `290-291`.
- Wind-aware transport: directed graph from wind projection, `src/pmonet/models/pmonet.py:176-220`.
- Reaction/residual physical MLP: `reaction_net`, `src/pmonet/models/pmonet.py:106-110` and `294`.
- Decay: learnable positive `gamma`, `src/pmonet/models/pmonet.py:102-104`, `162-167`, and `295`.

## Data-Driven Branch

The data-driven residual branch exists:

- `src/pmonet/models/pmonet.py:313-334`: `DataDrivenODEFunc` concatenates latent state with historical context and applies an MLP.
- `src/pmonet/models/pmonet.py:629-632`: data-driven derivative is computed at each ODE step when `use_unk_ode=True`.

## Adaptive Gating

Adaptive gating exists:

- `src/pmonet/models/pmonet.py:337-359`: `AdaptiveGating` computes sigmoid gates from latent state.
- `src/pmonet/models/pmonet.py:511-518`: gate is instantiated when both ODE branches and gating are enabled.
- `src/pmonet/models/pmonet.py:613-621`: branch weights are selected per step.
- `src/pmonet/models/pmonet.py:636`: fused derivative is `alpha_t * dz_phy + (1.0 - alpha_t) * dz_data`.

## Agreement With Manuscript Method Claims

Supported by code:

- diffusion: yes.
- wind-aware transport: yes, when `use_wind_advection=True` and `meteo_future` is provided.
- decay: yes.
- data-driven residual dynamics: yes.
- adaptive gating: yes.

Runtime consistency notes:

- The Python entry point defaults still define `--use_wind_advection false`, but the official GansuAir and KnowAir shell scripts override this with `--use_wind_advection true`.
- YAML configs document intended settings but are not automatically consumed by the training scripts.
- Beijing AirDualODE-style entry explicitly disables latent wind advection because its wind features are not u/v components.

## Recommended Action

For the manuscript main experiments, use the official GansuAir and KnowAir scripts or pass `--use_wind_advection true` manually. Do not claim wind-aware transport is active in a run unless the command includes `--use_wind_advection true` and the dataloader provides confirmed u/v meteorology.
