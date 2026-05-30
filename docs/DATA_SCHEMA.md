# Data Schema

The public code starts from an anonymized, model-ready knee-level CSV. The
default path is configured in `config.py`:

```text
data/model_ready_knees.csv
```

Each row represents one knee.

## Required Identifier Columns

```text
participant_id
knee_id
side
```

- `participant_id`: anonymized participant identifier. Used for
  participant-level train/validation/test splitting and clustered bootstrap.
- `knee_id`: anonymized knee identifier. Must be unique within participant.
- `side`: optional descriptive side label, for example `left` or `right`.

## Required Label Columns

```text
incidence_label
incidence_mask
progression_label
progression_mask
exclusion
```

Expected task coding:

- `incidence_mask = 1`: this knee is eligible for the incidence task.
- `progression_mask = 1`: this knee is eligible for the progression task.
- exactly one of `incidence_mask` and `progression_mask` should be active for
  model-ready rows.
- label columns should be binary where the corresponding mask is active.
- excluded or indeterminate knees should not be included in the model-ready CSV.

## Required Clinical Columns

```text
bl_kl
m30_kl
bl_pfoa
m30_pfoa
```

- `bl_kl`: baseline tibiofemoral KL grade.
- `m30_kl`: 30-month tibiofemoral KL grade.
- `bl_pfoa`: baseline patellofemoral OA marker.
- `m30_pfoa`: 30-month patellofemoral OA marker.

Values may be numeric or categorical as long as they can be parsed by pandas and
used in the benchmark logistic models.

## Required Image Path Columns

```text
bl_pa_path
bl_lat_path
m30_pa_path
m30_lat_path
```

Paths can be absolute or relative. Relative paths are resolved against the
repository root.

Recommended image preparation:

- de-identify all image files before use;
- orient left and right knees consistently;
- crop around the knee region before model training;
- use a consistent image format such as PNG or JPEG.

## Example Header

```csv
participant_id,knee_id,side,incidence_label,incidence_mask,progression_label,progression_mask,exclusion,bl_kl,m30_kl,bl_pfoa,m30_pfoa,bl_pa_path,bl_lat_path,m30_pa_path,m30_lat_path
```
