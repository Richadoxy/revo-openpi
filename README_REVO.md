# revo-openpi training notes

This fork adds a Revo3 bimanual fine-tuning path for `pi05_base`.

## Dataset

Default config:

```text
/home/xyd/datasets/original-revomate_revo3_pick_and_place/original/lerobot_v21/revomate_revo3_mit_3cam_test
```

The dataset is LeRobot v2.1 and stores 70D `observation.state` / `action` vectors:

```text
0:7    left arm pose
7:14   right arm pose
14:21  left arm joints
21:28  right arm joints
28:49  left Revo3 hand joints
49:70  right Revo3 hand joints
```

`openpi.policies.revo_policy.RevoInputs` drops the first 14 pose dimensions, so the model trains on 56D joint actions:

```text
0:7    left arm joints
7:14   right arm joints
14:35  left Revo3 hand joints
35:56  right Revo3 hand joints
```

## Config

Training config:

```text
pi05_revo_revo3_56d
```

It creates `Pi0Config(pi05=True, action_dim=56, action_horizon=50, max_token_len=256)` and loads
`pi05_base` with `PartialCheckpointWeightLoader`. Shape-compatible pretrained weights are loaded, while
action projection layers that depend on the 32D upstream action size are kept randomly initialized.

## Commands

```bash
uv run scripts/compute_norm_stats.py --config-name pi05_revo_revo3_56d

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py pi05_revo_revo3_56d \
  --exp-name=revo3_pick_place_test \
  --overwrite
```
