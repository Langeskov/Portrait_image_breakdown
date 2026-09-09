# Local model weights

Place runtime model checkpoints in this directory.

The default pose detector is `yolo26x-pose.pt`:

```text
model/
└── yolo26x-pose.pt
```

Supported built-in YOLO26 pose checkpoints are:

- `yolo26n-pose.pt`
- `yolo26s-pose.pt`
- `yolo26m-pose.pt`
- `yolo26l-pose.pt`
- `yolo26x-pose.pt`

Model files are intentionally ignored by Git because they are large binary assets. Ultralytics provides the official pretrained YOLO26 pose weights and supports all five scales. citeturn790269search1turn790269search6

For development, the active model can also be selected with the `PIB_POSE_MODEL` environment variable. Built-in values are `n`, `s`, `m`, `l`, and `x`; a local checkpoint path may also be supplied.
