# Image benchmark presets

Each JSON file launches directly with:

```bash
python examples/flow_matching_image_benchmark_runner.py --config <file>
```

Preset files may use `extends` only to avoid duplicating common training and
evaluation fields. The runner resolves inheritance before validation and saves
the complete resolved JSON in every session.

The pinned DiffuseNNX `vae_trial1.pkl` is not publicly released as of the
specification date. VAE presets therefore contain a 64-zero placeholder for
`expected_sha256`. Replace it with a trusted, independently obtained checksum
before running; the runner intentionally rejects a checkpoint that does not
match it.
