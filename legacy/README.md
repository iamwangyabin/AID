# Legacy Artifacts

This directory keeps code and configs that are not wired into the current AID
runtime, but may still be useful as references when restoring older methods or
dataset recipes.

- `data/`: old dataset loaders with signatures that do not match the current
  `train.py` / `test.py` dataset construction path.
- `tools/`: one-off dataset recipe notebooks that are not part of the documented
  Arrow dataset generation flow.
- `networks/`: standalone upstream scripts that are not imported by the current
  evaluation functions.
- `cfgs/`: configs for methods that are not registered in the current model
  factory.
