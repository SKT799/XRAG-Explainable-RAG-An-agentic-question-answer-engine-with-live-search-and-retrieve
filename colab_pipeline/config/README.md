# config

One YAML file that holds every knob in the project. Change something here and the rest of the code picks it up.

## Files

`config.yaml` is the readable settings. Model IDs, thresholds, the chunk size, the cache TTL, all of it.

`settings.py` reads the YAML into a typed pydantic object. It also lets you override anything from the environment.

`__init__.py` makes the folder importable.

## Use it from code

```python
from config.settings import settings

print(settings.generator.model_id)
print(settings.attribution.threshold)
```

## Override without editing the file

Drop a `.env` file at the project root, or use environment variables. The prefix is `XRAG_` and nested keys use `__`:

```bash
export XRAG_GENERATOR__TEMPERATURE=0
export XRAG_ATTRIBUTION__THRESHOLD=0.8
```

PowerShell uses the same names with `$env:` instead.

## Check what is loaded

```bash
python config/settings.py
```

That prints the effective config as JSON. Useful when something looks off.

## Why one config

The same code is meant to run on a free Colab T4 and on a rented H100. Things like the quantization mode, the retrieval mode, and the trust threshold are the only differences between those two setups. Keeping them in one place means switching machines is a one-line edit.
