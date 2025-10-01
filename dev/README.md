# Local Development Resources

This folder contains everything you need for local OpenClio development and testing.

## Quick Start

```bash
# From the repo root
cd dev
./setup_dev.sh
```

## Files

- **`QUICKSTART.md`** - 5-minute getting started guide
- **`LOCAL_SETUP.md`** - Comprehensive local development guide
- **`setup_dev.sh`** - One-command setup script
- **`test_local.py`** - Automated tests for development

## Workflow

1. **Install**: Run `./setup_dev.sh` from repo root
2. **Authenticate**: `gcloud auth application-default login`
3. **Test**: `python dev/test_local.py`
4. **Develop**: Edit code in `openclio/`, changes are immediately reflected
5. **Notebook**: `jupyter notebook notebook.ipynb` (from repo root)

## Local Development

The package is installed in **editable mode** (`pip install -e .`), so:
- Code changes are immediately available
- No need to reinstall after edits
- Just restart your Python kernel/process

## Testing Loop

```bash
# 1. Make changes to openclio/*.py
# 2. Run quick tests
python dev/test_local.py

# 3. Or test in notebook
jupyter notebook notebook.ipynb
```

## Clearing Cache

Force recomputation by deleting checkpoint files:

```bash
# Clear all cached results
rm -rf ./output_*/*.pkl

# Or specific stages
rm ./output_*/facetValues.pkl  # Re-extract facets
rm ./output_*/baseClusters.pkl  # Re-cluster
```

## Tips

- Use small datasets (5-10 items) for fast iteration
- Set `llmBatchSize=5` or lower
- Use `gemini-1.5-flash-002` (faster, cheaper)
- Enable `verbose=True` to see progress
- Check `./output_*/` for cached intermediate results

## Resources

See the main **README.md** for full documentation.
