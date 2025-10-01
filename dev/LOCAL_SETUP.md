# Local Development Setup

Quick guide to get OpenClio running locally for development and testing.

## Prerequisites

- Python 3.8+
- GCP account with Vertex AI enabled
- gcloud CLI installed

## Setup Steps

### 1. Clone and Install

```bash
cd /Users/george/git/openclio
pip install -e .
```

The `-e` flag installs in "editable" mode, so code changes are immediately reflected.

### 2. Authenticate with GCP

```bash
gcloud auth application-default login
```

This creates credentials at `~/.config/gcloud/application_default_credentials.json`

### 3. Set Your Project ID

In the notebook or your Python code:
```python
PROJECT_ID = "your-gcp-project-id"  # Replace with actual project
```

### 4. Verify Installation

```python
import openclio
from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

# Test with minimal data
results = openclio.runClio(
    facetSchema=openclio.SystemPromptFacets,
    facetMetadata=openclio.systemPromptFacetMetadata,
    embeddingModel=embedding_model,
    data=["Test prompt"],
    outputDirectory="./test_output",
    project_id=PROJECT_ID,
    model_name="gemini-1.5-flash-002",
    verbose=True,
)
```

## Development Workflow

### Running the Notebook

```bash
jupyter notebook example_system_prompts.ipynb
```

### Quick Testing Loop

1. Make code changes in `openclio/openclio.py` or other source files
2. Run a cell in the notebook (no need to reinstall!)
3. Check results
4. Iterate

### Clearing Cached Results

To force recomputation:
```bash
rm -rf ./output_*/*.pkl
```

Or delete specific checkpoint files:
```bash
rm ./output_music/facetValues.pkl  # Force re-extraction
```

## Common Issues

### ImportError: No module named 'openclio'

Make sure you're in the right directory and installed with `-e`:
```bash
cd /Users/george/git/openclio
pip install -e .
```

### Authentication Failed

Re-authenticate:
```bash
gcloud auth application-default login
```

Verify your project:
```bash
gcloud config get-value project
```

### Quota Exceeded

Lower the batch size:
```python
llmBatchSize=5  # or even lower
```

Or wait a minute between batches (Vertex AI limits ~60 req/min).

### Widget Not Displaying

Make sure you're in a Jupyter environment:
```bash
pip install ipywidgets jupyter
jupyter nbextension enable --py widgetsnbextension
```

## Testing Changes

### Minimal Test Dataset

Use 5-10 items for fast iteration:
```python
test_data = ["prompt 1", "prompt 2", "prompt 3", "prompt 4", "prompt 5"]
```

### Check Each Stage

Results are cached, so you can inspect intermediate outputs:
```python
import cloudpickle

# Check extracted facets
with open('./output/facetValues.pkl', 'rb') as f:
    facet_values = cloudpickle.load(f)
    print(facet_values[0].facetValues)
```

### Debug Mode

Set verbose to see what's happening:
```python
results = openclio.runClio(..., verbose=True)
```

## Performance Tips

- **Model**: Use `gemini-1.5-flash-002` for development (faster, cheaper)
- **Batch Size**: Keep `llmBatchSize=5` for local testing
- **Data Size**: Start with 5-10 items, scale up once it works
- **Checkpointing**: Leverage cached results - only delete what you need to recompute

## Useful Commands

```bash
# Watch for changes and run tests
watch -n 5 python -c "import openclio; print('OK')"

# Clean all output directories
rm -rf ./output_*/

# Check Vertex AI quota
gcloud alpha services quota describe \
    aiplatform.googleapis.com/generate_content_requests_per_minute_per_region \
    --project=YOUR_PROJECT_ID

# Monitor API calls
gcloud logging read "resource.type=aiplatform.googleapis.com" --limit 10
```

## Next Steps

Once local testing works:
1. Test with larger datasets (100-1000 items)
2. Experiment with custom facets
3. Try different Gemini models (Flash vs Pro)
4. Explore the widget interactively
