# OpenClio Quickstart

## Installation (30 seconds)

```bash
./setup_dev.sh
```

Or manually:
```bash
pip install -e .
gcloud auth application-default login
```

## Test It Works (1 minute)

```bash
python test_local.py
```

## Run the Notebook (5 minutes)

```bash
jupyter notebook example_system_prompts.ipynb
```

Update the `PROJECT_ID` in the second cell, then run all cells.

## Your First Custom Facets (10 minutes)

```python
from pydantic import BaseModel, Field
import openclio
from sentence_transformers import SentenceTransformer

# Define what to extract
class MyFacets(BaseModel):
    topic: str = Field(description="Main topic in 1-2 sentences")
    sentiment: str = Field(description="Positive, negative, or neutral")

# Which facets get clustering
facet_metadata = {
    "topic": openclio.FacetMetadata(
        name="Topic",
        summaryCriteria="Cluster by topic area"
    )
}

# Your data
data = [
    "This product is amazing! Best purchase ever.",
    "Terrible experience, would not recommend.",
    "The weather forecast for tomorrow looks good.",
]

# Run it
embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
results = openclio.runClio(
    facetSchema=MyFacets,
    facetMetadata=facet_metadata,
    embeddingModel=embedding_model,
    data=data,
    outputDirectory="./output",
    project_id="your-project-id",
    model_name="gemini-1.5-flash-002",
    displayWidget=True,
)
```

## Key Concepts

**Multi-facet extraction**: All facets extracted in ONE LLM call per text
- 1000 texts × 4 facets = 1000 API calls (not 4000!)
- Much faster and cheaper

**Pydantic schema**: Define facets as a Python class
```python
class MusicFacets(BaseModel):
    genre: str = Field(description="...")
    tone: str = Field(description="...")
```

**Facet metadata**: Control which facets get cluster hierarchies
```python
facet_metadata = {
    "genre": FacetMetadata(name="Genre", summaryCriteria="..."),
    # Other facets won't get hierarchies
}
```

## Development Tips

**Fast iteration**:
1. Small dataset (5-10 items)
2. `llmBatchSize=5`
3. `gemini-1.5-flash-002`
4. Delete `./output/*.pkl` to force recomputation

**Debug**:
```python
results = openclio.runClio(..., verbose=True)
```

**Check cached results**:
```python
import cloudpickle
with open('./output/facetValues.pkl', 'rb') as f:
    facets = cloudpickle.load(f)
```

## Common Issues

**No module named 'pandas'**: Run `pip install -e .`

**Auth failed**: Run `gcloud auth application-default login`

**Quota exceeded**: Lower `llmBatchSize`, wait between batches

**Widget not showing**: Install Jupyter widgets: `pip install ipywidgets jupyter`

## Files

- `example_system_prompts.ipynb` - Working examples
- `LOCAL_SETUP.md` - Detailed setup guide
- `test_local.py` - Quick tests
- `README.md` - Full documentation

## Next Steps

1. Try the notebook examples
2. Modify the music facets for your use case
3. Test with your own data
4. Scale up to larger datasets
5. Explore the widget features

Questions? Check `LOCAL_SETUP.md` or the main `README.md`
