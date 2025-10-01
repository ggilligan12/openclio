# OpenClio

Analyze and explore AI agent system prompts at scale using hierarchical clustering and interactive visualizations.

Based on [Anthropic's Clio: A system for privacy-preserving insights into real-world AI use](https://www.anthropic.com/research/clio).

## Features

- **Efficient Multi-Facet Extraction**: Extract all facets in one LLM call (N calls instead of N×M)
- **Vertex AI Integration**: Uses Google Gemini with structured outputs (guaranteed JSON)
- **System Prompt Analysis**: Pre-built facets for analyzing AI agent purposes and capabilities
- **Interactive Widget**: Explore results in Jupyter/Colab with UMAP plots and hierarchical trees
- **Automatic Clustering**: Discover patterns in thousands of system prompts
- **Checkpointing**: Resume interrupted analyses from cached results

## Quick Start

### Installation

```bash
pip install git+https://github.com/ggilligan12/openclio.git
```

### Basic Usage

```python
import openclio
from sentence_transformers import SentenceTransformer

# Your system prompts
prompts = [
    "You are a customer support agent...",
    "You are a creative writing assistant...",
    # ... more prompts
]

# Initialize embedding model
embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

# Run analysis
results = openclio.runClio(
    facetSchema=openclio.SystemPromptFacets,
    facetMetadata=openclio.systemPromptFacetMetadata,
    embeddingModel=embedding_model,
    data=prompts,
    outputDirectory="./output",
    project_id="your-gcp-project-id",
    model_name="gemini-1.5-flash-002",
    displayWidget=True,
)
```

### What Gets Analyzed

The default `SystemPromptFacets` extract:
- **Primary Purpose**: Main function or role of the AI agent
- **Domain**: Subject area (healthcare, finance, education, etc.)
- **Key Capabilities**: Important features and abilities
- **Interaction Style**: Tone and personality (professional, casual, etc.)

## How It Works

1. **Multi-Facet Extraction**: LLM analyzes each prompt once, extracting ALL facets in a single call (4x fewer API calls for 4 facets!)
2. **Structured Outputs**: Pydantic schemas ensure guaranteed JSON responses
3. **Embedding**: Converts facet values to dense vectors
4. **Clustering**: Groups similar prompts hierarchically
5. **Visualization**: UMAP projection + interactive tree view

## Authentication

OpenClio uses Application Default Credentials:

**In Colab:**
```python
# Workload identity handles auth automatically
results = openclio.runClio(
    project_id="your-project-id",
    ...
)
```

**Local development:**
```bash
gcloud auth application-default login
```

## Configuration

Key parameters in `runClio()`:

```python
results = openclio.runClio(
    facetSchema=openclio.SystemPromptFacets,      # Pydantic schema defining facets
    facetMetadata=openclio.systemPromptFacetMetadata,  # Clustering config per facet
    embeddingModel=embedding_model,               # SentenceTransformer for clustering
    data=prompts,                                 # Your text data
    outputDirectory="./output",                   # Cache location
    project_id="your-gcp-project",                # GCP project ID
    model_name="gemini-1.5-flash-002",            # Gemini model to use
    location="us-central1",                       # GCP region
    displayWidget=True,                           # Show interactive widget
    llmBatchSize=10,                              # Batch size for API calls
    maxTextChars=32000,                           # Truncate long texts
    verbose=True,                                 # Progress output
)
```

## Custom Facets

Define your own facets using Pydantic models:

```python
from pydantic import BaseModel, Field
from openclio import FacetMetadata

class MusicFacets(BaseModel):
    genre: str = Field(description="In the greatest degree of specificity possible, what genre of music is this")
    language: list[str] = Field(description="What language (if any) is this music being produced in, eg. English, French, Hindi, etc.")
    tone: str = Field(description="What is the tone of the music, eg. calm, soothing, upbeat, aggressive etc.")
    instrumentation: str = Field(description="What instrumentation is in use, ie. Electronic, Guitar, Orchstra etc.")

# Define which facets should get cluster hierarchies
facet_metadata = {
    "genre": FacetMetadata(
        name="Genre",
        summaryCriteria="Cluster name should describe the overall genre"
    ),
    "alignment": FacetMetadata(
        name="Tone",
        summaryCriteria="Cluster name should describe the tone of the music"
    ),
    # Keywords and narrative won't get hierarchies (no metadata entry)
}

results = openclio.runClio(
    facetSchema=MusicFacets,
    facetMetadata=facet_metadata,
    ...
)
```

**Key insight**: All facets are extracted in one LLM call, making this dramatically more efficient than the old approach!

## Widget Features

The interactive widget provides:
- **UMAP Plot**: 2D visualization of all prompts
- **Hierarchy Tree**: Click to explore cluster structure
- **Text Viewer**: See prompts in selected clusters
- **Facet Annotations**: View extracted facets for each prompt

## Examples

See `notebook.ipynb` for a complete walkthrough with both system prompt analysis and custom facet examples.

## Performance Tips

- **Batch Size**: Use `llmBatchSize=10` for Vertex AI (rate limits ~60 req/min)
- **Model Choice**: `gemini-1.5-flash-002` is fast and cheap, `gemini-1.5-pro` is more accurate
- **Checkpointing**: Results cached in `outputDirectory` - rerun resumes automatically
- **Large Datasets**: For 1000 prompts with 4 facets, expect ~15-20 minutes with Flash (only 1000 calls!)

## Programmatic Access

Access results without the widget:

```python
# Get facets for a specific prompt
prompt_facets = results.facetValues[0]
for fv in prompt_facets.facetValues:
    print(f"{fv.facet.name}: {fv.value}")

# Explore cluster hierarchy
for facet_idx, facet in enumerate(results.facets):
    root_clusters = results.rootClusters[facet_idx]
    if root_clusters:
        print(f"\n{facet.name} clusters:")
        for cluster in root_clusters:
            print(f"  {cluster.name} ({len(cluster.indices)} items)")
```

## Architecture

- **Multi-Facet Extraction**: Single Pydantic schema extracts all facets in one LLM call
- **Vertex AI Direct**: No abstraction layer - Gemini called directly for simplicity
- **Structured Outputs**: Vertex AI `response_schema` ensures valid JSON
- **Facet Pipeline**: Extract → Embed → Cluster → Hierarchy → Visualize
- **Widget**: ipywidgets + Plotly for interactive exploration

## Requirements

- Python 3.8+
- Google Cloud Project with Vertex AI enabled
- Dependencies: `google-cloud-aiplatform`, `sentence-transformers`, `ipywidgets`, `plotly`, `pydantic`, `umap-learn`, etc.

## Troubleshooting

**Quota errors**: Lower `llmBatchSize`, use Flash instead of Pro

**Auth errors**: Verify project ID, check Vertex AI API is enabled

**Widget not showing**: Make sure you're in Jupyter/Colab environment

**Out of memory**: Reduce `embedBatchSize` or process in smaller batches

## Migration from Old API

If you have existing code using the old `VertexLLMInterface`:

**Old:**
```python
llm = openclio.VertexLLMInterface(model_name="gemini-1.5-flash", project_id="...")
results = openclio.runClio(facets=openclio.systemPromptFacets, llm=llm, ...)
```

**New:**
```python
results = openclio.runClio(
    facetSchema=openclio.SystemPromptFacets,
    facetMetadata=openclio.systemPromptFacetMetadata,
    project_id="...",
    model_name="gemini-1.5-flash-002",
    ...
)
```

## Citation

If you use OpenClio in research, please cite the original Clio paper:

```
@article{clio2024,
  title={Clio: Privacy-Preserving Insights into Real-World AI Use},
  author={Anthropic},
  year={2024}
}
```

## License

MIT License - see LICENSE file

## Local Development

Want to contribute or develop locally? See the **`dev/`** folder:

- **`dev/QUICKSTART.md`** - Get started in 5 minutes
- **`dev/LOCAL_SETUP.md`** - Comprehensive development guide
- **`dev/setup_dev.sh`** - One-command setup
- **`dev/test_local.py`** - Automated tests

Quick setup:
```bash
./dev/setup_dev.sh
gcloud auth application-default login
python dev/test_local.py
```

## Contributing

Issues and PRs welcome at https://github.com/ggilligan12/openclio
