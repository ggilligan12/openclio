# OpenClio

Analyze and explore AI agent system prompts at scale using hierarchical clustering and interactive visualizations.

Based on [Anthropic's Clio: A system for privacy-preserving insights into real-world AI use](https://www.anthropic.com/research/clio).

## Features

- **Vertex AI Integration**: Uses Google Gemini for facet extraction with structured outputs
- **System Prompt Analysis**: Specialized facets for understanding AI agent purposes and capabilities
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

# Initialize models
llm = openclio.VertexLLMInterface(
    model_name="gemini-1.5-flash",
    project_id="your-gcp-project-id"
)
embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

# Run analysis
results = openclio.runClio(
    facets=openclio.systemPromptFacets,
    llm=llm,
    embeddingModel=embedding_model,
    data=prompts,
    outputDirectory="./output",
    displayWidget=True,
)
```

### What Gets Analyzed

The default `systemPromptFacets` extract:
- **Primary Purpose**: Main function or role of the AI agent
- **Domain**: Subject area (healthcare, finance, education, etc.)
- **Key Capabilities**: Important features and abilities
- **Interaction Style**: Tone and personality (professional, casual, etc.)

## How It Works

1. **Facet Extraction**: LLM analyzes each prompt using structured outputs (guaranteed JSON)
2. **Embedding**: Converts facet values to dense vectors
3. **Clustering**: Groups similar prompts hierarchically
4. **Visualization**: UMAP projection + interactive tree view

## Authentication

OpenClio uses Application Default Credentials in Colab:

```python
# In Colab, no special auth needed - workload identity handles it
llm = openclio.VertexLLMInterface(
    model_name="gemini-1.5-flash",
    project_id="your-project-id"
)
```

For local development, set up ADC:
```bash
gcloud auth application-default login
```

## Configuration

Key parameters in `runClio()`:

```python
results = openclio.runClio(
    facets=openclio.systemPromptFacets,  # What to extract
    llm=llm,                              # LLM for analysis
    embeddingModel=embedding_model,       # For clustering
    data=prompts,                         # Your text data
    outputDirectory="./output",           # Cache location
    displayWidget=True,                   # Show interactive widget
    llmBatchSize=10,                      # Lower for rate limits
    maxTextChars=32000,                   # Truncate long texts
    verbose=True,                         # Progress output
)
```

## Custom Facets

Define your own facets for domain-specific analysis:

```python
from openclio import Facet

custom_facets = [
    Facet(
        name="Security Level",
        question="What security constraints does this agent have?",
        prefill="The security constraints are",
        summaryCriteria="Cluster name should describe security approach"
    ),
    # ... more facets
]

results = openclio.runClio(facets=custom_facets, ...)
```

## Widget Features

The interactive widget provides:
- **UMAP Plot**: 2D visualization of all prompts
- **Hierarchy Tree**: Click to explore cluster structure
- **Text Viewer**: See prompts in selected clusters
- **Facet Annotations**: View extracted facets for each prompt

## Examples

See `example_system_prompts.ipynb` for a complete walkthrough.

## Performance Tips

- **Batch Size**: Use `llmBatchSize=10` for Vertex AI (rate limits)
- **Model Choice**: `gemini-1.5-flash` is fast, `gemini-1.5-pro` is more accurate
- **Checkpointing**: Results cached in `outputDirectory` - rerun resumes automatically
- **Large Datasets**: For >1000 prompts, expect 30-60 minutes with Flash

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
        print(f"\\n{facet.name} clusters:")
        for cluster in root_clusters:
            print(f"  {cluster.name} ({len(cluster.indices)} items)")
```

## Architecture

- **LLM Interface**: Abstracts Vertex AI (supports other backends)
- **Structured Outputs**: Pydantic models ensure correct JSON from Gemini
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

## Contributing

Issues and PRs welcome at https://github.com/ggilligan12/openclio
