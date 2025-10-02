# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

OpenClio analyzes AI agent system prompts at scale using Vertex AI (Gemini) and hierarchical clustering. It extracts structured facets, builds cluster hierarchies, and provides interactive visualizations in Jupyter/Colab.

## Installation & Setup

```bash
pip install git+https://github.com/ggilligan12/openclio.git
```

## Core Architecture

### Main Pipeline (openclio/openclio.py)

The primary entry point is `runClio()` which orchestrates:

1. **Deduplication**: Removes duplicate text blocks using string comparison
2. **Facet Extraction**: LLM extracts facets using Pydantic structured outputs (guaranteed JSON)
3. **Embedding**: SentenceTransformer converts facet values to vectors
4. **Base Clustering**: FaissKMeans creates initial clusters
5. **Hierarchy Building**: Iteratively clusters into higher-level categories
6. **UMAP Projection**: 2D visualization of embeddings
7. **Widget Display**: Interactive exploration in Jupyter/Colab

### Key Components

**openclio/openclio.py**: Main pipeline with integrated Vertex AI
- Uses google-generativeai SDK directly (no separate LLM interface)
- Gemini models via `genai.GenerativeModel(model_name)`
- Structured outputs using Pydantic models with `response.parsed`
- Rate limiting with exponential backoff built into each LLM call
- Authentication via `genai.configure(project=project_id, location=location)`

**Pydantic models for structured outputs** (defined in openclio.py):
- `SystemPromptFacets`: Built-in facets for system prompt analysis
- `ClusterNameAndSummary`: For cluster naming with name + summary
- All use `response_schema` in config for guaranteed JSON parsing

**openclio/opencliotypes.py**: Core data structures
- `FacetMetadata`: Defines facet configuration (name, summaryCriteria)
- `OpenClioConfig`: Configuration (batch sizes, hierarchy params, rate limits)
- `OpenClioResults`: Output (facets, clusters, hierarchies, UMAP)
- **Type aliasing**: `ConversationFacetData` = `DataPointFacetData` (backwards compat)

**openclio/prompts.py**: Prompt templates
- `getFacetPrompt()`: Main facet extraction prompt (works with text or conversations)
- `format_messages_as_text()`: Simple formatter when tokenizer unavailable
- `doCachedReplacements()`: Optimized prompt building with caching

**openclio/widget.py**: Interactive Jupyter/Colab widget
- `ClioWidget`: Main widget class
- UMAP plot (Plotly), hierarchy tree (HTML), text viewer (ipywidgets)
- Click clusters to explore, facet annotations

**openclio/utils.py**: Utilities
- `dedup()`: Generic deduplication
- `runBatched()`: Batch processing framework
- No longer used: `runWebui()` (deprecated in favor of widget)

**openclio/faissKMeans.py**: Fast k-means using FAISS

**openclio/writeOutput.py**: Legacy web output (optional)
- `convertOutputToWebpage()`: Static HTML with chunked JSON
- Still available but widget is primary interface

## Working with Data

### Data Format

OpenClio now expects **List[str]** (text blocks), not conversations:

```python
data = [
    "You are a customer support agent...",
    "You are a code reviewer...",
    # ... more system prompts
]
```

Legacy conversation format `List[List[Dict]]` still works for backwards compatibility.

### Facets

**System prompt facets** (built-in):
OpenClio includes `SystemPromptFacets` Pydantic model and `systemPromptFacetMetadata`:
- Primary Purpose
- Domain
- Key Capabilities
- Interaction Style

**Custom facets** (two steps):
1. Define Pydantic schema:
```python
from pydantic import BaseModel, Field

class MusicFacets(BaseModel):
    genre: str = Field(description="The specific genre of music")
    tone: str = Field(description="The tone (calm, upbeat, aggressive, etc.)")
```

2. Define metadata (controls which facets get hierarchical clustering):
```python
music_metadata = {
    "genre": FacetMetadata(
        name="Genre",
        summaryCriteria="Cluster name should describe the overall genre"
    ),
    "tone": FacetMetadata(
        name="Tone",
        summaryCriteria="Cluster name should describe the tone"
    ),
}
```

**Important**: Only facets with `summaryCriteria` will get hierarchical clustering. Facets without it are extracted but not clustered.

## Configuration

### Common Settings (openclio/opencliotypes.py)

**Rate Limiting** (new in this version):
- `vertexRateLimitPerMin=30`: Conservative default for Vertex AI (increase if you have higher quota)
- `maxParallelLLMCalls=5`: Number of parallel API calls (lower = more reliable rate limiting)
- `minDelayBetweenRequests=0.5`: Minimum delay between requests in seconds
- Exponential backoff on rate limit errors (2s, 4s, 8s retries)

**Performance**:
- `llmBatchSize=50`: Lower for Vertex API rate limits (default: 50)
- `embedBatchSize=1000`: Embedding batch size
- `maxTextChars=32000`: Truncate long texts (suitable for system prompts)

**Hierarchy shape**:
- `minTopLevelSize=5`: Stop when ≤ this many top clusters
- `nBaseClustersFunc=lambda n: n//10`: Base cluster count
- `nDesiredHigherLevelNamesPerClusterFunc=lambda n: n//3`: Branching factor

**Widget/Output**:
- `displayWidget=True`: Show interactive widget in Colab
- `htmlRoot=None`: Optional web output path (legacy)

## Output & Visualization

### Widget Mode (Primary)

```python
results = openclio.runClio(
    facetSchema=openclio.SystemPromptFacets,
    facetMetadata=openclio.systemPromptFacetMetadata,
    embeddingModel=embedding_model,
    data=prompts,
    outputDirectory="./output",
    project_id="your-gcp-project",
    model_name="gemini-1.5-flash",  # or gemini-1.5-pro
    displayWidget=True,  # Shows widget
)
```

Widget features:
- Facet selector dropdown
- UMAP scatter plot
- Hierarchical tree (clickable)
- Text viewer with facet annotations

### Programmatic Access

```python
# Get facets for specific text
facets = results.facetValues[0].facetValues
for fv in facets:
    print(f"{fv.facet.name}: {fv.value}")

# Explore hierarchy
for cluster in results.rootClusters[facet_idx]:
    print(f"{cluster.name} ({len(cluster.indices)} items)")
```

## Development Notes

### Data Flow

`List[str]` → dedup → facet extraction (structured JSON) → embedding → base clustering → hierarchy → UMAP → widget

### Checkpointing

Results cached in `outputDirectory`:
- `dedupedData.pkl`
- `facetValues.pkl`
- `facetValuesEmbeddings.pkl`
- `baseClusters.pkl`
- `rootClusters.pkl`
- `umapResults.pkl`
- `results.pkl`

Delete specific files to recompute those stages.

### Structured Outputs

Gemini returns guaranteed JSON matching Pydantic schemas via google-generativeai SDK:

```python
# In openclio.py
response = vertex_model.generate_content(
    contents=[prompt],
    config={
        "response_mime_type": "application/json",
        "response_schema": SystemPromptFacets,  # Pass Pydantic class directly
        "max_output_tokens": 1000,
        "temperature": 0.7,
    }
)
# Google AI SDK parses and validates for us!
facet_data = response.parsed  # Already a Pydantic object
```

No manual JSON parsing or XML tag extraction needed - the SDK handles everything.

### Vertex AI Specifics

- **Auth**: Uses `genai.configure(project=project_id, location=location)`
  - In Colab: Automatically uses workload identity
  - Locally: Uses Application Default Credentials (run `gcloud auth application-default login`)
- **Rate Limits**: Default 30 req/min (conservative), configurable via `vertexRateLimitPerMin`
- **Models**: `gemini-1.5-flash` (fast, cheap), `gemini-1.5-pro` (accurate, expensive), `gemini-2.0-flash-exp` (newest)
- **Retries**: Built-in exponential backoff (2s, 4s, 8s) on rate limit errors (429, quota, rate)
- **No tokenizer**: Uses character-based truncation (`maxTextChars=32000`)

### Type System

Keep both old and new names for backwards compatibility:
- `ConversationFacetData` (old) = `DataPointFacetData` (new)
- `ConversationCluster` (old) = `DataCluster` (new)
- `data: List[Any]` supports both `List[str]` and `List[List[Dict]]`

## Common Tasks

### Add new facet
1. Define Pydantic schema with Field descriptions
2. Create metadata dict with `FacetMetadata` for facets you want clustered
3. Pass both to `runClio(facetSchema=..., facetMetadata=..., ...)`

Example:
```python
class CustomFacets(BaseModel):
    topic: str = Field(description="Main topic discussed")
    sentiment: str = Field(description="Overall sentiment (positive/negative/neutral)")

custom_metadata = {
    "topic": FacetMetadata(name="Topic", summaryCriteria="Cluster by topic"),
    # sentiment won't be clustered (no entry in metadata)
}
```

### Customize widget
1. Edit `openclio/widget.py`
2. Modify `_update_plot()`, `_update_tree()`, or `_update_text_viewer()`
3. Add new ipywidgets components to `_create_widgets()`

### Debug facet extraction
1. Check `facetValues.pkl` in outputDirectory
2. Look at LLM prompts and responses: `cfg.verbose=True`
3. Verify structured output parsing: `response.parsed` should return Pydantic object
4. Check for rate limit errors in output (will show retry messages)

### Adjust rate limiting
If getting rate limit errors:
```python
cfg = OpenClioConfig(
    vertexRateLimitPerMin=20,  # Lower if still hitting limits
    maxParallelLLMCalls=3,     # Reduce parallelism
    minDelayBetweenRequests=1.0,  # Increase delay
)
```

## Dependencies

Core:
- `google-generativeai`: Vertex AI via modern SDK (replaces google-cloud-aiplatform)
- `pydantic`: Structured outputs and schema definitions
- `sentence-transformers`: Embeddings
- `ipywidgets`, `plotly`: Interactive widget
- `umap-learn`, `faiss-cpu` (or `faiss-gpu`): Clustering/projection
- Standard library: `threading`, `time` for rate limiting

## Migration Notes

If you have code using the old VLLM + conversations version:

**Old**:
```python
llm = vllm.LLM(model="Qwen/Qwen3-8B")
data = [[{"role": "user", "content": "hi"}], ...]  # conversations
results = clio.runClio(..., hostWebui=True)
```

**New (current version)**:
```python
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

# Define facets as Pydantic schema
class MyFacets(BaseModel):
    purpose: str = Field(description="What this does")

# Define metadata
my_metadata = {
    "purpose": FacetMetadata(name="Purpose", summaryCriteria="...")
}

# Run with Vertex AI (no separate LLM object needed)
embedding_model = SentenceTransformer('all-mpnet-base-v2')
data = ["You are a helpful assistant...", ...]  # text

results = openclio.runClio(
    facetSchema=MyFacets,
    facetMetadata=my_metadata,
    embeddingModel=embedding_model,
    data=data,
    outputDirectory="./output",
    project_id="your-gcp-project",
    model_name="gemini-1.5-flash",
    displayWidget=True
)
```

**Key changes**:
1. No separate LLM object - Vertex AI integrated directly
2. Use Pydantic schemas for facets instead of `Facet()` objects
3. Pass `facetSchema` and `facetMetadata` instead of `facets`
4. `displayWidget=True` instead of `hostWebui=True`
5. Conversations still work but text (system prompts) is primary use case
