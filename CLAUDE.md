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

**openclio/llm_interface.py**: Abstract LLM interface
- `LLMInterface`: Base class with `generate_batch()` method
- Supports `response_schema` parameter for structured outputs

**openclio/vertex_llm.py**: Vertex AI implementation
- Uses Gemini models (Flash or Pro)
- Handles rate limiting, retries, exponential backoff
- Authentication via workload identity (Colab) or ADC (local)
- Structured outputs via `response_schema` parameter

**openclio/structured_outputs.py**: Pydantic models for guaranteed JSON
- `FacetAnswer`: For facet extraction
- `ClusterNames`: For cluster naming
- `DeduplicatedNames`, `ClusterAssignment`, `ClusterRenaming`: For hierarchy building

**openclio/opencliotypes.py**: Core data structures
- `Facet`: Defines what to extract (question, prefill, summaryCriteria)
- `OpenClioConfig`: Configuration (batch sizes, hierarchy params, etc.)
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

**systemPromptFacets** (default for system prompts):
- Primary Purpose
- Domain
- Key Capabilities
- Interaction Style

**Custom facets**:
```python
Facet(
    name="Security Level",
    question="What security constraints does this agent have?",
    prefill="The security constraints are",
    summaryCriteria="Cluster name should describe security approach"
)
```

**Important**: `summaryCriteria` is required for hierarchy building.

## Configuration

### Common Settings (openclio/opencliotypes.py)

**Performance**:
- `llmBatchSize=50`: Lower for Vertex API rate limits (default: 50, was 1000)
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
    facets=openclio.systemPromptFacets,
    llm=llm,
    embeddingModel=embedding_model,
    data=prompts,
    outputDirectory="./output",
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

Gemini returns guaranteed JSON matching Pydantic schemas:

```python
# In openclio.py
llm.generate_batch(prompts, response_schema=FacetAnswer, ...)

# Returns JSON like: {"answer": "Customer support agent"}
```

Fallback to tag extraction if JSON parsing fails (backwards compat).

### Vertex AI Specifics

- **Auth**: Workload identity in Colab, ADC elsewhere
- **Rate Limits**: ~60 req/min for Flash, lower for Pro
- **Models**: `gemini-1.5-flash` (fast), `gemini-1.5-pro` (accurate)
- **Retries**: Exponential backoff with tenacity
- **No tokenizer**: Uses character-based truncation instead

### Type System

Keep both old and new names for backwards compatibility:
- `ConversationFacetData` (old) = `DataPointFacetData` (new)
- `ConversationCluster` (old) = `DataCluster` (new)
- `data: List[Any]` supports both `List[str]` and `List[List[Dict]]`

## Common Tasks

### Add new facet
1. Create `Facet()` with name, question, prefill, summaryCriteria
2. Pass to `runClio(facets=[...], ...)`

### Change LLM backend
1. Implement `LLMInterface` with `generate_batch()` method
2. Support `response_schema` parameter for structured outputs
3. Return list of strings (JSON if schema provided)

### Customize widget
1. Edit `openclio/widget.py`
2. Modify `_update_plot()`, `_update_tree()`, or `_update_text_viewer()`
3. Add new ipywidgets components to `_create_widgets()`

### Debug facet extraction
1. Check `facetValues.pkl` in outputDirectory
2. Look at LLM prompts: `cfg.verbose=True`
3. Verify JSON parsing: inspect `processOutputFunc` in `getFacetValues()`

## Dependencies

Core:
- `google-cloud-aiplatform`: Vertex AI
- `pydantic`: Structured outputs
- `sentence-transformers`: Embeddings
- `ipywidgets`, `plotly`: Interactive widget
- `umap-learn`, `faiss-cpu`: Clustering/projection
- `tenacity`: Retry logic

## Migration Notes

If you have code using the old VLLM + conversations version:

**Old**:
```python
llm = vllm.LLM(model="Qwen/Qwen3-8B")
data = [[{"role": "user", "content": "hi"}], ...]  # conversations
results = clio.runClio(..., hostWebui=True)
```

**New**:
```python
llm = openclio.VertexLLMInterface(model_name="gemini-1.5-flash", project_id="...")
data = ["You are a helpful assistant...", ...]  # text
results = openclio.runClio(..., displayWidget=True)
```

Conversations still work but text is primary.
