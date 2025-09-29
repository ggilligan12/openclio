# OpenClio Refactor Plan: GCP Vertex + Text Blocks + Colab Widget

## Overview of Changes

1. **VLLM → GCP Vertex AI**: Replace local LLM with Vertex AI API
2. **Conversations → Text Blocks**: Simplify from conversation format to plain text
3. **Web Server → Colab Widget**: Replace static HTML + server with interactive Jupyter widget

---

## Phase 1: LLM Abstraction Layer (Foundation)

**Goal**: Create abstraction that allows both VLLM and Vertex to work, making migration safer.

### 1.1 Create LLM Interface (`openclio/llm_interface.py`)
- [ ] Define abstract `LLMInterface` class with methods:
  - `generate_batch(prompts: List[str], **kwargs) -> List[str]`
  - `get_tokenizer()` (optional, may return None for Vertex)
  - `supports_chat_template() -> bool`
- [ ] Implement `VLLMLLMInterface` wrapper around existing `vllm.LLM`
- [ ] Implement `VertexLLMInterface` with:
  - Authentication (ADC, service account, or API key)
  - Rate limiting and retry logic
  - Batching strategy (Vertex has different limits than VLLM)
  - Error handling for quota/permission issues

**Files to modify**:
- `openclio/openclio.py`: Accept `LLMInterface` instead of `vllm.LLM`
- `openclio/opencliotypes.py`: Update type hints from `vllm.LLM` to `LLMInterface`

**Dependencies to add**:
```
google-cloud-aiplatform>=1.38.0
tenacity  # for retry logic
```

**Testing checkpoint**: Ensure VLLM wrapper works with existing code before proceeding.

---

## Phase 2: Tokenizer Independence

**Goal**: Remove dependency on tokenizer for core functionality since Vertex doesn't provide one.

### 2.1 Refactor Prompt Building (`openclio/prompts.py`)
- [ ] Make `doCachedReplacements()` work without tokenizer:
  - Add parameter `use_chat_template: bool = True`
  - When False, skip `apply_chat_template` and just format messages as text
  - Create simple message formatter: `format_messages_as_text(messages) -> str`
- [ ] Update all prompt functions to have dual-mode:
  - If tokenizer available + supports chat template: use it
  - Otherwise: use simple text formatting
- [ ] Remove tokenizer requirement from `getFacetPrompt`, `getNeighborhoodClusterNamesPrompt`, etc.

### 2.2 Handle UMAP Tokenization (`openclio/writeOutput.py`)
- [ ] In `computeUmap()`: make tokenizer optional
- [ ] If no tokenizer, use character-based truncation instead of token-based

**Files to modify**:
- `openclio/prompts.py`: All prompt generation functions
- `openclio/writeOutput.py`: `computeUmap()` function
- `openclio/openclio.py`: Pass `llm.get_tokenizer()` but handle None

**Testing checkpoint**: Test with `use_chat_template=False` to simulate Vertex behavior.

---

## Phase 3: Vertex AI Implementation

**Goal**: Complete Vertex AI integration with proper production features.

### 3.1 Vertex LLM Interface (`openclio/vertex_llm.py`)
```python
class VertexLLMInterface(LLMInterface):
    def __init__(
        self,
        model_name: str,
        project_id: str,
        location: str = "us-central1",
        credentials_path: Optional[str] = None,
        max_retries: int = 3,
        requests_per_minute: int = 60,
        batch_size: int = 5  # Vertex has lower concurrent limits
    )
```

- [ ] Implement authentication:
  - Try credentials_path if provided
  - Fall back to Application Default Credentials
  - Validate permissions on initialization
- [ ] Implement `generate_batch()`:
  - Chunk large batches to respect rate limits
  - Use `tenacity` for exponential backoff retry
  - Handle Vertex-specific errors (quota, safety filters, etc.)
  - Progress tracking for large batches
- [ ] Implement safety settings configuration
- [ ] Add cost tracking/logging (token counts)

### 3.2 Configuration Updates
- [ ] Add Vertex-specific config to `OpenClioConfig`:
  ```python
  vertexProjectId: Optional[str] = None
  vertexLocation: str = "us-central1"
  vertexCredentialsPath: Optional[str] = None
  vertexSafetySettings: Dict[str, Any] = field(default_factory=dict)
  ```

**Files to create**:
- `openclio/vertex_llm.py`

**Files to modify**:
- `openclio/opencliotypes.py`: Add Vertex config fields
- `openclio/__init__.py`: Export Vertex interface
- `pyproject.toml` & `setup.py`: Add Vertex dependencies

**Testing checkpoint**: Create minimal Colab notebook testing Vertex API calls.

---

## Phase 4: Data Format Simplification

**Goal**: Change from `List[List[Dict]]` (conversations) to `List[str]` (text blocks).

### 4.1 Update Core Types (`openclio/opencliotypes.py`)
- [ ] Rename types for clarity:
  - `ConversationFacetData` → `DataPointFacetData`
  - `ConversationEmbedding` → `DataPointEmbedding`
  - `ConversationCluster` → `DataCluster`
- [ ] Change type hints:
  - `data: List[List[Dict[str, str]]]` → `data: List[str]`
  - `conversation: List[Any]` → `text: str`
- [ ] Update `OpenClioConfig`:
  - Remove `getConversationFunc` (no longer needed)
  - Remove `conversationToStrFunc` (identity function)
  - Remove `maxConversationTokens` (use `maxTextTokens`)
  - Simplify `dedupKeyFunc` default to `lambda text: text.strip().lower()`

### 4.2 Update Prompts (`openclio/prompts.py`)
- [ ] Remove `conversationToString()` function
- [ ] Create new generic prompts:
  ```python
  def getFacetPromptForText(facet: Facet, text: str, cfg: OpenClioConfig) -> str:
      # Simple template without conversation structure
  ```
- [ ] Update default facets or create new ones:
  ```python
  textFacets = [
      Facet(name="Topic", question="What is the main topic of this text?", ...),
      Facet(name="Sentiment", question="What is the sentiment?", ...),
      Facet(name="Category", question="What category does this belong to?", ...)
  ]
  ```
- [ ] Remove conversation-specific prompt templates

### 4.3 Update Main Pipeline (`openclio/openclio.py`)
- [ ] Change function signatures to accept `List[str]`
- [ ] Update `runClio()`:
  ```python
  def runClio(
      facets: List[Facet],
      llm: LLMInterface,  # Changed
      embeddingModel: SentenceTransformer,
      data: List[str],  # Changed
      outputDirectory: str,
      htmlRoot: str = None,  # Now optional
      displayWidget: bool = False,  # New
      cfg: OpenClioConfig = None,
      **kwargs
  ) -> OpenClioResults:
  ```
- [ ] Remove conversation preprocessing
- [ ] Simplify deduplication logic

### 4.4 Update Utilities (`openclio/utils.py`)
- [ ] Update `dedup()` for text instead of conversations
- [ ] Update or remove `getExampleData()`:
  ```python
  def getExampleData() -> List[str]:
      # Load text samples instead of conversations
      # Could extract just user messages from wildchat
  ```

**Files to modify**:
- `openclio/opencliotypes.py`
- `openclio/prompts.py`
- `openclio/openclio.py`
- `openclio/utils.py`

**Testing checkpoint**: Test with simple list of strings before moving to widget.

---

## Phase 5: Colab Widget Output

**Goal**: Replace web server with interactive Jupyter widget.

### 5.1 Widget Architecture Design

**Component structure**:
```
ColabClioWidget
├── FacetSelector (dropdown)
├── UMAPPlot (plotly scatter with hulls)
├── HierarchyTree (ipytree or custom HTML)
├── TextViewer (text area with facet tags)
└── FilterPanel (box select, word cloud)
```

### 5.2 Create Widget (`openclio/widget.py`)
```python
class ColabClioWidget:
    def __init__(self, results: OpenClioResults):
        self.results = results
        self.selected_facet = 0
        self.selected_cluster = None

    def _create_umap_plot(self, facet_idx: int) -> go.Figure:
        # Plotly scatter plot with cluster hulls

    def _create_hierarchy_tree(self, facet_idx: int) -> widgets.HTML:
        # Interactive tree view

    def _create_text_viewer(self, indices: List[int]) -> widgets.HTML:
        # Display selected texts with facets

    def display(self):
        # Compose widget and display
```

### 5.3 Implementation Details
- [ ] Install dependencies:
  ```
  ipywidgets>=8.0
  plotly>=5.0
  ipytree  # for tree view
  ```
- [ ] UMAP plot features:
  - Plotly scatter plot (better than matplotlib for Colab)
  - Click cluster to highlight
  - Box select to filter points
  - Toggle hull visibility
  - Color by cluster
- [ ] Tree view features:
  - Expand/collapse nodes
  - Click to filter data
  - Show statistics (count, percentage)
  - Color code by depth
- [ ] Text viewer features:
  - Paginated display (50 per page)
  - Facet value tags
  - Search/filter
  - Export selected
- [ ] Data loading:
  - Embed data in widget (no separate files needed in Colab)
  - Lazy load text blocks (only load when viewed)
  - Cache rendered components

### 5.4 Update Main Pipeline
- [ ] Modify `runClio()`:
  ```python
  if displayWidget:
      from .widget import ColabClioWidget
      widget = ColabClioWidget(output)
      widget.display()
      return output
  elif htmlRoot is not None:
      # Keep old web output as option
      convertOutputToWebpage(...)
  else:
      # Just return results
      return output
  ```
- [ ] Remove `hostWebui` parameter
- [ ] Make `convertOutputToWebpage()` optional (for backwards compat)

**Files to create**:
- `openclio/widget.py`

**Files to modify**:
- `openclio/openclio.py`: Widget display logic
- `openclio/writeOutput.py`: Make web output optional
- `openclio/utils.py`: Remove or deprecate `runWebui()`
- `pyproject.toml` & `setup.py`: Add widget dependencies

**Testing checkpoint**: Test widget in Colab with sample data.

---

## Phase 6: Cleanup & Documentation

### 6.1 Remove Dead Code
- [ ] Remove conversation-specific utilities
- [ ] Remove `runWebui()` and related server code
- [ ] Remove conversation rendering from `websiteTemplate.html` (or keep for legacy)
- [ ] Clean up unused imports

### 6.2 Update Examples
- [ ] Create Colab notebook: `examples/quickstart_vertex.ipynb`
  ```python
  from openclio import VertexLLMInterface, textFacets, runClio
  from sentence_transformers import SentenceTransformer

  # Load data
  texts = ["Example text 1", "Example text 2", ...]

  # Setup models
  llm = VertexLLMInterface(
      model_name="gemini-1.5-pro",
      project_id="my-project"
  )
  embedding_model = SentenceTransformer('all-mpnet-base-v2')

  # Run analysis
  results = runClio(
      facets=textFacets,
      llm=llm,
      embeddingModel=embedding_model,
      data=texts,
      outputDirectory="output",
      displayWidget=True
  )
  ```

### 6.3 Update Documentation
- [ ] Update `README.md` with new usage
- [ ] Update `CLAUDE.md` with new architecture
- [ ] Add migration guide for users of old version
- [ ] Document Vertex AI setup (project, permissions, billing)
- [ ] Add troubleshooting section (quota errors, auth issues)

### 6.4 Testing
- [ ] Unit tests for `VertexLLMInterface`
- [ ] Integration test with Vertex API
- [ ] Widget rendering tests
- [ ] End-to-end test in Colab
- [ ] Test with various text types (short, long, special chars)

---

## Phase 7: Optimization & Polish

### 7.1 Performance
- [ ] Optimize Vertex API batching (find sweet spot for concurrent requests)
- [ ] Cache embeddings more aggressively
- [ ] Widget: virtual scrolling for large datasets
- [ ] Widget: debounce interactions

### 7.2 UX Improvements
- [ ] Progress bars for Vertex API calls (can be slow)
- [ ] Better error messages for Vertex auth/quota
- [ ] Widget: keyboard shortcuts
- [ ] Widget: export functionality (CSV, JSON)
- [ ] Dark mode for widget

### 7.3 Configurability
- [ ] Allow custom Vertex models (Gemini, Claude on Vertex, etc.)
- [ ] Temperature/sampling params per facet
- [ ] Widget theming options
- [ ] Pluggable facet templates

---

## Migration Path (Backwards Compatibility)

To allow gradual migration:

1. **Keep VLLM support**: Don't remove VLLM, just add Vertex as option
2. **Keep conversation format**: Add `data_format` parameter:
   ```python
   def runClio(
       data: Union[List[str], List[List[Dict]]],
       data_format: Literal["text", "conversation"] = "text",
       ...
   )
   ```
3. **Keep web output**: Make both widget and web available:
   ```python
   output_mode: Literal["widget", "web", "none"] = "widget"
   ```

This allows:
- Users to test Vertex without changing everything
- Gradual codebase migration
- Easier debugging (compare old vs new)

---

## Dependency Changes Summary

**Add**:
```toml
google-cloud-aiplatform>=1.38.0
tenacity>=8.0.0
ipywidgets>=8.0.0
plotly>=5.0.0
ipytree>=0.2.0
```

**Keep** (no changes):
```toml
umap-learn
sentence_transformers
numpy
scikit-learn
scipy
torch
pandas
```

**Optional/Remove**:
```toml
vllm  # Make optional
concave_hull  # Still needed for widget hulls
faiss-cpu
cryptography  # Remove (no password protection needed in Colab)
```

---

## File-by-File Change Summary

| File | Phase | Changes |
|------|-------|---------|
| `openclio/llm_interface.py` | 1 | **CREATE** - Abstract LLM interface |
| `openclio/vertex_llm.py` | 3 | **CREATE** - Vertex implementation |
| `openclio/widget.py` | 5 | **CREATE** - Colab widget |
| `openclio/openclio.py` | 1,2,4,5 | **MAJOR** - New LLM interface, text format, widget display |
| `openclio/opencliotypes.py` | 1,3,4 | **MAJOR** - Type updates, Vertex config |
| `openclio/prompts.py` | 2,4 | **MAJOR** - Tokenizer independence, text prompts |
| `openclio/writeOutput.py` | 2,5 | **MODERATE** - Optional tokenizer, optional web |
| `openclio/utils.py` | 4,5 | **MODERATE** - Text dedup, remove webserver |
| `openclio/__init__.py` | 3,5 | **MINOR** - Export new classes |
| `openclio/faissKMeans.py` | - | **NO CHANGE** |
| `openclio/lzutf8.py` | - | **NO CHANGE** |
| `pyproject.toml` | 3,5 | **MINOR** - Dependencies |
| `setup.py` | 3,5 | **MINOR** - Dependencies |
| `README.md` | 6 | **MAJOR** - New usage docs |
| `CLAUDE.md` | 6 | **MAJOR** - New architecture docs |

---

## Risk Mitigation

1. **Vertex API costs**: Add budget warnings, token counting, dry-run mode
2. **Rate limits**: Implement conservative defaults, exponential backoff
3. **Auth complexity**: Clear error messages, setup validation on init
4. **Widget rendering**: Graceful degradation for large datasets
5. **Breaking changes**: Semantic versioning (2.0.0), migration guide

---

## Timeline Estimate

- **Phase 1**: 1 day (LLM abstraction)
- **Phase 2**: 1 day (Tokenizer independence)
- **Phase 3**: 2 days (Vertex implementation + testing)
- **Phase 4**: 2 days (Data format changes)
- **Phase 5**: 3 days (Widget implementation)
- **Phase 6**: 1 day (Cleanup + docs)
- **Phase 7**: 2 days (Polish)

**Total**: ~12 days of focused work

---

## Next Steps

1. Review this plan together
2. Prioritize phases (can we skip any?)
3. Decide on backwards compatibility requirements
4. Start with Phase 1 (LLM abstraction)
