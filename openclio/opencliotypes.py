from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple, Callable, Any, List, TypeAlias, Type
import numpy as np
from numpy import typing as npt
from pydantic import BaseModel

from .faissKMeans import FaissKMeans

EmbeddingArray: TypeAlias = npt.NDArray[np.float32]

@dataclass(frozen=True)
class FacetMetadata:
    """Metadata for a facet field - used for clustering configuration"""
    name: str
    summaryCriteria: Optional[str] = None  # If provided, will build cluster hierarchy for this facet

def shouldMakeFacetClusters(facet: FacetMetadata) -> bool:
    """Returns true if we should make the cluster hierarchy for the given facet"""
    return facet.summaryCriteria is not None

# Legacy Facet class - deprecated but kept for backwards compatibility
@dataclass(frozen=True)
class Facet:
    name: str
    question: str = ""
    prefill: str = ""
    summaryCriteria: Optional[str] = None
    numeric: Optional[Tuple[int, int]] = None
    getFacetPrompt: Optional[Callable[[Any, "Facet", Any, "OpenClioConfig"], str]] = None

@dataclass
class FacetValue:
    facet: FacetMetadata  # Changed from Facet to FacetMetadata
    value: str

@dataclass
class DataPointFacetData:
    """Facet data for a single data point (text)"""
    data: Any  # Text string
    facetValues: List[FacetValue]

# Backwards compatibility alias for old pickle files
ConversationFacetData = DataPointFacetData

@dataclass
class DataPointEmbedding:
    """Embedding for a single data point"""
    data: Any  # Text string
    embedding: Any

@dataclass
class DataCluster:
    """Cluster of data points with similar facet values"""
    facet: FacetMetadata  # Changed from Facet to FacetMetadata
    summary: str
    name: str
    children: Optional[List['DataCluster']] = None
    parent: Optional['DataCluster'] = None
    indices: Optional[np.ndarray] = None

    def __hash__(self):
        return hash((self.summary, self.name, self.facet))

    def __eq__(self, other):
        if not isinstance(other, DataCluster):
            return False
        return self.summary == other.summary and self.name == other.name and self.facet == other.facet

# Backwards compatibility alias for old pickle files
ConversationCluster = DataCluster

@dataclass
class OpenClioConfig:
    """
    Configuration for a run of openclio.

    Quick start guide:
    - llmBatchSize: Adjust based on API rate limits (default: 1000, lower for Vertex AI)
    - maxTextChars: Maximum characters per text block (default: 32000 - suitable for long system prompts)
    - llmExtraInferenceArgs: Sampling parameters passed to LLM (temperature, top_p, etc.)

    Hierarchy tuning:
    - minTopLevelSize=5: Stop building hierarchy when reaching this many top-level clusters
    - nBaseClustersFunc=lambda n: n//10: Number of base clusters (10 = avg 10 items per cluster)
    - nDesiredHigherLevelNamesPerClusterFunc=lambda n: n//3: Branching factor (3 = ~1/3 reduction per level)

    The default values work well for most use cases and will adapt to your data size.
    """
    # 
    

    # The rest of the parameters here are fairly reasonable and you can leave at default settings, feel free to consult the comments by them if you'd like to know what they do.

    ### General params
    seed: int = 27 #: Useful so runs are deterministic
    verbose: bool = True #: Whether to print intermediate outputs and progress bars
    llmBatchSize: int = 50 #: Batch size for LLM calls. Lower for API rate limits (Vertex AI), higher for local models
    embedBatchSize: int = 100 #: Batch size for embedding. Reduced to 100 to prevent memory issues
    dedupData: bool = True #: Whether to deduplicate the data. Important to avoid large clusters of identical values
    dedupKeyFunc: Optional[Callable[[Any], Any]] = None #: Function for comparing data equivalence. If None, uses str() for text comparison

    ### Performance params
    numWorkers: int = -1 #: Number of worker threads for parallel operations (-1 = all cores)
    useLowMemoryMode: bool = False #: If True, use less RAM but slower (set False for speed)
    vertexRateLimitPerMin: int = 15 #: Vertex AI rate limit (requests per minute). Very conservative to avoid crashes. Increase if stable
    maxParallelLLMCalls: int = 1 #: Maximum number of parallel LLM API calls. Set to 1 for stability, increase if stable
    minDelayBetweenRequests: float = 2.0 #: Minimum delay (seconds) between API requests. Helps avoid bursts and crashes

    ### Generate Base Clusters params
    getDataFunc: Callable[[Any], Any] = lambda data: data #: Function to extract text from data point. Identity function by default (data is already text)
    maxTextChars: int = 32000 #: Max characters per text block. Texts will be truncated to this length to avoid overwhelming model context
    nBaseClustersFunc: Callable[[int], int] = lambda n: n//10 # Number of base clusters to start with, depends on data size. If unspecified, will set to lambda n: n//10
    autoSelectK: bool = False #: If True, automatically select optimal K using Calinski-Harabasz score instead of nBaseClustersFunc
    kSearchRange: Tuple[int, int] = (5, 50) #: Range of K values to try when autoSelectK=True. Format: (min_k, max_k)
    kSearchStep: int = 1 #: Step size when searching K range (e.g., step=5 tries K=5,10,15,...)
    maxPointsToSampleInsideCluster: int = 10 #: Number of points we sample inside the cluster, when determining base cluster names and summaries. More will make longer contexts but give the llm more information
    maxPointsToSampleOutsideCluster: int = 10 #: Number of points we sample outside the cluster (as examples of stuff closest to, but *not* in the cluster), when determining base cluster names and summaries. More will make longer contexts but give the llm more information

    ### Hierarchy params
    minTopLevelSize: int = 5 #: Once we've reached this many or less clusters, we have reached the top, stop going higher
    # neighborhoods
    nAverageClustersPerNeighborhood: Callable[[int], int] = lambda n: max(1, n//10) #: Function that tells us how many number of clusters to have per neighborhood, on average. From G.7, "average number of clusters per neighborhood is 40", so default is lambda n: max(1, n//40) But that's too many for a small model, lets do smaller like 10
    nSamplesOutsideNeighborhood: int = 5 #: How many samples from outside the k-means cluster to add to each neighborhood. From G.7, "Including the nearest clusters beyond the neighborhood ensures that clusters (or groups of clusters on the boundary between neighborhoods are neither overcounted nor undercounted)." 
    # get names from neighborhoods
    nDesiredHigherLevelNamesPerClusterFunc: Callable[[int], int] = lambda n: max(1, n//3) #: Given number of elements in our neighborhood, return how many higher level cluster names we should have. The default of lambda n: max(1, n//3) will result in there being rougly half the amount of cluster names at each level in the hierarchy.
    # dedup (none)
    # assign lower level to higher level categories - no sampling, just call once
    # rename once we see what's in the categories
    maxChildrenForRenaming: int = 10 #: Maximum number of children in category to display when deciding what to name it, more will make longer prompt but give more accurate classification

    ### Extra Params
    llmExtraInferenceArgs: Dict[str, Any] = field(default_factory=lambda: {
        "max_tokens": 1000,
         # default qwen non-thinking sampling params
        'temperature': 0.7,
        'top_p': 0.8,
        'top_k': 20,
        'min_p': 0.0,
    }) #: Extra parameters to pass to LLM generation (e.g., temperature, top_p, etc.)

    kmeansArgs: Dict[str, Any] = field(default_factory = lambda: {
        "approximate": True, # since we only have 10 elements per term, by default this would take many hours, this speeds it up a lot
        "verbose": True,
    })

    ### Umap settings
    dataToStrFunc: Optional[Callable[[Any], str]] = None #: Function to convert data points into strings to be embedded for 2D umap plot. If None, uses str() for text data

@dataclass
class OpenClioResults:
    """Results from running Clio analysis on text data"""
    facets: List[FacetMetadata]  # Changed from List[Facet]
    facetValues: List[DataPointFacetData]
    facetValuesEmbeddings: List[Optional[EmbeddingArray]]
    baseClusters: List[Optional[List[DataCluster]]]
    rootClusters: List[Optional[List[DataCluster]]]
    data: List[Any]  # List of text strings
    umap: List[Optional[npt.NDArray[np.float32]]]
    cfg: OpenClioConfig