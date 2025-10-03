from typing import List, Any, Optional
from sentence_transformers import SentenceTransformer
from .opencliotypes import OpenClioConfig, EmbeddingArray
from .utils import runBatched
from numpy import typing as npt
import umap
import numpy as np


def computeUmapHelper(embeddingArr: EmbeddingArray, verbose: bool = False):
    # unique=True is very important otherwise it gets stuck
    umapModel = umap.UMAP(
        n_components=2,
        unique=True,
        verbose=verbose,
        n_jobs=-1,  # Use all CPU cores
        low_memory=False  # Faster but uses more RAM - disable for speed
    )
    return umapModel.fit_transform(embeddingArr)

def computeUmap(data: List[Any], facetValuesEmbeddings: List[Optional[EmbeddingArray]], embeddingModel: SentenceTransformer, cfg: OpenClioConfig):
    cfg.print("Running umap on facet values")
    resUmaps = [(computeUmapHelper(embeddingArr, verbose=cfg.verbose) if embeddingArr is not None else None) for embeddingArr in facetValuesEmbeddings]
    cfg.print("Embedding data for umap")
    # fallback to default data to string
    dataToStringFunc = cfg.dataToStrFunc if cfg.dataToStrFunc is not None else lambda text: text

    def processBatchFunc(batchOfTextInputs: List[str]) -> List[npt.NDArray[np.float32]]:
        # Use GPU/MPS if available
        import torch
        device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
        embedded = embeddingModel.encode(
            batchOfTextInputs,
            show_progress_bar=False,
            device=device,
            batch_size=len(batchOfTextInputs)  # Process entire batch at once
        )
        return [embedded[i] for i in range(len(batchOfTextInputs))]

    embeddedData = np.stack(runBatched(data,
                    getInputs=lambda dataPoint: dataToStringFunc(dataPoint),
                    processBatch=processBatchFunc,
                    processOutput=lambda dataPoint, inputs, emb: emb,
                    batchSize=cfg.embedBatchSize,
                    verbose=cfg.verbose))

    cfg.print("Running umap on embedded data")
    dataUmap = computeUmapHelper(embeddingArr=embeddedData, verbose=cfg.verbose)
    # last index is the umap over embeddings of data (instead of facet values)
    resUmaps.append(dataUmap)
    return resUmaps
