from dataclasses import dataclass, field
import pandas as pd
from typing import Any, Union, Tuple, Optional, Callable, Dict, List, TypeAlias, Type
import traceback
import faiss
import functools
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn import preprocessing
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import cdist
from collections import defaultdict
from numpy import typing as npt
import json
import torch
import os
import re
import random
import cloudpickle
from pathlib import Path
import time
from pydantic import BaseModel, Field, create_model
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import gc

from .prompts import (
    getFacetClusterNamePrompt,
    getNeighborhoodClusterNamesPrompt,
    getDeduplicateClusterNamesPrompt,
    getAssignToHighLevelClusterPrompt,
    getRenamingHigherLevelClusterPrompt,
)
from .utils import flatten, unflatten, runBatched, dedup
from .opencliotypes import FacetMetadata, FacetValue, DataPointFacetData, DataPointEmbedding, DataCluster, OpenClioConfig, OpenClioResults, EmbeddingArray, shouldMakeFacetClusters
from .faissKMeans import FaissKMeans
from .writeOutput import computeUmap

# System prompt facets as Pydantic model
class SystemPromptFacets(BaseModel):
    """Facets for analyzing AI agent system prompts"""
    primary_purpose: str = Field(description="The primary purpose or role of the AI agent. Be clear and concise, 1-2 sentences.")
    domain: str = Field(description="The domain or subject area the AI agent operates in (e.g., healthcare, finance, education, etc.). 1-2 sentences.")
    key_capabilities: str = Field(description="The key capabilities or features emphasized. List the most important ones. 1-2 sentences.")
    interaction_style: str = Field(description="The interaction style or personality the AI agent adopts (e.g., professional, casual, empathetic). 1-2 sentences.")

# Pydantic models for structured cluster naming
class ClusterNameAndSummary(BaseModel):
    """Structured output for cluster naming"""
    summary: str = Field(description="A clear, precise, two-sentence description of the cluster")
    name: str = Field(description="A short, specific name for the cluster (at most 10 words)")

# Metadata for system prompt facets (defines which should have cluster hierarchies)
systemPromptFacetMetadata = {
    "primary_purpose": FacetMetadata(
        name="Primary Purpose",
        summaryCriteria="The cluster name should be a clear phrase describing the agent's main function or role, such as 'Customer support chatbot' or 'Code generation assistant'."
    ),
    "domain": FacetMetadata(
        name="Domain",
        summaryCriteria="The cluster name should describe the domain or industry, such as 'Healthcare and medical advice' or 'Software development'."
    ),
    "key_capabilities": FacetMetadata(
        name="Key Capabilities",
        summaryCriteria="The cluster name should summarize the main capabilities, such as 'Multi-language translation and cultural adaptation' or 'Data analysis and visualization'."
    ),
    "interaction_style": FacetMetadata(
        name="Interaction Style",
        summaryCriteria="The cluster name should describe the tone and manner, such as 'Professional and concise' or 'Friendly and conversational'."
    ),
}


def runClio(
    facetSchema: Type[BaseModel],
    facetMetadata: Dict[str, FacetMetadata],
    embeddingModel: SentenceTransformer,
    data: List[str],
    outputDirectory: str,
    project_id: str,
    model_name: str = "gemini-1.5-flash-002",
    location: str = "us-central1",
    displayWidget: bool = False,
    cfg: OpenClioConfig = None,
    **kwargs
) -> OpenClioResults:
    """
    Runs the Clio algorithm on text data using Vertex AI and embeddings.

    Keyword arguments:
    facetSchema -- Pydantic BaseModel defining all facets to extract (e.g., SystemPromptFacets)
    facetMetadata -- Dict mapping field names to FacetMetadata (defines which facets get cluster hierarchies)
    embeddingModel -- SentenceTransformer for clustering
    data -- List of text strings to analyze
    outputDirectory -- Where to store checkpoints/outputs
    project_id -- GCP project ID for Vertex AI
    model_name -- Vertex AI model (default: gemini-1.5-flash-002)
    location -- GCP region (default: us-central1)
    displayWidget -- (Default: False) Show interactive widget in Jupyter/Colab
    cfg -- Optional OpenClioConfig for advanced settings

    Returns:
    OpenClioResults object (or ClioWidget if displayWidget=True)
    """

    # make the output directory to store checkpoints if it does not exist
    Path(outputDirectory).mkdir(parents=True, exist_ok=True)

    if cfg is None:
        cfg = OpenClioConfig(**kwargs)
    else:
        for k,v in kwargs.items():
            if hasattr(cfg, k):
                cfg.k = v
            else:
                raise ValueError(f"Unknown OpenClioConfig key {k} with value {v}")

    cfg.print = print if cfg.verbose else lambda *a, **b: None
    cfg.kmeansArgs['verbose'] = cfg.verbose

    # Initialize Vertex AI client
    try:
        import google.genai as genai
        vertex_client = genai.Client(
            vertexai=True,
            project=project_id,
            location=location
        )
        cfg.print(f"Initialized Vertex AI client for project {project_id}")
    except ImportError:
        raise ImportError("Vertex AI dependencies not installed. Install with: pip install google-genai")
    except Exception as e:
        raise RuntimeError(f"Failed to initialize Vertex AI: {e}")

    # Convert facetMetadata dict to list in consistent order
    facet_field_names = list(facetSchema.model_fields.keys())
    facets = [facetMetadata[field_name] for field_name in facet_field_names]

    dependencyModified = False
    checkpoint_counter = [0]  # Use list to allow modification in nested function

    def runIfNotExist(path, runFunc, dependencyModified):
        # Add counter prefix for ordering
        counter_str = f"{checkpoint_counter[0]:04d}"
        prefixed_path = f"{counter_str}_{path}"
        fullPath = os.path.join(outputDirectory, prefixed_path)

        # Also check old path without prefix for backward compatibility
        oldPath = os.path.join(outputDirectory, path)

        # Try to load from prefixed path first, then old path
        if os.path.exists(fullPath) and not dependencyModified:
            try:
                cfg.print(f"Resuming from {fullPath}...")
                with open(fullPath, "rb") as f:
                    result = cloudpickle.load(f)
                cfg.print(f"Resumed from {fullPath}")
                checkpoint_counter[0] += 1
                return result, False
            except:
                cfg.print(f"Failed to load from path {fullPath}, ignoring")
                cfg.print(traceback.format_exc())
        elif os.path.exists(oldPath) and not dependencyModified:
            try:
                cfg.print(f"Resuming from old checkpoint {oldPath}...")
                with open(oldPath, "rb") as f:
                    result = cloudpickle.load(f)
                cfg.print(f"Resumed from {oldPath}")
                checkpoint_counter[0] += 1
                return result, False
            except:
                cfg.print(f"Failed to load from path {oldPath}, ignoring")
                cfg.print(traceback.format_exc())

        # Run the function and save with prefixed name
        res = runFunc()
        with open(fullPath, "wb") as f:
            cloudpickle.dump(res, f)
        cfg.print(f"Saved checkpoint: {fullPath}")
        checkpoint_counter[0] += 1
        return res, True
    
    def getResults():
        nonlocal data
        nonlocal dependencyModified

        # Filter out None values from data
        cfg.print("Filtering data")
        original_count = len(data)
        data = [d for d in data if d is not None]
        filtered_count = original_count - len(data)
        if filtered_count > 0:
            cfg.print(f"Filtered out {filtered_count} None values from data")

        cfg.print("Deduping data")
        if cfg.dedupData:
            dedupKeyFunc = cfg.dedupKeyFunc
            if dedupKeyFunc is None:
                # For text data, use string comparison
                cfg.print("Using text dedup (string comparison)")
                dedupKeyFunc = lambda x: x.strip().lower() if x else ""
            data, dependencyModified = runIfNotExist("dedupedData.pkl", lambda:
                dedup(data, dedupKeyFunc=dedupKeyFunc, batchSize=cfg.llmBatchSize, verbose=cfg.verbose),
                dependencyModified=dependencyModified
            )
        

        cfg.print("Getting facet values")
        setSeed(cfg.seed) # doing this before each function call helps ensure reproducability if they resume
        facetValues, dependencyModified = \
            runIfNotExist("facetValues.pkl", lambda:
                getFacetValues(
                    facets=facets,
                    facetSchema=facetSchema,
                    vertex_client=vertex_client,
                    model_name=model_name,
                    data=data,
                    cfg=cfg
                ),
                dependencyModified=dependencyModified
            )
        
        cfg.print("Getting facet value embeddings")
        setSeed(cfg.seed)
        facetValuesEmbeddings, dependencyModified = \
            runIfNotExist("facetValuesEmbeddings.pkl", lambda:
                getFacetValuesEmbeddings(
                    facets=facets,
                    facetValues=facetValues,
                    embeddingModel=embeddingModel,
                    cfg=cfg
                ),
                dependencyModified=dependencyModified
            )
        
        cfg.print("Getting base clusters")
        setSeed(cfg.seed)
        baseClusters, dependencyModified = \
            runIfNotExist("baseClusters.pkl", lambda:
                getBaseClusters(
                    facets=facets,
                    vertex_client=vertex_client,
                    model_name=model_name,
                    embeddingModel=embeddingModel,
                    facetValues=facetValues,
                    facetValuesEmbeddings=facetValuesEmbeddings,
                    cfg=cfg,
                    runIfNotExist=runIfNotExist,
                    dependencyModified=dependencyModified,
                ),
                dependencyModified=dependencyModified
            )

        cfg.print("Getting higher level clusters")
        setSeed(cfg.seed)
        rootClusters, dependencyModified = \
            runIfNotExist("rootClusters.pkl", lambda:
                getHierarchy(
                    facets=facets,
                    vertex_client=vertex_client,
                    model_name=model_name,
                    embeddingModel=embeddingModel,
                    baseClusters=baseClusters,
                    cfg=cfg
                ),
                dependencyModified=dependencyModified
            )
        
        cfg.print("Running umap on data")
        setSeed(cfg.seed)
        umap, dependencyModified = \
            runIfNotExist("umapResults.pkl", lambda:
                computeUmap(
                    data=data,
                    facetValuesEmbeddings=facetValuesEmbeddings,
                    embeddingModel=embeddingModel,
                    cfg=cfg
                ),
                dependencyModified=dependencyModified
            )
        
        cfg.print("Saving results")
        return OpenClioResults(
            facets=facets,
            facetValues=facetValues,
            facetValuesEmbeddings=facetValuesEmbeddings,
            baseClusters=baseClusters,
            rootClusters=rootClusters,
            data=data,
            umap=umap,
            cfg=cfg
        )

    output, dependencyModified = runIfNotExist("results.pkl", getResults,
        dependencyModified=dependencyModified
    )

    # Display widget
    if displayWidget:
        from .widget import ClioWidget
        widget = ClioWidget(output)
        # Store results in widget so user can access them
        widget.results = output
        # Return widget - Jupyter will auto-display it (like embedding-atlas does)
        return widget

    return output



def getNeighborhoods(
    facetStrValues: List[str],
    embeddingModel: SentenceTransformer,
    cfg: OpenClioConfig,
    nSamplesOutsideNeighborhood: int) -> Tuple[FaissKMeans, List[List[int]]]:
    """
    Embed map(valueMap, facetValues) into 
    cfg.nAverageClustersPerNeighborhood(len(facetStrValues)) clusters
    using kmeans,
    then add cfg.nSamplesOutsideNeighborhood extra closest samples to each cluster
    return (kmeans, [[neighborhood0...], [neighborhood1...]])
    """
    def processBatchFunc(batchOfTextInputs):
        embedded = embeddingModel.encode(batchOfTextInputs, show_progress_bar=False)
        return [embedded[i] for i in range(len(batchOfTextInputs))]
    
    facetClusterEmbeddings = np.stack(runBatched(facetStrValues,
                                   getInputs=lambda facetStrValue: facetStrValue,
                                   processBatch=processBatchFunc,
                                   processOutput=lambda facetValue, facetValuePrompt, outputEmbeddings: outputEmbeddings,
                                   batchSize=cfg.embedBatchSize,
                                   verbose=cfg.verbose))
    facetNeighborhoods = []
    numValues = len(facetStrValues)
    # in the paper this is numClusters // 40
    k = cfg.nAverageClustersPerNeighborhood(numValues)
    kmeans = FaissKMeans(n_clusters=min(numValues, k), random_state=cfg.seed, **cfg.kmeansArgs)
    # we have to normalize for this to be cosine similarity
    kmeans.fit(preprocessing.normalize(facetClusterEmbeddings))
    distances = cdist(facetClusterEmbeddings, kmeans.cluster_centers_)
    for clusterIndex in range(len(kmeans.cluster_centers_)):
        # Get points belonging to this cluster
        clusterPointsIndices = np.where(kmeans.labels_ == clusterIndex)[0]
        # Get closest points not in this cluster
        outsideClusterIndices = np.where(kmeans.labels_ != clusterIndex)[0]
        closestPointsOutsideClusterIndices = outsideClusterIndices[np.argsort(distances[kmeans.labels_ != clusterIndex, clusterIndex])]
        # From G.7:
        # "Including the nearest clusters beyond the neighborhood ensures that clusters (or groups of clusters)
        #  on the boundary between neighborhoods are neither overcounted nor undercounted"
        clusterIndicesInNeighborhood = list(clusterPointsIndices) + list(closestPointsOutsideClusterIndices[:nSamplesOutsideNeighborhood])
        facetNeighborhoods.append(clusterIndicesInNeighborhood)

    return kmeans, facetNeighborhoods

def getHierarchy(
    facets: List[FacetMetadata],
    vertex_client: Any,
    model_name: str,
    embeddingModel: SentenceTransformer,
    baseClusters: List[Optional[List[DataCluster]]],
    cfg: OpenClioConfig) -> List[Optional[List[DataCluster]]]:
    """
    Extracts a hierarchy of labels, starting with baseClusters at the lowest level.
    Returns a list one element per facet.
    """
    seed = cfg.seed

    def processBatchFuncLLM(prompts: List[str]) -> List[str]:
        nonlocal seed
        seed += 1
        results = []
        for prompt in prompts:
            # Enforce minimum delay
            if len(results) > 0:
                time.sleep(cfg.minDelayBetweenRequests)

            # Retry with exponential backoff on rate limits
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = vertex_client.models.generate_content(
                        model=model_name,
                        contents=[prompt],
                        config={
                            "max_output_tokens": cfg.llmExtraInferenceArgs.get("max_tokens", 1000),
                            "temperature": cfg.llmExtraInferenceArgs.get("temperature", 0.7),
                            "top_p": cfg.llmExtraInferenceArgs.get("top_p", 0.8),
                            "top_k": cfg.llmExtraInferenceArgs.get("top_k", 40),
                        }
                    )
                    results.append(response.text)
                    break  # Success, move to next prompt

                except Exception as e:
                    error_msg = str(e).lower()
                    # Check if it's a rate limit error
                    if "429" in error_msg or "quota" in error_msg or "rate" in error_msg:
                        if attempt < max_retries - 1:
                            wait_time = (2 ** attempt) * 2  # Exponential backoff: 2s, 4s, 8s
                            cfg.print(f"Rate limit hit, waiting {wait_time}s before retry {attempt + 1}/{max_retries}")
                            time.sleep(wait_time)
                            continue
                    cfg.print(f"Error in LLM call: {e}")
                    results.append("")
                    break  # Don't retry non-rate-limit errors
        return results

    def processBatchFuncLLMWithStructuredNaming(prompts: List[str]) -> List[ClusterNameAndSummary]:
        """Version of processBatchFuncLLM that uses structured outputs for cluster naming"""
        nonlocal seed
        seed += 1
        results = []
        for prompt_idx, prompt in enumerate(prompts):
            # Enforce minimum delay
            if len(results) > 0:
                time.sleep(cfg.minDelayBetweenRequests)

            # Retry with exponential backoff on rate limits
            max_retries = 3
            success = False
            for attempt in range(max_retries):
                try:
                    # Use structured outputs with Pydantic class
                    cfg.print(f"Making API call for prompt {prompt_idx + 1} (attempt {attempt + 1}/{max_retries})...")

                    # Log prompt length for debugging
                    cfg.print(f"Prompt length: {len(prompt)} chars")

                    response = vertex_client.models.generate_content(
                        model=model_name,
                        contents=[prompt],
                        config={
                            "response_mime_type": "application/json",
                            "response_schema": ClusterNameAndSummary,
                            "max_output_tokens": 8192,  # High limit needed for gemini-2.5-flash
                            "temperature": 0.0,  # Deterministic
                        }
                    )

                    cfg.print(f"API call successful for prompt {prompt_idx + 1}, parsing response...")
                    # Check if response has parsed attribute
                    if hasattr(response, 'parsed') and response.parsed is not None:
                        cfg.print(f"Response parsed successfully: {response.parsed.name}")
                        results.append(response.parsed)
                        success = True
                        break
                    else:
                        # Try getting candidates and parts
                        cfg.print(f"No parsed attribute, trying candidates...")
                        import json
                        try:
                            # Try to get the full response text from candidates
                            if hasattr(response, 'candidates') and len(response.candidates) > 0:
                                candidate = response.candidates[0]

                                # Check finish reason
                                if hasattr(candidate, 'finish_reason'):
                                    cfg.print(f"Finish reason: {candidate.finish_reason}")

                                # Check if response was blocked by safety filters
                                if hasattr(candidate, 'finish_reason') and candidate.finish_reason != 1:  # 1 = STOP (normal completion)
                                    cfg.print(f"Response did not complete normally. Finish reason code: {candidate.finish_reason}")
                                    # Return a default result instead of retrying
                                    results.append(ClusterNameAndSummary(summary="Cluster description", name="Music Genre Cluster"))
                                    success = True
                                    break

                                if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                                    full_text = ''.join([part.text for part in candidate.content.parts])
                                    cfg.print(f"Got full text from candidates (length={len(full_text)}): {full_text[:200]}...")
                                    if len(full_text) > 50:
                                        cfg.print(f"Full text ends with: ...{full_text[-50:]}")
                                    json_response = json.loads(full_text)
                                    parsed = ClusterNameAndSummary(**json_response)
                                    cfg.print(f"Manual parsing successful: {parsed.name}")
                                    results.append(parsed)
                                    success = True
                                    break

                            # Fallback to response.text
                            if hasattr(response, 'text'):
                                cfg.print(f"Trying response.text (len={len(response.text)}): {response.text[:200]}...")
                                json_response = json.loads(response.text)
                                parsed = ClusterNameAndSummary(**json_response)
                                cfg.print(f"Manual parsing successful: {parsed.name}")
                                results.append(parsed)
                                success = True
                                break

                            cfg.print(f"Could not extract text from response. Type: {type(response)}")
                            results.append(ClusterNameAndSummary(summary="Error in naming", name="Error"))
                            success = True
                            break
                        except json.JSONDecodeError as je:
                            cfg.print(f"JSON decode error: {je}. Full text: {full_text if 'full_text' in locals() else 'N/A'}")
                            # If JSON is incomplete, retry this attempt
                            if attempt < max_retries - 1:
                                cfg.print(f"Retrying due to malformed JSON...")
                                time.sleep(2)
                                continue
                            results.append(ClusterNameAndSummary(summary="Error in naming", name="Error"))
                            success = True
                            break

                except Exception as e:
                    error_msg = str(e).lower()
                    cfg.print(f"Exception for prompt {prompt_idx + 1} (attempt {attempt + 1}/{max_retries}): {e}")
                    # Check if it's a rate limit error
                    if "429" in error_msg or "quota" in error_msg or "rate" in error_msg:
                        if attempt < max_retries - 1:
                            wait_time = (2 ** attempt) * 2  # Exponential backoff: 2s, 4s, 8s
                            cfg.print(f"Rate limit hit, waiting {wait_time}s before retry {attempt + 1}/{max_retries}")
                            time.sleep(wait_time)
                            continue
                    cfg.print(f"Error in cluster naming for prompt {prompt_idx + 1}: {e}")
                    results.append(ClusterNameAndSummary(summary="Error in naming", name="Error"))
                    success = True
                    break  # Don't retry non-rate-limit errors

            # If all retries failed, add error result
            if not success:
                cfg.print(f"All retries exhausted for prompt {prompt_idx}")
                results.append(ClusterNameAndSummary(summary="Error in naming", name="Error"))
        return results 
    
    topLevelParents = []
    for facetI, facet in enumerate(facets):
        if not shouldMakeFacetClusters(facet):
            topLevelParents.append(None)
            continue

        curLevelFacetClusters: List[DataCluster] = baseClusters[facetI]
        level = 0
        while len(curLevelFacetClusters) > cfg.minTopLevelSize:

            #### Get embeddings for clusters ####

            Sources: TypeAlias = List[int]

            cfg.print(f"facet {facet} level {level}")

            cfg.print("getting category neighborhoods")
            kmeans, facetNeighborhoods = getNeighborhoods(
                        facetStrValues=list(map(lambda cluster: f"{cluster.name}\n{cluster.summary}", curLevelFacetClusters)),
                        embeddingModel=embeddingModel,
                        cfg=cfg,
                        nSamplesOutsideNeighborhood=cfg.nSamplesOutsideNeighborhood)

            #### Get higher level category names ####

            cfg.print("getting higher level category names")
            def getInputsFunc(clusterIndicesInNeighborhood: Sources) -> str:
                clustersInNeighborhood = [curLevelFacetClusters[i] for i in clusterIndicesInNeighborhood]
                clusterNamePrompts = []
                for _ in range(2):
                    random.shuffle(clustersInNeighborhood)# shuffle ordering
                    clusterNamePrompts.append(getNeighborhoodClusterNamesPrompt(facet, clustersInNeighborhood, cfg.nDesiredHigherLevelNamesPerClusterFunc(len(clustersInNeighborhood))))
                return clusterNamePrompts                    
            
            def processOutputFunc(clusterIndicesInNeighborhood: Sources, clusterNamePrompts: str, clusterNamesOutputs: str) -> List[Tuple[str, Sources]]:
                # also store where it came from
                # also remove punctuation
                for clusterNamesOutput in clusterNamesOutputs:
                    clusterNames = extractAnswerNumberedList(clusterNamesOutput)
                    if len(clusterNames) > 0:
                        return [(removePunctuation(clusterName).strip(), clusterIndicesInNeighborhood) for clusterName in clusterNames]
                '''
                # don't extract partial because stuck in loops tends to be low quality
                cfg.print("Failed, extracting partial")
                cfg.print(clusterNamePrompts[0])
                cfg.print(clusterNamesOutputs[0])
                for clusterNamesOutput in clusterNamesOutputs:
                    clusterNames = extractAnswerNumberedList(clusterNamesOutput, ignoreNoTrailing=True)
                    # cut it off at desired because it probably got stuck in a loop and made lots of unhelpful ones
                    desired = max(1, cfg.nDesiredHigherLevelNamesPerClusterFunc(len(clusterIndicesInNeighborhood)))
                    clusterNames = clusterNames[:desired]
                    if len(clusterNames) != 0: # success at partial parsing
                        break
                '''
                cfg.print("Failed to extract any names for cluster: manually retrying")
                clustersInNeighborhood = [curLevelFacetClusters[i] for i in clusterIndicesInNeighborhood]
                # shuffle ordering
                while True:
                    random.shuffle(clustersInNeighborhood)
                    prompt = getNeighborhoodClusterNamesPrompt(facet, clustersInNeighborhood, cfg.nDesiredHigherLevelNamesPerClusterFunc(len(clustersInNeighborhood)))
                    cfg.print(prompt)
                    nonlocal seed
                    seed += 1
                    clusterNamesOutput = processBatchFuncLLM([prompt])[0]
                    clusterNames = extractAnswerNumberedList(clusterNamesOutput, ignoreNoTrailing=True)
                    if len(clusterNames) != 0:
                        cfg.print("Success at manual retry")
                        return [(removePunctuation(clusterName).strip(), clusterIndicesInNeighborhood) for clusterName in clusterNames]
                    else:
                        cfg.print("Failed manual retry, trying again")
                        cfg.print(clusterNamesOutput)
            
            def dedupAndMergeSources(values: List[Tuple[str, Sources]]) -> List[Tuple[str, Sources]]:
                resultValues = defaultdict(lambda: set())
                for (value, sources) in values:
                    resultValues[value] |= set(sources)
                return sorted(list(resultValues.items()))
            
            # dedup exact named matches
            higherCategories = dedupAndMergeSources(
                    flatten(
                        runBatched(facetNeighborhoods,
                            getInputs=getInputsFunc,
                            processBatch=processBatchFuncLLM,
                            processOutput=processOutputFunc,
                            batchSize=cfg.llmBatchSize,
                            verbose=cfg.verbose)
                    )
            )
            cfg.print(f"Got {len(higherCategories)} potential higher categories")

            #### Dedup higher categories ####
            
            cfg.print("getting higher level category neighborhoods")
            kmeans, facetNeighborhoods = getNeighborhoods(
                        facetStrValues=[name for (name, sources) in higherCategories],
                        embeddingModel=embeddingModel,
                        cfg=cfg,
                        nSamplesOutsideNeighborhood=0) # don't grab extra outside neighborhood or our "dedup" will result in more categories, not less (as the overlap leads to double counting worded differently so they don't get deduped)
            
            cfg.print("deduping higher level categories")
            def getInputsFunc(higherCategoryIndicesInNeighborhoods: List[Tuple[str, Sources]]) -> str:
                # 0 is value, 1 is sources
                higherCategoriesInNeighborhood = [higherCategories[i][0] for i in higherCategoryIndicesInNeighborhoods]
                targetAmount =  max(1, len(higherCategoriesInNeighborhood)//2) # aim for -1 (arbitrary), but prompt lets it do more or less as needed
                if len(higherCategoriesInNeighborhood) == 2:
                    targetAmount = 2 # for only two, it'll mangle the categories if we ask it to dedup them into one, so don't do that
                return getDeduplicateClusterNamesPrompt(facet, higherCategoriesInNeighborhood, targetAmount)
            
            def processOutputFunc(
                    higherCategoryIndicesInNeighborhoods: List[Tuple[str, Sources]],
                    higherCategoryDedupPrompt: str,
                    higherCategoryDedupOutput: str) -> List[Tuple[str, Sources]]:
                # get sources in terms of original categories (union over all the different higher category inputs to this dedup)
                allSources = set()
                higherCategoriesInNeighborhood = [higherCategories[i] for i in higherCategoryIndicesInNeighborhoods]
                for (higherCategory, higherCategorySources) in higherCategoriesInNeighborhood:
                    allSources |= set(higherCategorySources)
                allSources = sorted(list(allSources))
                extractedOptions = extractAnswerNumberedList(higherCategoryDedupOutput)
                # fall back to dedup based on embedding (usually this means model got stuck in a loop, and it's better we ignore its outputs)
                if len(extractedOptions) == 0:
                    # no dedup extracted, falling back to dedup based on embedding 
                    extractedOptions = [value for value, _ in deduplicateByEmbeddingsAndMergeSources(valuesAndSources=higherCategoriesInNeighborhood, embeddingModel=embeddingModel, tau=0.1)]
                return [(removePunctuation(output).strip(), allSources) for output in extractedOptions]


            
            # todo: size 1 or 2 clusters, just ignore them (size 2 maybe set desired to 2? unless very high overlap in embed? idk)
            dedupedCategories = dedupAndMergeSources(
                    flatten(
                        runBatched(facetNeighborhoods,
                            getInputs=getInputsFunc,
                            processBatch=processBatchFuncLLM,
                            processOutput=processOutputFunc,
                            batchSize=cfg.llmBatchSize,
                            verbose=cfg.verbose)
                    )
            )

            # also just dedup by embeddings as an extra check, as the deduplicateByEmbeddings above can add too many extras because categories overlap
            dedupedCategories = deduplicateByEmbeddingsAndMergeSources(
                valuesAndSources=dedupedCategories,
                embeddingModel=embeddingModel,
                tau=0.1,
            )
            
            cfg.print(f"Got {len(dedupedCategories)} deduped higher categories")

            #### Assign to new best fit higher-level cluster ####

            # (they didn't specify how to choose what to put here, but I figure just tracking where parents came from and using all those that might have come from x should work fine)
            cfg.print("Assigning to best fit higher-level clusters")
            baseClusterPotentialHigherLevelClusters: List[List[str]] = [[] for _ in curLevelFacetClusters]
            for category, sources in dedupedCategories:
                for sourceI in sources:
                    baseClusterPotentialHigherLevelClusters[sourceI].append(category)
            
            def getInputsFunc(facetClusterData: Tuple[DataCluster, List[str]]) -> str:
                facetCluster, potentialHigherLevelClusters = facetClusterData
                # No sampling - just generate one prompt
                return getAssignToHighLevelClusterPrompt(clusterToAssign=facetCluster, higherLevelClusters=potentialHigherLevelClusters)

            # name and summary will be generated later
            parents: Dict[str, DataCluster] = dict(
                [
                    (categoryName.lower().strip(), DataCluster(facet=facet, name=categoryName, summary=""))
                    for (categoryName, categorySources) in dedupedCategories
                ]
            )

            def processOutputFunc(
                    facetClusterData: Tuple[DataCluster, List[str]],
                    assignToHigherCategoryPrompt: str,
                    assignToHigherCategoryOutput: str
                ):
                facetCluster, potentialHigherLevelClusters = facetClusterData
                # No sampling - just use single output
                foundOutput, outputValue = extractTagValue(assignToHigherCategoryOutput, "answer")
                # remove cluster and punctuation if it added it
                outputValue = removePunctuation(outputValue.replace("<cluster>", "").replace("</cluster>", "").strip()).strip()

                if not foundOutput or not outputValue:
                    # failed to extract cluster from llm, fall back to embedding of the cluster
                    outputValue = facetCluster.summary + "\n" + facetCluster.name

                if len(potentialHigherLevelClusters) == 0:
                    cfg.print("got empty potentialHigherLevelClusters??")
                    cfg.print(outputValue)
                    cfg.print(potentialHigherLevelClusters)

                # Find best match in potentialHigherLevelClusters using embeddings
                bestHigherLevelClusterAssignedTo = bestMatch(outputValue, potentialHigherLevelClusters, embeddingModel)
                parent = parents[bestHigherLevelClusterAssignedTo.lower().strip()]
                if parent.children is None:
                    parent.children = []
                parent.children.append(facetCluster)
                facetCluster.parent = parent
                return None

            runBatched(list(zip(curLevelFacetClusters, baseClusterPotentialHigherLevelClusters)),
                getInputs=getInputsFunc,
                processBatch=processBatchFuncLLM,
                processOutput=processOutputFunc,
                batchSize=cfg.llmBatchSize,
                verbose=cfg.verbose)

            # remove any parents that didn't have any children assigned
            for parentKey, parentValue in list(parents.items()):
                if parentValue.children is None or len(parentValue.children) == 0:
                    del parents[parentKey]

            #### Rename categories based on which children they were given ####

            cfg.print("Renaming categories based on children")
            def getInputsFunc(parent: DataCluster) -> str:
                # No sampling - just use children as-is
                return getRenamingHigherLevelClusterPrompt(facet, parent.children[:cfg.maxChildrenForRenaming])

            def processOutputFunc(parent: DataCluster, renamePrompt: str, renamingOutput: ClusterNameAndSummary):
                # if only have one child, just copy name and summary, no need to drift
                uniqueChildren = set()
                for child in parent.children:
                    uniqueChildren.add((child.name.lower(), child.summary.lower()))
                if len(uniqueChildren) == 1:
                    child = parent.children[0]
                    parent.name = child.name
                    parent.summary = child.summary
                else:
                    # No sampling - just use the single output directly
                    parent.summary = renamingOutput.summary
                    parent.name = renamingOutput.name
        
            runBatched(list(parents.values()),
                getInputs=getInputsFunc,
                processBatch=processBatchFuncLLMWithStructuredNaming,
                processOutput=processOutputFunc,
                batchSize=cfg.llmBatchSize,
                verbose=cfg.verbose)

            # Now those parents are our current level, go up higher
            curLevelFacetClusters = list(parents.values())
            level += 1
            cfg.print(f"Now have {len(curLevelFacetClusters)} on level {level}")
        topLevelParents.append(curLevelFacetClusters)
    return topLevelParents

def selectOptimalK(
        embeddings: EmbeddingArray,
        k_min: int,
        k_max: int,
        k_step: int,
        seed: int,
        cfg: OpenClioConfig
    ) -> int:
    """
    Select optimal K using Calinski-Harabasz score.
    Higher CH score = better defined clusters.
    """
    from sklearn.metrics import calinski_harabasz_score

    # Normalize embeddings once
    normalized_embeddings = preprocessing.normalize(embeddings)
    n = embeddings.shape[0]

    # Clamp search range
    k_min = max(2, k_min)  # Need at least 2 clusters
    k_max = min(n - 1, k_max)  # Can't have more clusters than points

    cfg.print(f"Selecting optimal K using Calinski-Harabasz score")
    cfg.print(f"  Search range: K={k_min} to K={k_max} (step={k_step})")

    best_k = k_min
    best_score = -np.inf
    scores = []

    for k in range(k_min, k_max + 1, k_step):
        if k >= n:
            break

        # Fit k-means
        kmeans = FaissKMeans(n_clusters=k, random_state=seed, **cfg.kmeansArgs)
        kmeans.fit(normalized_embeddings)

        # Compute CH score
        score = calinski_harabasz_score(embeddings, kmeans.labels_)
        scores.append((k, score))

        cfg.print(f"  K={k}: CH score={score:.2f}")

        if score > best_score:
            best_score = score
            best_k = k

    cfg.print(f"  ✓ Selected K={best_k} (CH score={best_score:.2f})")
    return best_k


def getBaseClusters(
        facets: List[FacetMetadata],
        vertex_client: Any,
        model_name: str,
        embeddingModel: SentenceTransformer,
        facetValues: List[DataPointFacetData],
        facetValuesEmbeddings: List[Optional[EmbeddingArray]],
        cfg: OpenClioConfig,
        runIfNotExist: Callable[[str, Callable[[], Any], bool], Tuple[Any, bool]],
        dependencyModified: bool,
    ) -> Tuple[List[Optional[FaissKMeans]], List[Optional[List[DataCluster]]]]:
    """
    Gets the base-level clusters for all facets that have shouldMakeFacetClusters(facet) True.
    """
    seed = cfg.seed
    baseClusters = [None] * len(facets)
    for facetI, facet in enumerate(facets):
        if shouldMakeFacetClusters(facet):
            facetEmbeddings = facetValuesEmbeddings[facetI]
            n = facetEmbeddings.shape[0]

            # Determine K (either auto-select or use function)
            def selectK():
                if cfg.autoSelectK:
                    k_min, k_max = cfg.kSearchRange
                    return selectOptimalK(
                        facetEmbeddings,
                        k_min=k_min,
                        k_max=k_max,
                        k_step=cfg.kSearchStep,
                        seed=cfg.seed,
                        cfg=cfg
                    )
                else:
                    return cfg.nBaseClustersFunc(n)

            # Cache K selection
            optimal_k, _ = runIfNotExist(f"optimal_k_facet{facetI}.pkl", selectK, dependencyModified)
            optimal_k = min(n, optimal_k)  # Sanity check

            def getKMeans():
                cfg.print(f"Running kmeans for facet {facet.name} with K={optimal_k}")
                kmeans = FaissKMeans(n_clusters=optimal_k, random_state=cfg.seed, **cfg.kmeansArgs)
                kmeans.fit(preprocessing.normalize(facetEmbeddings))
                return kmeans.labels_, kmeans.cluster_centers_

            (kmeansLabels, kmeansClusterCenters), _ = runIfNotExist(f"basekmeans{facetI}.pkl", getKMeans, dependencyModified)

            def getInputsFunc(clusterIndex : int) -> str:
                clusterPointsIndices = np.where(kmeansLabels == clusterIndex)[0]
                sampledClusterIndices = np.random.choice(clusterPointsIndices, size=min(cfg.maxPointsToSampleInsideCluster, clusterPointsIndices.shape[0]), replace=False)
                outsideClusterIndices = np.where(kmeansLabels != clusterIndex)[0]
                distancesToCenter = cdist(facetEmbeddings, kmeansClusterCenters[clusterIndex].reshape(1, -1))[:,0]
                closestPointsOutsideClusterIndices = outsideClusterIndices[np.argsort(distancesToCenter[kmeansLabels != clusterIndex])]
                sampledOutsideClusterIndices = closestPointsOutsideClusterIndices[:min(cfg.maxPointsToSampleOutsideCluster, closestPointsOutsideClusterIndices.shape[0])]

                clusterFacetValues = sorted(list(set([facetValues[i].facetValues[facetI].value for i in clusterPointsIndices])))
                clusterOutsideValues = sorted(list(set([facetValues[i].facetValues[facetI].value for i in sampledOutsideClusterIndices])))
                # No sampling - just generate one prompt
                prompt = getFacetClusterNamePrompt(facet, clusterFacetValues, clusterOutsideValues)
                return prompt

            # Rate limiting with threading
            rate_limit_lock = threading.Lock()
            last_request_times = []

            def processBatchFunc(batchOfPrompts: List[str]) -> List[ClusterNameAndSummary]:
                nonlocal seed
                seed += 1

                def call_api(prompt: str) -> ClusterNameAndSummary:
                    # Rate limiting with minimum delay
                    with rate_limit_lock:
                        current_time = time.time()

                        # Enforce minimum delay between requests
                        if len(last_request_times) > 0:
                            time_since_last = current_time - last_request_times[-1]
                            if time_since_last < cfg.minDelayBetweenRequests:
                                time.sleep(cfg.minDelayBetweenRequests - time_since_last)
                                current_time = time.time()

                        cutoff_time = current_time - 60.0
                        last_request_times[:] = [t for t in last_request_times if t > cutoff_time]

                        if len(last_request_times) >= cfg.vertexRateLimitPerMin:
                            sleep_time = last_request_times[0] - cutoff_time + 0.1
                            time.sleep(sleep_time)
                            current_time = time.time()
                            cutoff_time = current_time - 60.0
                            last_request_times[:] = [t for t in last_request_times if t > cutoff_time]

                        last_request_times.append(current_time)

                    # Retry with exponential backoff on rate limits
                    max_retries = 3
                    for attempt in range(max_retries):
                        try:
                            # Use structured outputs with Pydantic class
                            cfg.print(f"Making API call (attempt {attempt + 1}/{max_retries})...")

                            # Log prompt length for debugging
                            cfg.print(f"Prompt length: {len(prompt)} chars")

                            response = vertex_client.models.generate_content(
                                model=model_name,
                                contents=[prompt],
                                config={
                                    "response_mime_type": "application/json",
                                    "response_schema": ClusterNameAndSummary,
                                    "max_output_tokens": 8192,  # High limit needed for gemini-2.5-flash
                                    "temperature": 0.0,  # Deterministic
                                }
                            )

                            cfg.print(f"API call successful, parsing response...")
                            # Check if response has parsed attribute
                            if hasattr(response, 'parsed') and response.parsed is not None:
                                cfg.print(f"Response parsed successfully: {response.parsed.name}")
                                return response.parsed
                            else:
                                # Try getting candidates and parts
                                cfg.print(f"No parsed attribute, trying candidates...")
                                import json
                                try:
                                    # Try to get the full response text from candidates
                                    if hasattr(response, 'candidates') and len(response.candidates) > 0:
                                        candidate = response.candidates[0]

                                        # Check finish reason
                                        if hasattr(candidate, 'finish_reason'):
                                            cfg.print(f"Finish reason: {candidate.finish_reason}")

                                        # Check if response was blocked by safety filters
                                        if hasattr(candidate, 'finish_reason') and candidate.finish_reason != 1:  # 1 = STOP (normal completion)
                                            cfg.print(f"Response did not complete normally. Finish reason code: {candidate.finish_reason}")
                                            # Return a default result instead of retrying
                                            return ClusterNameAndSummary(summary="Cluster description", name="Music Genre Cluster")

                                        if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                                            full_text = ''.join([part.text for part in candidate.content.parts])
                                            cfg.print(f"Got full text from candidates (length={len(full_text)}): {full_text[:200]}...")
                                            if len(full_text) > 50:
                                                cfg.print(f"Full text ends with: ...{full_text[-50:]}")
                                            json_response = json.loads(full_text)
                                            parsed = ClusterNameAndSummary(**json_response)
                                            cfg.print(f"Manual parsing successful: {parsed.name}")
                                            return parsed

                                    # Fallback to response.text
                                    if hasattr(response, 'text'):
                                        cfg.print(f"Trying response.text (len={len(response.text)}): {response.text[:200]}...")
                                        json_response = json.loads(response.text)
                                        parsed = ClusterNameAndSummary(**json_response)
                                        cfg.print(f"Manual parsing successful: {parsed.name}")
                                        return parsed

                                    cfg.print(f"Could not extract text from response. Type: {type(response)}")
                                    return ClusterNameAndSummary(summary="Error in naming", name="Error")
                                except json.JSONDecodeError as je:
                                    cfg.print(f"JSON decode error: {je}. Full text: {full_text if 'full_text' in locals() else 'N/A'}")
                                    # If JSON is incomplete, retry this attempt
                                    if attempt < max_retries - 1:
                                        cfg.print(f"Retrying due to malformed JSON...")
                                        time.sleep(2)
                                        continue
                                    return ClusterNameAndSummary(summary="Error in naming", name="Error")

                        except Exception as e:
                            error_msg = str(e).lower()
                            cfg.print(f"Exception in cluster naming (attempt {attempt + 1}/{max_retries}): {e}")
                            # Check if it's a rate limit error
                            if "429" in error_msg or "quota" in error_msg or "rate" in error_msg:
                                if attempt < max_retries - 1:
                                    wait_time = (2 ** attempt) * 2  # Exponential backoff: 2s, 4s, 8s
                                    cfg.print(f"Rate limit hit, waiting {wait_time}s before retry {attempt + 1}/{max_retries}")
                                    time.sleep(wait_time)
                                    continue
                            cfg.print(f"Error in cluster naming: {e}")
                            return ClusterNameAndSummary(summary="Error in naming", name="Error")

                    # If we get here, all retries failed
                    cfg.print(f"All retries exhausted for cluster naming")
                    return ClusterNameAndSummary(summary="Error in naming", name="Error")

                # Since maxParallelLLMCalls=1, just run sequentially to save memory
                results = []
                for idx, prompt in enumerate(batchOfPrompts):
                    try:
                        result = call_api(prompt)
                        results.append(result)
                        cfg.print(f"Completed prompt {idx + 1}/{len(batchOfPrompts)}")
                        # Force garbage collection after each API call to prevent memory buildup
                        gc.collect()
                    except Exception as e:
                        cfg.print(f"Exception in call_api for prompt {idx}: {e}")
                        results.append(ClusterNameAndSummary(summary="Error in naming", name="Error"))

                return results

            def processOutputFunc(clusterIndex: int, clusterPrompt: str, clusterOutput: ClusterNameAndSummary) -> DataCluster:
                clusterPointsIndices = np.arange(len(facetEmbeddings))[kmeansLabels == clusterIndex]
                # No sampling - just use the single output directly
                return DataCluster(
                    facet=facet,
                    summary=clusterOutput.summary,
                    name=clusterOutput.name,
                    indices=clusterPointsIndices,
                )

            facetBaseClusters = runBatched(range(len(kmeansClusterCenters)),
               getInputs=getInputsFunc,
               processBatch=processBatchFunc,
               processOutput=processOutputFunc,
               batchSize=cfg.llmBatchSize,
               verbose=cfg.verbose)
            baseClusters[facetI] = facetBaseClusters

            # Force garbage collection after completing a facet to free memory
            gc.collect()
            cfg.print(f"Completed clustering for facet {facet.name}, memory freed")
    return baseClusters

def getFacetValuesEmbeddings(
        facets: List[FacetMetadata],
        facetValues: List[DataPointFacetData],
        embeddingModel: SentenceTransformer,
        cfg: OpenClioConfig) -> List[Optional[EmbeddingArray]]:
    """
    Gets the embeddings of all facet values that have shouldMakeFacetClusters(facet) True
    (this is when the facet has a summaryCriteria that is not None)
    Returns one element for each facet value
    That element will either be None if shouldMakeFacetClusters(facet) is False,
    or a numpy array of size [numDataPoints, embeddingDim]
    """
    def getInputsFunc(facetI: int) -> List[str]:
        facetInputs = []
        facet = facets[facetI]
        if shouldMakeFacetClusters(facet):
            for facetData in facetValues:
                facetValue = facetData.facetValues[facetI].value
                facetInputs.append(facetValue)
        return facetInputs
    
    def processBatchFunc(batchOfTextInputs: List[str]) -> List[npt.NDArray[np.float32]]:
        embedded = embeddingModel.encode(batchOfTextInputs, show_progress_bar=False)
        return [embedded[i] for i in range(len(batchOfTextInputs))]

    def processOutputFunc(facetI: int, facetInputs: List[str], embeddings: List[npt.NDArray[np.float32]]) -> Optional[EmbeddingArray]:
        facet = facets[facetI]
        if shouldMakeFacetClusters(facet):
            return np.stack(embeddings)
        else:
            return None
    
    return runBatched(list(range(len(facets))),
                    getInputs=getInputsFunc,
                    processBatch=processBatchFunc,
                    processOutput=processOutputFunc,
                    batchSize=cfg.embedBatchSize,
                    verbose=cfg.verbose)


def getFacetValues(
        facets: List[FacetMetadata],
        facetSchema: Type[BaseModel],
        vertex_client: Any,
        model_name: str,
        data: List[Any],
        cfg: OpenClioConfig
    ) -> List[DataPointFacetData]:
    """
    Gets ALL facet values for each data point in a single LLM call using structured outputs.
    Returns a list of DataPointFacetData objects, one for each data point.
    """
    facet_field_names = list(facetSchema.model_fields.keys())

    def getInputsFunc(data_point: Any) -> str:
        data_point = cfg.getDataFunc(data_point)

        # Truncate text if too long
        if isinstance(data_point, str) and len(data_point) > cfg.maxTextChars:
            data_point = data_point[:cfg.maxTextChars]

        # Create a single prompt that extracts all facets at once
        prompt = f"""Analyze the following text and extract all requested information:

<text>
{data_point}
</text>

Provide your analysis in JSON format according to the schema."""
        return prompt

    seed = cfg.seed

    # Rate limiting with threading
    rate_limit_lock = threading.Lock()
    last_request_times = []

    def rate_limited_call(prompt: str) -> BaseModel:
        """Make a rate-limited API call - returns parsed Pydantic object"""
        # Rate limiting: ensure we don't exceed requests per minute
        with rate_limit_lock:
            current_time = time.time()

            # Enforce minimum delay between requests
            if len(last_request_times) > 0:
                time_since_last = current_time - last_request_times[-1]
                if time_since_last < cfg.minDelayBetweenRequests:
                    time.sleep(cfg.minDelayBetweenRequests - time_since_last)
                    current_time = time.time()

            # Remove requests older than 1 minute
            cutoff_time = current_time - 60.0
            last_request_times[:] = [t for t in last_request_times if t > cutoff_time]

            # If we're at the limit, wait
            if len(last_request_times) >= cfg.vertexRateLimitPerMin:
                sleep_time = last_request_times[0] - cutoff_time + 0.1
                time.sleep(sleep_time)
                # Clean up again after sleep
                current_time = time.time()
                cutoff_time = current_time - 60.0
                last_request_times[:] = [t for t in last_request_times if t > cutoff_time]

            last_request_times.append(current_time)

        # Make the actual API call with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = vertex_client.models.generate_content(
                    model=model_name,
                    contents=[prompt],
                    config={
                        "response_mime_type": "application/json",
                        "response_schema": facetSchema,
                        "max_output_tokens": cfg.llmExtraInferenceArgs.get("max_tokens", 1000),
                        "temperature": cfg.llmExtraInferenceArgs.get("temperature", 0.7),
                        "top_p": cfg.llmExtraInferenceArgs.get("top_p", 0.8),
                        "top_k": cfg.llmExtraInferenceArgs.get("top_k", 40),
                    }
                )
                # SDK parses and validates for us
                return response.parsed
            except Exception as e:
                error_msg = str(e).lower()
                # Check if it's a rate limit error
                if "429" in error_msg or "quota" in error_msg or "rate" in error_msg:
                    if attempt < max_retries - 1:
                        wait_time = (2 ** attempt) * 2  # Exponential backoff: 2s, 4s, 8s
                        cfg.print(f"Rate limit hit, waiting {wait_time}s before retry {attempt + 1}/{max_retries}")
                        time.sleep(wait_time)
                        continue
                cfg.print(f"Error extracting facets: {e}")
                # Return empty Pydantic object
                return facetSchema(**{field: "" for field in facet_field_names})

    def processBatchFunc(batchOfPrompts: List[str]) -> List[BaseModel]:
        nonlocal seed
        seed += 1

        # Parallel API calls with rate limiting
        results = [None] * len(batchOfPrompts)
        with ThreadPoolExecutor(max_workers=cfg.maxParallelLLMCalls) as executor:
            future_to_idx = {
                executor.submit(rate_limited_call, prompt): idx
                for idx, prompt in enumerate(batchOfPrompts)
            }

            with tqdm(total=len(batchOfPrompts), desc="Extracting facets", disable=not cfg.verbose) as pbar:
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    results[idx] = future.result()
                    pbar.update(1)

        return results

    def processOutputFunc(data_point: Any, data_point_prompt: str, facetOutput: BaseModel) -> DataPointFacetData:
        facet_values = []

        try:
            # Extract from Pydantic object directly
            for facet, field_name in zip(facets, facet_field_names):
                value = getattr(facetOutput, field_name, "")
                if value is None:
                    value = ""
                facet_values.append(FacetValue(facet=facet, value=str(value).strip()))

        except Exception as e:
            cfg.print(f"Failed to extract facet values: {e}")
            # Create empty facet values on error
            for facet in facets:
                facet_values.append(FacetValue(facet=facet, value=""))

        return DataPointFacetData(
            data=data_point,
            facetValues=facet_values
        )

    return runBatched(data,
               getInputs=getInputsFunc,
               processBatch=processBatchFunc,
               processOutput=processOutputFunc,
               batchSize=cfg.llmBatchSize,
               verbose=cfg.verbose)

##### Various utility parsing stuff #####
def connectedComponentsFromMask(mask: np.ndarray) -> List[np.ndarray]:
    """
    mask: dense *boolean* adjacency matrix (n × n, symmetric, no self-loops)
    returns: list of 1-D index arrays – one per connected component
    """
    graph = csr_matrix(mask, dtype=bool)
    n_components, labels = connected_components(graph, directed=False)
    return [np.flatnonzero(labels == k) for k in range(n_components)]

def medoidFromEmbeddings(indices: np.ndarray,
                            embs: np.ndarray) -> int:
    """
    embs: unit-norm embeddings (n × d)
    indices: indices of the points that form one component
    returns: index (WITHIN indices) of the true medoid under cosine distance
    """
    sub = embs[indices]                       # |C| × d
    sim = cosine_similarity(sub)              # |C| × |C|
    distSums = (1.0 - sim).sum(axis=1)
    return indices[int(np.argmin(distSums))] # global index


def deduplicateByEmbeddingsAndMergeSources(
    valuesAndSources: List[Tuple[str, List[int]]],
    embeddingModel: SentenceTransformer,
    tau: float = 0.15,          # distance threshold (0.15 ≈ cosine ≥ 0.85)
    ):
    """
    Single-link deduplication.  Returns one representative per duplicate set,
    chosen as the exact medoid of each connected component.
    Sources for each representitive will be the union of the all sources in connected component
    """
    if len(valuesAndSources) == 0:
        return []

    # 1. Embed once, L2-normalise so cosine == dot product
    valuesAsStr = [value for (value, sources) in valuesAndSources]
    emb = preprocessing.normalize(embeddingModel.encode(valuesAsStr, show_progress_bar=False))

    # 2. Dense distance matrix  (O(n²) memory!)
    sim = cosine_similarity(emb)
    dist = 1.0 - sim

    # 3. Boolean adjacency under threshold (no self-loops)
    mask = (dist <= tau) & ~np.eye(len(valuesAsStr), dtype=bool)

    # 4. Connected components (single-link duplicate sets)
    components = connectedComponentsFromMask(mask)

    # 5. Medoid for every component
    representatives = []
    for comp in components:
        if comp.size == 1:
            representatives.append(valuesAndSources[comp[0]])
        else:
            # union all members of connected component
            sourcesUnion = set()
            for index in comp:
                value, sources = valuesAndSources[index]
                sourcesUnion |= set(sources)
            # use medoid element as representative
            medoid_idx = medoidFromEmbeddings(comp, emb)
            representatives.append((valuesAsStr[medoid_idx], sorted(list(sourcesUnion))))

    return representatives

def deduplicateByEmbeddings(
        values: List[str],
        embeddingModel: SentenceTransformer,
        tau: float = 0.15,          # distance threshold (0.15 ≈ cosine ≥ 0.85)
        valueMap: Optional[Callable[[Any], str]] = None
) -> List[str]:
    """
    Single-link deduplication.  Returns one representative per duplicate set,
    chosen as the exact medoid of each connected component.
    """
    if len(values) == 0:
        return []

    # 1. Embed once, L2-normalise so cosine == dot product
    valuesAsStr = list(map(valueMap, values)) if valueMap is not None else values
    emb = preprocessing.normalize(embeddingModel.encode(valuesAsStr, show_progress_bar=False))

    # 2. Dense distance matrix  (O(n²) memory!)
    sim = cosine_similarity(emb)
    dist = 1.0 - sim

    # 3. Boolean adjacency under threshold (no self-loops)
    mask = (dist <= tau) & ~np.eye(len(values), dtype=bool)

    # 4. Connected components (single-link duplicate sets)
    components = connectedComponentsFromMask(mask)

    # 5. Medoid for every component
    representatives = []
    for comp in components:
        if comp.size == 1:
            representatives.append(values[comp[0]])
        else:
            medoid_idx = medoidFromEmbeddings(comp, emb)
            representatives.append(values[medoid_idx])

    return representatives

# from https://gist.github.com/gatheluck/c57e2a40e3122028ceaecc3cb0d152ac
def setSeed(seed):
    """
    Set seeds (to lots of things)
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def bestMatch(
    query: str,
    candidates: List[str],
    model: SentenceTransformer
) -> str:
    """
    Find the best matching candidate for a query string using cosine similarity.
    Returns the candidate with highest similarity to query.
    """
    if len(candidates) == 0:
        raise ValueError("candidates must be non-empty")
    if len(candidates) == 1:
        return candidates[0]

    # Encode query and candidates
    queryEmb = preprocessing.normalize(model.encode([query], convert_to_numpy=True, show_progress_bar=False))
    candidatesEmb = preprocessing.normalize(model.encode(candidates, convert_to_numpy=True, show_progress_bar=False))

    # Compute cosine similarity
    similarities = cosine_similarity(queryEmb, candidatesEmb)[0]

    # Return candidate with highest similarity
    best_idx = np.argmax(similarities)
    return candidates[best_idx]


def bestRepresentativePair(
    A: List[str],
    B: List[str],
    model: SentenceTransformer
) -> Tuple[str, str]:
    """
    Return (a_star, b_star) where:
      • b_star  minimises Σ_{a∈A} (1 - cos(a, b))
      • a_star  is the element of A closest to that b_star
    """
    if len(A) == 0 or len(B) == 0:
        raise ValueError("A and B must be non-empty")

    # 1 . Encode & L2-normalise so cosine == dot product
    AEmb = preprocessing.normalize(model.encode(A, convert_to_numpy=True, show_progress_bar=False))
    BEmb = preprocessing.normalize(model.encode(B, convert_to_numpy=True, show_progress_bar=False))

    # 2 . Cosine similarity matrix  (n × m)
    sim = cosine_similarity(AEmb, BEmb)         # fast, vectorised

    # 3 . For every b_j compute total distance to all a_i
    distSumsForEachB = (1.0 - sim).sum(axis=0)           # shape (m,)

    closestB = int(np.argmin(distSumsForEachB))
    bStar = B[closestB]

    # 4 . Find a_i closest to that bStar            
    distOfEachAToBStar = 1.0 - sim[:, closestB]                 # shape (n,)
    bestAIndex = int(np.argmin(distOfEachAToBStar))
    aStar = A[bestAIndex]

    return aStar, bStar

def getMedoidViaEmbeddings(
    values: List[str],
    embeddingModel: SentenceTransformer
) -> str:
    """
    Return the element of `values` that minimises the sum of
    cosine distances ( = 1 - cosine similarity ) to all others.

    This is the exact medoid under cosine distance.
    Complexity:  O(n²) in time,  O(n²) in memory.
    """
    if len(values) == 0:
        raise ValueError("`values` must contain at least one string.")

    # 1. Embed and L2-normalise so that
    #    cosine(u, v) == u @ v   (dot product after normalisation)
    embeddings = preprocessing.normalize(
        embeddingModel.encode(values, convert_to_numpy=True, show_progress_bar=False)
    )                                # shape = (n, d)

    # 2. Pair-wise cosine similarity matrix  (n × n)
    #    sim[i, j] = cosine( values[i], values[j] )
    sim = cosine_similarity(embeddings)           # fast & vectorised

    # 3. Convert similarity → distance  and add up per row
    #    dist = 1 - sim   (because vectors are unit-norm)
    dist_sums = (1.0 - sim).sum(axis=1)           # shape = (n,)

    # 4. Index of smallest total distance = medoid
    medoid_idx = int(np.argmin(dist_sums))
    return values[medoid_idx]

def getCentroidViaEmbeddings(
    values: List[str],
    embeddingModel: SentenceTransformer) -> str:
    """
    Computes the average of the values embeddings,
    then finds the value that is closest to that average
    This is sort of a continuous version of "pick the most common element"
    But actually what we want is the medoid (term that is closest to all the others)
    """
    normalizedValues = preprocessing.normalize(embeddingModel.encode(values, show_progress_bar=False))
    avg = normalizedValues.mean(axis=0)
    sims = cosine_similarity(wow, wow.mean(axis=0).reshape(1, -1)).flatten()
    return values[np.argmax(sims)]

def getMedoidSummaryAndName(outputs: List[ClusterNameAndSummary], embeddingModel: SentenceTransformer) -> Tuple[str, str]:
    """
    Continuous version of "get most common"
    That gets the embedded value that is closest to all other items (the medoid)
    returns (summary, name)
    Takes structured ClusterNameAndSummary objects directly from Pydantic validation
    """
    # Filter out None values and extract summaries and names from structured objects
    valid_outputs = [output for output in outputs if output is not None]
    summaries = [removePunctuation(output.summary.strip()) for output in valid_outputs if output.summary and output.summary.strip()]
    names = [removePunctuation(output.name.strip()) for output in valid_outputs if output.name and output.name.strip()]

    # Fallback if no valid data
    if len(summaries) == 0:
        summaries.append("Unknown cluster")
    if len(names) == 0:
        names.append("Unknown")

    return getMedoidViaEmbeddings(summaries, embeddingModel), getMedoidViaEmbeddings(names, embeddingModel)

def getMostCommonSummaryAndName(outputs: List[str]) -> Tuple[str, str]:
    """
    Gets most common thing in <summary> tag and <name> tag,
    returns (summary, name)
    I recommend to use getMedoidSummaryAndName so we act in embedding space instead
    """
    summaryCounts = defaultdict(lambda: 0)
    nameCounts = defaultdict(lambda: 0)
    for output in outputs:
        # re.DOTALL makes . match newlines too (by default it does not)
        matches = re.findall(r"(.*?)</summary>.*?<name>(.*?)</name>", output, re.DOTALL)
        if len(matches) > 0:
            for summary, name in matches:
                summaryCounts[cleanTrailingTagsInOutput(summary)] += 1
                nameCounts[cleanTrailingTagsInOutput(name)] += 1
    def largestCountItem(counts, fieldName):
        if len(counts) == 0: return f"<Could not extract {fieldName}>"
        counts = sorted(list(counts.items()), key=lambda x: (-x[1], x[0]))
        largestKey, largestValue = counts[0]
        return largestKey
    summary = largestCountItem(summaryCounts, "summary")
    name = largestCountItem(nameCounts, "name")
    return summary, name

def removePunctuation(output: str) -> str:
    """
    Removes ., ?, and ! from end of a string.
    (and strips it before and afterwards)
    """
    output = output.strip()
    if output.endswith("."):
        output = output[:-1].strip()
    if output.endswith("?"):
        output = output[:-1].strip()
    if output.endswith("!"):
        output = output[:-1].strip()
    return output

def extractTagValue(output: str, tag: str) -> Tuple[bool, str]:
    """
    Gets value contained in <tag>VALUE_HERE</tag>
    returns (foundTag, valueInTag)
    where foundTag is True if the tag was found
    """
    posOfTag = output.lower().find(f"<{tag}>")
    if posOfTag != -1:
        output = output[posOfTag + len(f"<{tag}>"):].strip()
    endOfTagPos = output.lower().find(f"</{tag}>")
    if endOfTagPos != -1:
        output = output[:endOfTagPos].strip()
    return posOfTag != -1 and endOfTagPos != -1, output

def extractAnswerNumberedList(output: str, ignoreNoTrailing: bool = False) -> List[str]:
    """
    If we have
    <answer>
    1. blah
    2. blahhh
    3. wow
    etc.
    </answer>
    This will return 
    ["blah", "blahhh", "wow", ...]
    """
    results = []
    foundAnswerTag, answer = extractTagValue(output, "answer")
    if foundAnswerTag:
        results += [removeNumberFromOutput(line) for line in answer.split("\n") if len(line.strip()) > 0]
    elif ignoreNoTrailing:
        posOfTag = output.lower().find("<answer>")
        if posOfTag != -1:
            output = output[posOfTag + len(f"<answer>"):].strip()
        results += [removeNumberFromOutput(line) for line in answer.split("\n") if len(line.strip()) > 0]
        results = results[:-1] # ignore last one since it's probably partially formed, we got cut off early
    return results

def removeNumberFromOutput(output: str) -> str:
    """
    Removes number. from the front of the output
    Like 
    "1. hi"
    becomes
    "hi"
    or
    "144. wow"
    becomes
    "wow"
    """
    return re.sub(r"^\d*?\.", "", output.strip(), count=1).strip()

def cleanTrailingTagsInOutput(output: str) -> str:
    """
    Removes any trailing </tag> that may existing in the output
    Also strips it before and afterwards
    """
    return re.findall(r"(.*?)(?:(?:</)|$)", output.strip(), re.DOTALL)[0].strip()


def printHierarchyHelper(
    parents: List[DataCluster],
    indent: str) -> List[str]:
    lines = []
    for parent in parents:
        lines.append(indent + parent.name)
        if not parent.children is None:
            lines += printHierarchyHelper(parent.children, indent + "  ")
    return lines

def printHierarchy(parents: List[DataCluster]):
    """
    helper function to manually print hierarchy of a specific facet
    """
    resLines = printHierarchyHelper(parents, indent="")
    print("\n".join(resLines))
    with open("hierarchy.txt", "w") as f:
        f.write("\n".join(resLines))