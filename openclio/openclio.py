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
import shutil
import os
import re
import random
import cloudpickle
from pathlib import Path
import time
from pydantic import BaseModel, Field, create_model
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from tqdm import tqdm

from .prompts import getFacetClusterNamePrompt, getNeighborhoodClusterNamesPrompt, getDeduplicateClusterNamesPrompt, getAssignToHighLevelClusterPrompt, getRenamingHigherLevelClusterPrompt
from .utils import flatten, unflatten, runBatched, dedup
from .opencliotypes import FacetMetadata, FacetValue, ConversationFacetData, ConversationEmbedding, ConversationCluster, OpenClioConfig, OpenClioResults, EmbeddingArray, shouldMakeFacetClusters
from .faissKMeans import FaissKMeans
from .writeOutput import convertOutputToWebpage, computeUmap

# System prompt facets as Pydantic model
class SystemPromptFacets(BaseModel):
    """Facets for analyzing AI agent system prompts"""
    primary_purpose: str = Field(description="The primary purpose or role of the AI agent. Be clear and concise, 1-2 sentences.")
    domain: str = Field(description="The domain or subject area the AI agent operates in (e.g., healthcare, finance, education, etc.). 1-2 sentences.")
    key_capabilities: str = Field(description="The key capabilities or features emphasized. List the most important ones. 1-2 sentences.")
    interaction_style: str = Field(description="The interaction style or personality the AI agent adopts (e.g., professional, casual, empathetic). 1-2 sentences.")

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


# Vertex AI helper functions
def pydantic_to_vertex_schema(model: Type[BaseModel]) -> Dict[str, Any]:
    """Convert a Pydantic model to Vertex AI schema format"""
    schema = model.model_json_schema()
    properties = {}
    for field_name, field_info in schema.get("properties", {}).items():
        field_type = field_info.get("type", "string")
        vertex_type = {
            "string": "STRING",
            "integer": "INTEGER",
            "number": "NUMBER",
            "boolean": "BOOLEAN",
            "array": "ARRAY",
            "object": "OBJECT"
        }.get(field_type, "STRING")

        properties[field_name] = {
            "type": vertex_type,
            "description": field_info.get("description", "")
        }

        if field_type == "array" and "items" in field_info:
            items_type = field_info["items"].get("type", "string")
            properties[field_name]["items"] = {
                "type": {"string": "STRING", "integer": "INTEGER", "number": "NUMBER", "boolean": "BOOLEAN"}.get(items_type, "STRING")
            }

    return {
        "type": "OBJECT",
        "properties": properties,
        "required": schema.get("required", [])
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
            htmlRoot: str = None,
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
    htmlRoot -- (Optional) Web output path
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

    # Initialize Vertex AI
    try:
        import vertexai
        from vertexai.generative_models import GenerativeModel
        vertexai.init(project=project_id, location=location)
        vertex_model = GenerativeModel(model_name)
        cfg.print(f"Initialized Vertex AI with model {model_name}")
    except ImportError:
        raise ImportError("Vertex AI dependencies not installed. Install with: pip install google-cloud-aiplatform")
    except Exception as e:
        raise RuntimeError(f"Failed to initialize Vertex AI: {e}")

    # Convert facetMetadata dict to list in consistent order
    facet_field_names = list(facetSchema.model_fields.keys())
    facets = [facetMetadata[field_name] for field_name in facet_field_names]

    dependencyModified = False
    def runIfNotExist(path, runFunc, dependencyModified):
        fullPath = os.path.join(outputDirectory, path)
        if os.path.exists(fullPath) and not dependencyModified: # recompute if dependency modified
            try:
                cfg.print(f"Resuming from {fullPath}...")
                with open(fullPath, "rb") as f:
                    result = cloudpickle.load(f)
                cfg.print(f"Resumed from {fullPath}")
                return result, False
            except:
                cfg.print(f"Failed to load from path {fullPath}, ignoring")
                cfg.print(traceback.format_exc())
        res = runFunc()
        with open(fullPath, "wb") as f:
            cloudpickle.dump(res, f)
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
                    vertex_model=vertex_model,
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
                    vertex_model=vertex_model,
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
                    vertex_model=vertex_model,
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
                    tokenizer=None,  # No tokenizer with Vertex AI
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

    # Display widget or generate web output
    if displayWidget:
        from .widget import ClioWidget
        widget = ClioWidget(output)
        # Store results in widget so user can access them
        widget.results = output
        # Return widget - Jupyter will auto-display it (like embedding-atlas does)
        return widget
    elif htmlRoot is not None:
        htmlOutputPath = os.path.join(outputDirectory, htmlRoot.strip()[1:] if htmlRoot.strip().startswith("/") else htmlRoot)
        cfg.print(f"Outputting to webpage at path {htmlOutputPath}")
        # clear old outputs
        if not htmlRoot in ["/", ""] and os.path.exists(htmlOutputPath):
            cfg.print(f"Removing old outputs at {htmlOutputPath}")
            shutil.rmtree(htmlOutputPath)
        Path(htmlOutputPath).mkdir(parents=True, exist_ok=True)
        convertOutputToWebpage(
            output=output,
            rootHtmlPath=htmlRoot,
            targetDir=htmlOutputPath,
            maxSizePerFile=cfg.htmlMaxSizePerFile,
            conversationFilter=cfg.htmlConversationFilterFunc,
            dataToJson=cfg.htmlDataToJsonFunc,
            verbose=cfg.verbose,
            password=cfg.password
        )

        # write redirect page if we are nested, so the webui opens to it nicely
        if not htmlRoot in ["/", ""]:
            redirectPage = f"""
<!DOCTYPE html>
<html>
<head>
    <meta http-equiv="refresh" content="0; url={htmlRoot}">
</head>
<body>
    <p>If you are not redirected, <a href="{htmlRoot}">click here</a></p>
</body>
</html>
            """
            with open(os.path.join(outputDirectory, "index.html"), "w") as f:
                f.write(redirectPage)

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
    vertex_model: Any,
    embeddingModel: SentenceTransformer,
    baseClusters: List[Optional[List[ConversationCluster]]],
    cfg: OpenClioConfig) -> List[Optional[List[ConversationCluster]]]:
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
            try:
                response = vertex_model.generate_content(
                    prompt,
                    generation_config={
                        "max_output_tokens": cfg.llmExtraInferenceArgs.get("max_tokens", 1000),
                        "temperature": cfg.llmExtraInferenceArgs.get("temperature", 0.7),
                        "top_p": cfg.llmExtraInferenceArgs.get("top_p", 0.8),
                        "top_k": cfg.llmExtraInferenceArgs.get("top_k", 40),
                    }
                )
                if hasattr(response, 'text'):
                    results.append(response.text)
                elif hasattr(response, 'candidates') and len(response.candidates) > 0:
                    results.append(response.candidates[0].content.parts[0].text)
                else:
                    results.append("")
            except Exception as e:
                cfg.print(f"Error in LLM call: {e}")
                results.append("")
            time.sleep(60.0 / 60)  # Rate limiting
        return results 
    
    topLevelParents = []
    for facetI, facet in enumerate(facets):
        if not shouldMakeFacetClusters(facet):
            topLevelParents.append(None)
            continue

        curLevelFacetClusters: List[ConversationCluster] = baseClusters[facetI]
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
                    clusterNamePrompts.append(getNeighborhoodClusterNamesPrompt(facet, None, clustersInNeighborhood, cfg.nDesiredHigherLevelNamesPerClusterFunc(len(clustersInNeighborhood)), tokenizerArgs=cfg.tokenizerArgs))
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
                    prompt = getNeighborhoodClusterNamesPrompt(facet, None, clustersInNeighborhood, cfg.nDesiredHigherLevelNamesPerClusterFunc(len(clustersInNeighborhood)), tokenizerArgs=cfg.tokenizerArgs)
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
                return getDeduplicateClusterNamesPrompt(facet, None, higherCategoriesInNeighborhood, targetAmount, tokenizerArgs=cfg.tokenizerArgs)
            
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
            
            def getInputsFunc(facetClusterData: Tuple[ConversationCluster, List[str]]) -> List[str]:
                facetCluster, potentialHigherLevelClusters = facetClusterData
                assignToHigherCategoryPrompts = []
                for i in range(cfg.nCategorizeSamples):
                    random.shuffle(potentialHigherLevelClusters)
                    assignToHigherCategoryPrompts.append(getAssignToHighLevelClusterPrompt(None, clusterToAssign=facetCluster, higherLevelClusters=potentialHigherLevelClusters, tokenizerArgs=cfg.tokenizerArgs))
                return assignToHigherCategoryPrompts

            # name and summary will be generated later
            parents: Dict[str, ConversationCluster] = dict(
                [
                    (categoryName.lower().strip(), ConversationCluster(facet=facet, name=categoryName, summary="")) 
                    for (categoryName, categorySources) in dedupedCategories
                ]
            )
            
            def processOutputFunc(
                    facetClusterData: Tuple[ConversationCluster, List[str]],
                    assignToHigherCategoryPrompts: List[str],
                    assignToHigherCategoryOutput: List[str]
                ):
                facetCluster, potentialHigherLevelClusters = facetClusterData
                assignedClusters = []
                for output in assignToHigherCategoryOutput:
                    foundOutput, outputValue = extractTagValue(output, "answer")
                    # remove cluster and punctuation if it added it
                    outputValue = removePunctuation(outputValue.replace("<cluster>", "").replace("</cluster>", "").strip()).strip()
                    if foundOutput:
                        assignedClusters.append(outputValue)
                # in the embedding space, find the entry
                # bestHigherLevelClusterAssignedTo
                # in potentialHigherLevelClusters that has smallest total distance to all entries of assignedClusters
                # once we have that, bestAssignedCluster is the entry that has smallest distance to bestHigherLevelClusterAssignedTo
                # This approach helps us avoid the model slightly renaming things and helps us pick the most representative pair
                # I invented this idk what they do but this seems the obvious thing to do imo so they probably do this and just didn't say
                if len(assignedClusters) == 0:
                    # failed to extract cluster from llm, fall back to embedding of the cluster
                    assignedClusters.append(facetCluster.summary + "\n" + facetCluster.name)
                if len(potentialHigherLevelClusters) == 0:
                    cfg.print("got empty potentialHigherLevelClusters??")
                    cfg.print(assignedClusters)
                    cfg.print(potentialHigherLevelClusters)
                # lookup in embedding space the best representative pair
                # this finds term in potentialHigherLevelClusters that has smallest total distance summed over all assignedClusters
                bestAssignedCluster, bestHigherLevelClusterAssignedTo = bestRepresentativePair(assignedClusters, potentialHigherLevelClusters, embeddingModel)
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
            def getInputsFunc(parent: ConversationCluster) -> List[str]:
                renamingPrompts = []
                for _ in range(cfg.nRenameSamples):
                    random.shuffle(parent.children)
                    renamingPrompts.append(getRenamingHigherLevelClusterPrompt(facet, None, parent.children[:cfg.maxChildrenForRenaming], tokenizerArgs=cfg.tokenizerArgs))
                return renamingPrompts
            
            def processOutputFunc(parent: ConversationCluster, renamePrompts: List[str], renamingOutputs: List[str]):
                # if only have one child, just copy name and summary, no need to drift
                uniqueChildren = set()
                for child in parent.children:
                    uniqueChildren.add((child.name.lower(), child.summary.lower()))
                if len(uniqueChildren) == 1:
                    child = parent.children[0]
                    parent.name = child.name
                    parent.summary = child.summary
                else:
                    summary, name = getMedoidSummaryAndName(renamingOutputs, embeddingModel)
                    parent.summary = summary
                    parent.name = name
        
            runBatched(list(parents.values()),
                getInputs=getInputsFunc,
                processBatch=processBatchFuncLLM,
                processOutput=processOutputFunc,
                batchSize=cfg.llmBatchSize,
                verbose=cfg.verbose)

            # Now those parents are our current level, go up higher
            curLevelFacetClusters = list(parents.values())
            level += 1
            cfg.print(f"Now have {len(curLevelFacetClusters)} on level {level}")
        topLevelParents.append(curLevelFacetClusters)
    return topLevelParents

def getBaseClusters(
        facets: List[FacetMetadata],
        vertex_model: Any,
        embeddingModel: SentenceTransformer,
        facetValues: List[ConversationFacetData],
        facetValuesEmbeddings: List[Optional[EmbeddingArray]],
        cfg: OpenClioConfig,
        runIfNotExist: Callable[[str, Callable[[], Any], bool], Tuple[Any, bool]],
        dependencyModified: bool,
    ) -> Tuple[List[Optional[FaissKMeans]], List[Optional[List[ConversationCluster]]]]:
    """
    Gets the base-level clusters for all facets that have shouldMakeFacetClusters(facet) True.
    """
    seed = cfg.seed
    baseClusters = [None] * len(facets)
    for facetI, facet in enumerate(facets):
        if shouldMakeFacetClusters(facet):
            facetEmbeddings = facetValuesEmbeddings[facetI]
            n = facetEmbeddings.shape[0]
            def getKMeans():
                cfg.print(f"Running kmeans for facet {facet.name}")
                kmeans = FaissKMeans(n_clusters=min(n, cfg.nBaseClustersFunc(n)), random_state=cfg.seed, **cfg.kmeansArgs)
                kmeans.fit(preprocessing.normalize(facetEmbeddings))
                return kmeans.labels_, kmeans.cluster_centers_

            (kmeansLabels, kmeansClusterCenters), _ = runIfNotExist(f"basekmeans{facetI}.pkl", getKMeans, dependencyModified)

            def getInputsFunc(clusterIndex : int) -> List[str]:
                clusterPointsIndices = np.where(kmeansLabels == clusterIndex)[0]
                sampledClusterIndices = np.random.choice(clusterPointsIndices, size=min(cfg.maxPointsToSampleInsideCluster, clusterPointsIndices.shape[0]), replace=False)
                outsideClusterIndices = np.where(kmeansLabels != clusterIndex)[0]
                distancesToCenter = cdist(facetEmbeddings, kmeansClusterCenters[clusterIndex].reshape(1, -1))[:,0]
                closestPointsOutsideClusterIndices = outsideClusterIndices[np.argsort(distancesToCenter[kmeansLabels != clusterIndex])]
                sampledOutsideClusterIndices = closestPointsOutsideClusterIndices[:min(cfg.maxPointsToSampleOutsideCluster, closestPointsOutsideClusterIndices.shape[0])]

                clusterFacetValues = sorted(list(set([facetValues[i].facetValues[facetI].value for i in clusterPointsIndices])))
                clusterOutsideValues = sorted(list(set([facetValues[i].facetValues[facetI].value for i in sampledOutsideClusterIndices])))
                clusterPrompts = []
                for _ in range(cfg.nNameDescriptionSamplesPerCluster):
                    random.shuffle(clusterFacetValues)
                    random.shuffle(clusterOutsideValues)
                    prompt = getFacetClusterNamePrompt(None, facet, clusterFacetValues, clusterOutsideValues, tokenizerArgs=cfg.tokenizerArgs)
                    clusterPrompts.append(prompt)
                return clusterPrompts

            def processBatchFunc(batchOfPrompts: List[str]) -> List[str]:
                nonlocal seed
                seed += 1
                results = []
                for prompt in batchOfPrompts:
                    try:
                        response = vertex_model.generate_content(
                            prompt,
                            generation_config={
                                "max_output_tokens": cfg.llmExtraInferenceArgs.get("max_tokens", 1000),
                                "temperature": cfg.llmExtraInferenceArgs.get("temperature", 0.7),
                                "top_p": cfg.llmExtraInferenceArgs.get("top_p", 0.8),
                                "top_k": cfg.llmExtraInferenceArgs.get("top_k", 40),
                            }
                        )
                        if hasattr(response, 'text'):
                            results.append(response.text)
                        elif hasattr(response, 'candidates') and len(response.candidates) > 0:
                            results.append(response.candidates[0].content.parts[0].text)
                        else:
                            results.append("")
                    except Exception as e:
                        cfg.print(f"Error: {e}")
                        results.append("")
                    time.sleep(60.0 / 60)
                return results

            def processOutputFunc(clusterIndex: int, clusterPrompts: List[str], clusterOutputs: List[str]) -> ConversationCluster:
                clusterPointsIndices = np.arange(len(facetEmbeddings))[kmeansLabels == clusterIndex]
                summary, name = getMedoidSummaryAndName(clusterOutputs, embeddingModel)
                return ConversationCluster(
                    facet=facet,
                    summary=summary,
                    name=name,
                    indices=clusterPointsIndices,
                )

            facetBaseClusters = runBatched(range(len(kmeansClusterCenters)),
               getInputs=getInputsFunc,
               processBatch=processBatchFunc,
               processOutput=processOutputFunc,
               batchSize=cfg.llmBatchSize,
               verbose=cfg.verbose)
            baseClusters[facetI] = facetBaseClusters
    return baseClusters

def getFacetValuesEmbeddings(
        facets: List[FacetMetadata],
        facetValues: List[ConversationFacetData],
        embeddingModel: SentenceTransformer,
        cfg: OpenClioConfig) -> List[Optional[EmbeddingArray]]:
    """
    Gets the embeddings of all facet values that have shouldMakeFacetClusters(facet) True
    (this is when the facet has a summaryCriteria that is not None)
    Returns one element for each facet value
    That element will either be None if shouldMakeFacetClusters(facet) is False,
    or a numpy array of size [numConversations, embeddingDim]
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
        vertex_model: Any,
        data: List[Any],
        cfg: OpenClioConfig
    ) -> List[ConversationFacetData]:
    """
    Gets ALL facet values for each data point in a single LLM call using structured outputs.
    Returns a list of ConversationFacetData objects, one for each data point.
    """
    facet_field_names = list(facetSchema.model_fields.keys())

    def getInputsFunc(data_point: Any) -> str:
        data_point = cfg.getConversationFunc(data_point)

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

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((Exception,)),
    )
    def processBatchFunc(batchOfPrompts: List[str]) -> List[str]:
        nonlocal seed
        seed += 1
        results = []

        # Convert schema for Vertex AI
        schema_dict = pydantic_to_vertex_schema(facetSchema)

        for prompt in tqdm(batchOfPrompts, desc="Extracting facets", disable=not cfg.verbose):
            try:
                response = vertex_model.generate_content(
                    prompt,
                    generation_config={
                        "response_mime_type": "application/json",
                        "response_schema": schema_dict,
                        "max_output_tokens": cfg.llmExtraInferenceArgs.get("max_tokens", 1000),
                        "temperature": cfg.llmExtraInferenceArgs.get("temperature", 0.7),
                        "top_p": cfg.llmExtraInferenceArgs.get("top_p", 0.8),
                        "top_k": cfg.llmExtraInferenceArgs.get("top_k", 40),
                    }
                )
                if hasattr(response, 'text'):
                    results.append(response.text)
                elif hasattr(response, 'candidates') and len(response.candidates) > 0:
                    results.append(response.candidates[0].content.parts[0].text)
                else:
                    raise ValueError(f"Unexpected response format: {response}")
            except Exception as e:
                cfg.print(f"Error extracting facets: {e}")
                # Return empty JSON on error
                results.append(json.dumps({field: "" for field in facet_field_names}))

            # Rate limiting
            time.sleep(60.0 / 60)  # ~60 requests per minute

        return results

    def processOutputFunc(data_point: Any, data_point_prompt: str, facetOutput: str) -> ConversationFacetData:
        facet_values = []

        try:
            # Parse the JSON output containing ALL facets
            parsed = json.loads(facetOutput)

            # Extract each facet value
            for facet, field_name in zip(facets, facet_field_names):
                value = parsed.get(field_name, "").strip()
                facet_values.append(FacetValue(facet=facet, value=value))

        except (json.JSONDecodeError, KeyError) as e:
            cfg.print(f"Failed to parse facet output: {e}")
            # Create empty facet values on error
            for facet in facets:
                facet_values.append(FacetValue(facet=facet, value=""))

        return ConversationFacetData(
            conversation=data_point,
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

def getMedoidSummaryAndName(outputs: List[str], embeddingModel: SentenceTransformer) -> Tuple[str, str]:
    """
    Continuous version of "get most common"
    That gets the embedded value that is closest to all other items (the medoid)
    returns (summary, name)
    """
    summaries = []
    names = []
    for output in outputs:
        # re.DOTALL makes . match newlines too (by default it does not)
        matches = re.findall(r"(.*?)</summary>.*?<name>(.*?)</name>", output, re.DOTALL)
        if len(matches) > 0:
            for summary, name in matches:
                summaries.append(removePunctuation(summary.strip()))
                names.append(removePunctuation(name.strip()))
    # remove empty strings
    summaries = [summary for summary in summaries if len(summary) > 0]
    names = [name for name in names if len(name) > 0]
    if len(summaries) == 0: summaries.append("<Could not extract summary>")
    if len(names) == 0: names.append("<Could not extract name>")
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
    parents: List[ConversationCluster],
    indent: str) -> List[str]:
    lines = []
    for parent in parents:
        lines.append(indent + parent.name)
        if not parent.children is None:
            lines += printHierarchyHelper(parent.children, indent + "  ")
    return lines

def printHierarchy(parents: List[ConversationCluster]):
    """
    helper function to manually print hierarchy of a specific facet
    """
    resLines = printHierarchyHelper(parents, indent="")
    print("\n".join(resLines))
    with open("hierarchy.txt", "w") as f:
        f.write("\n".join(resLines))