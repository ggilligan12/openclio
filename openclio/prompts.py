from typing import List, Dict, Optional, Any
from collections import defaultdict


def format_messages_as_text(messages: List[Dict[str, str]]) -> str:
    """
    Simple message formatter when tokenizer/chat_template not available.
    Formats messages as plain text.
    """
    result = []
    for msg in messages:
        role = msg.get('role', '').capitalize()
        content = msg.get('content', '')
        result.append(f"{role}:\n{content}")
    return "\n\n".join(result)


global cachedTokenizer
cachedTokenizer = None
global replacementCache
replacementCache = {}

def doCachedReplacements(funcName, tokenizer, getMessagesFunc, replacementsDict, tokenizerArgs):
    """
    Optimization to substantially speed up tokenization by caching the results and doing string substitutions at the end.
    If tokenizer is None, uses simple text formatting instead.
    Requires putting REPLACE at the end of each thing you replace, I did this to avoid overlaps with existing stuff in the data
    """
    global cachedTokenizer
    global replacementCache

    # if they change tokenizer, reset cache
    if cachedTokenizer != (tokenizerArgs, tokenizer):
        replacementCache = {}
        cachedTokenizer = (tokenizerArgs, tokenizer)

    if not funcName in replacementCache:
        messages = getMessagesFunc()

        if tokenizer is None:
            # No tokenizer available - use simple text formatting
            prompt = format_messages_as_text(messages)
        else:
            # Use tokenizer's chat template
            try:
                inputs = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=True, return_tensors="pt", continue_final_message=True, **tokenizerArgs)
                prompt = tokenizer.decode(inputs['input_ids'][0])
            except:
                # Fall back to simple formatting if chat template fails
                prompt = format_messages_as_text(messages)

        replacementCache[funcName] = prompt

    prompt = replacementCache[funcName]
    for key, value in replacementsDict.items():
        prompt = prompt.replace("{" + key + "REPLACE}", str(value))
    return prompt
        

def getSummarizeFacetPrompt(tokenizer, facet, data, cfg, dataToStr, tokenizerArgs):
    return doCachedReplacements(
        funcName="simpleFacetPrompt",
        tokenizer=tokenizer,
        getMessagesFunc=lambda: [
            {
                "role": "user",
                "content": """Please summarize the provided data in a single sentence:

<data>
{dataREPLACE}
</data>

Put your answer in this format:

<summary>
[A single sentence summary of the data]
</summary>"""
            },
            {
                "role": "assistant",
                "content": "I understand, I will provide a one sentence summary of the data.\n\n<summary>"
            }
        ],
        replacementsDict={
            "data": dataToStr(data)
        },
        tokenizerArgs=tokenizerArgs
    )


def truncateText(text: str, maxChars: int = -1) -> str:
    """Simple text truncation to maxChars characters."""
    if maxChars > 0:
        return text[:maxChars]
    return text


def getFacetPrompt(tokenizer, facet, text, cfg, tokenizerArgs):
    # Truncate text if needed
    truncated_text = truncateText(text, maxChars=cfg.maxTextChars)

    return doCachedReplacements(
        funcName="getFacetPrompt",
        tokenizer=tokenizer,
        getMessagesFunc=lambda: [
            {
                "role": "user",
                "content": """{questionREPLACE}

Text: {textREPLACE}

Return JSON with your answer (1-2 sentences, no PII or proper nouns)."""
            }],
        replacementsDict = {
            "text": truncated_text,
            "question": facet.question,
        },
        tokenizerArgs=tokenizerArgs
    )
    

def getFacetClusterNamePrompt(tokenizer, facet, clusterFacetValues, clusterOutsideValues, tokenizerArgs):
    # Generate name and description for cluster, from G.5
    inClusterStr = "\n".join(clusterFacetValues)
    outOfClusterStr = "\n".join(clusterOutsideValues)

    return doCachedReplacements(
        funcName="getFacetClusterNamePrompt",
        tokenizer=tokenizer,
        getMessagesFunc=lambda: [
        {
            "role": "user",
            "content": """Summarize these related statements with a 1-sentence summary and a short name (max 6 words).

Requirements: {summaryCriteriaREPLACE}

Statements:
{inClusterStrREPLACE}

Return JSON: {"summary": "...", "name": "..."}""",
            }
        ],
        replacementsDict= {
            "summaryCriteria": facet.summaryCriteria,
            "inClusterStr": inClusterStr,
            "outOfClusterStr": outOfClusterStr
        },
        tokenizerArgs=tokenizerArgs
    )



# Proposing cluster names per neighborhood, from G.7.1
def getNeighborhoodClusterNamesPrompt(facet, tokenizer, clusters, desiredNames, tokenizerArgs):
    clusterStr = "\n".join([f"<cluster> {cluster.name}: {cluster.summary} </cluster>" for cluster in clusters])

    return doCachedReplacements(
        funcName="getNeighborhoodClusterNamesPrompt",
        tokenizer=tokenizer,
        getMessagesFunc=lambda: [
            {
                "role": "user",
                "content": """You are tasked with creating higher-level cluster names based on a given list of clusters and their descriptions.
Your goal is to come up with broader categories that could encompass one or more of the provided clusters.

First, review the list of clusters and their descriptions:
<cluster_list>
{clusterStrREPLACE}
</cluster_list>

Your task is to create roughly {desiredNamesREPLACE} higher-level cluster names that could potentially include one or more of the provided clusters.
These higher-level clusters should represent broader categories or themes that emerge from the given clusters, while remaining as specific as possible.
If there are many clusters with a specific theme, ensure that the higher-level cluster name remains the maximum level of specificity.
You are helping to organize user behavior data in order to improve safety, monitoring, and observability.
You can generate more or less than {desiredNamesREPLACE} names if you feel that more or fewer are appropriate and accurately capture the clusters.
You should output at least {desiredNamesLowerREPLACE} and at most {desiredNamesUpperREPLACE} names, with {desiredNamesREPLACE} as a target.

Guidelines for creating higher-level cluster names:
1. Analyze the themes, topics, or characteristics common to multiple clusters.
2. Create names that are specific enough to be meaningful, but not so specific that they can’t meaningfully represent many different clusters. Avoid overly general or vague terms, and do not hesitate to describe socially harmful or sensitive topics (in fact, clusters that clearly describe harmful behavior are slightly preferred); specificity is necessary for observability and enforcement.
3. Ensure that the higher-level cluster names are distinct from one another.
4. Use clear, concise, and descriptive language for the cluster names.
Assume neither good nor bad faith for the content in the clusters.
The names you propose must follow these requirements:
<criteria> {summaryCriteriaREPLACE} </criteria>
Before providing your final list, use a scratchpad to brainstorm and refine your ideas. Think about the relationships between the given clusters and potential overarching themes.
<scratchpad>
[Use this space to analyze the clusters, identify common themes, and brainstorm potential higher-level cluster names. Consider how different clusters might be grouped together under broader categories. No longer than a paragraph or two.]
</scratchpad>
Now, provide your list of roughly {desiredNamesREPLACE} higher-level cluster names.
Present your answer in the following format:
<answer>
1. [First higher-level cluster name]
2. [Second higher-level cluster name]
3. [Third higher-level cluster name]
...
{desiredNamesREPLACE}. [Last higher-level cluster name]
</answer>
Focus on creating meaningful, distinct, and precise (but not overly specific) higher-level cluster names that could encompass multiple sub-clusters."""
            },
            {
                "role": "assistant",
                "content": "I understand. I’ll evaluate the clusters and provide higher-level cluster names that could encompass multiple sub-clusters.\n<scratchpad>"
            }
        ],
        replacementsDict = {
            "clusterStr": clusterStr,
            "desiredNames": desiredNames,
            "desiredNamesLower": int(0.5 * desiredNames),
            "desiredNamesUpper": int(1.5 * desiredNames),
            "summaryCriteria": facet.summaryCriteria,
        },
        tokenizerArgs=tokenizerArgs,
    )


# Deduplicating cluster names across neighborhoods, from G.7.1
def getDeduplicateClusterNamesPrompt(facet, tokenizer, clusters, desiredNames, tokenizerArgs):
    clusterStr = "\n".join([f"<cluster> {cluster} </cluster>" for cluster in clusters])
    # I modified to no less than 1 in case it has lots of the same thing
    return doCachedReplacements(
        funcName="getDeduplicateClusterNamesPrompt",
        tokenizer=tokenizer,
        getMessagesFunc=lambda: [
            {
                "role": "user",
                "content": """You are tasked with deduplicating a list of cluster names into a smaller set of distinct cluster names.
Your goal is to create approximately {desiredNamesREPLACE} relatively distinct clusters that best represent the original list.
You are helping to organize user behavior data in order to improve safety, monitoring, and observability.

Here are the inputs:
<cluster_names>
{clusterStrREPLACE}
</cluster_names>

Number of distinct clusters to create: approximately {desiredNamesREPLACE}
Follow these steps to complete the task:
1. Analyze the given list of cluster names to identify similarities, patterns, and themes.
2. Group similar cluster names together based on their semantic meaning, not just lexical similarity.
3. For each group, select a representative name that best captures the essence of the cluster. This can be one of the original names or a new name that summarizes the group effectively. Do not just pick the most vague or generic name.
4. Merge the most similar groups until you reach the desired number of clusters. Maintain as much specificity as possible while merging.
5. Ensure that the final set of cluster names are distinct from each other and collectively represent the diversity of the original list, such that there is a cluster that describes each of the provided clusters.
6. If you create new names for any clusters, make sure they are clear, concise, and reflective of the contents they represent.
7. You do not need to come up with exactly {desiredNamesREPLACE} names, but aim for no less than 1 and no more than {desiredNamesUpperREPLACE}. Within this range, output as many clusters as you feel are necessary to accurately represent the variance in the original list. Avoid outputting duplicate or near-duplicate clusters.
9. Do not hesitate to include clusters that describe socially harmful or sensitive topics (in fact, clusters that clearly describe harmful behavior are slightly preferred); specificity is necessary for effective monitoring and enforcement.
10. Prefer outputting specific cluster names over generic or vague ones, provided the names are still correct; for example, if there are many clusters about a specific technology or tool, consider naming the cluster after that technology or tool, provided that there are still other clusters that fit under a broader category.

The names you propose must follow these requirements:
<criteria>{summaryCriteriaREPLACE}</criteria>

Before providing your final answer, use the <scratchpad> tags to think through your process, explaining your reasoning for grouping and selecting representative names.
Spend no more than a few paragraphs in your scratchpad.
Present your final answer in the following format:
<answer>
1. [First cluster name]
2. [Second cluster name]
3. [Third cluster name]
...
N. [Nth cluster name]
</answer>
Remember, your goal is to create approximately {desiredNamesREPLACE} relatively distinct cluster names that best represent the original list.
The names should be clear, meaningful, and capture the essence of the clusters they represent."""
            },
            {
                "role": "assistant",
                "content": "I understand. I’ll deduplicate the cluster names into approximately {desiredNamesREPLACE} names.\n<scratchpad>"
            }
        ],
        replacementsDict = {
            "summaryCriteria": facet.summaryCriteria,
            "desiredNames": desiredNames,
            "desiredNamesUpper": int(1.5 * desiredNames),
            "clusterStr": clusterStr
        },
        tokenizerArgs=tokenizerArgs
    )

def getAssignToHighLevelClusterPrompt(tokenizer, clusterToAssign, higherLevelClusters, tokenizerArgs):
    clustersStr = "\n".join([f"<cluster> {cluster} </cluster>" for cluster in higherLevelClusters])
    # Assigning to higher-level clusters G.7.1
    return doCachedReplacements(
        funcName="getAssignToHighLevelClusterPrompt",
        tokenizer=tokenizer,
        getMessagesFunc=lambda: [
            {
                "role": "user",
                "content": """You are tasked with categorizing a specific cluster into one of the provided higher-level clusters for observability, monitoring, and content moderation.
Your goal is to determine which higher-level cluster best fits the given specific cluster based on its name and description.
You are helping to organize user behavior data in order to improve safety, monitoring, and observability.

First, carefully review the following list of higher-level clusters:

<higher_level_clusters>
{clustersStrREPLACE}
</higher_level_clusters>

To categorize the specific cluster:
1. Analyze the name and description of the specific cluster.
2. Consider the key characteristics, themes, or subject matter of the specific cluster.
3. Compare these elements to the higher-level clusters provided.
4. Determine which higher-level cluster best encompasses the specific cluster. You MUST assign the specific cluster to the best higher-level cluster, even if multiple higher-level clusters could be considered.
5. Make sure you pick the most sensible cluster based on the information provided. For example, don’t assign a cluster about "Machine Learning" to a higher-level cluster about "Social Media" just because both involve technology, and don’t assign a cluster about "Online Harassment" to a higher-level cluster about "Technology" just because both involve online platforms.

Be specific and accurate in your categorization.

First, use the <scratchpad> tags to think through your reasoning and decision-making process. Think through some possible clusters, explore each, and then pick the best fit.

<scratchpad>
In a few brief sentences, think step by step, explain your reasoning, and finally determine which higher-level cluster is the best fit for the specific cluster.
</scratchpad>

Then, provide your answer in the following format:

<answer>
[Full name of the chosen cluster, exactly as listed in the higher-level
clusters above, without enclosing <cluster> tags]
</answer>""",
            },
            {
                "role": "assistant",
                "content": "I understand. I’ll evaluate the specific cluster and assign it to the most appropriate higher-level cluster."
            },
            {
                "role": "user",
                "content": """Now, here is the specific cluster to categorize:
<specific_cluster>
Name: {clusterToAssignNameREPLACE}
Description: {clusterToAssignSummaryREPLACE}
</specific_cluster>

Based on this information, determine the most appropriate higher-level cluster and provide your answer as instructed."""
            },
            {
                "role": "assistant",
                "content": "Thank you, I will reflect on the cluster and categorize it most appropriately, which will help with safety, moderation, and observability.\n<scratchpad>"
            }
        ],
        replacementsDict = {
            "clusterToAssignName": clusterToAssign.name,
            "clusterToAssignSummary": clusterToAssign.summary,
            "clustersStr": clustersStr,
        },
        tokenizerArgs=tokenizerArgs
    )


# Renaming higher level clusters from G.7.1
def getRenamingHigherLevelClusterPrompt(facet, tokenizer, clusters, tokenizerArgs):
    clusterStr = "\n".join([f'<cluster> {cluster.name} </cluster>' for cluster in clusters])
    return doCachedReplacements(
        funcName="getRenamingHigherLevelClusterPrompt",
        tokenizer=tokenizer,
        getMessagesFunc=lambda: [
        {
                "role": "user",
                "content": """Summarize these cluster names with a 1-sentence summary and a short name (max 6 words).

Requirements: {summaryCriteriaREPLACE}

Clusters:
{clusterStrREPLACE}

Return JSON: {"summary": "...", "name": "..."}""",
            }
        ],
        replacementsDict = {
            "summaryCriteria": facet.summaryCriteria,
            "clusterStr": clusterStr
        },
        tokenizerArgs=tokenizerArgs
    )