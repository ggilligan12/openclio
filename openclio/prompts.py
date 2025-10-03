"""
Simplified prompt templates for OpenClio.
No XML, no verbose instructions, no unnecessary abstractions.
"""

from typing import List


def truncateText(text: str, maxChars: int = -1) -> str:
    """Simple text truncation to maxChars characters."""
    if maxChars > 0:
        return text[:maxChars]
    return text


def getFacetClusterNamePrompt(facet, clusterFacetValues, clusterOutsideValues):
    """
    Generate cluster name and summary using Pydantic structured output.
    Output format is enforced by response_schema=ClusterNameAndSummary.
    """
    items = "\n".join(clusterFacetValues)
    return f"""Name this cluster.

Items:
{items}

Requirements: {facet.summaryCriteria}
Be concise (name: 2-4 words, summary: 6-10 words)."""


def getNeighborhoodClusterNamesPrompt(facet, clusters, desiredNames):
    """
    Propose ~desiredNames higher-level cluster names.
    Output format: numbered list parsed by extractAnswerNumberedList().
    """
    cluster_list = "\n".join([f"{c.name}: {c.summary}" for c in clusters])
    return f"""Create {desiredNames} broader category names for these clusters.

Clusters:
{cluster_list}

Requirements: {facet.summaryCriteria}

Output {desiredNames} category names (±50% is fine).
Format as numbered list:
1. Category name one
2. Category name two
..."""


def getDeduplicateClusterNamesPrompt(facet, clusters, desiredNames):
    """
    Deduplicate cluster names to ~desiredNames distinct names.
    Output format: numbered list parsed by extractAnswerNumberedList().
    """
    cluster_list = "\n".join(clusters)
    return f"""Deduplicate these cluster names to ~{desiredNames} distinct names.

Cluster names:
{cluster_list}

Requirements: {facet.summaryCriteria}

Merge similar names. Keep specific over generic.
Output {desiredNames} names (±50% is fine).
Format as numbered list:
1. Name one
2. Name two
..."""


def getAssignToHighLevelClusterPrompt(clusterToAssign, higherLevelClusters):
    """
    Assign a cluster to its best parent category.
    Output format: single line parsed by extractTagValue(output, "answer").
    """
    categories = "\n".join(higherLevelClusters)
    return f"""Assign this cluster to the best matching category.

Cluster to assign:
Name: {clusterToAssign.name}
Description: {clusterToAssign.summary}

Available categories:
{categories}

Pick the single best match. Output only the category name."""


def getRenamingHigherLevelClusterPrompt(facet, clusters):
    """
    Rename a higher-level cluster based on its children.
    Uses Pydantic structured output (ClusterNameAndSummary).
    """
    cluster_list = "\n".join([c.name for c in clusters])
    return f"""Name this parent cluster based on its children.

Child clusters:
{cluster_list}

Requirements: {facet.summaryCriteria}
Be concise (name: 2-4 words, summary: 6-10 words)."""


# Legacy text parsing utilities
def removePunctuation(output: str) -> str:
    """Removes ., ?, ! from end of string."""
    output = output.strip()
    while output and output[-1] in '.?!':
        output = output[:-1].strip()
    return output


def extractTagValue(output: str, tag: str) -> tuple[bool, str]:
    """
    Extract value from <tag>VALUE</tag>.
    Returns (found, value).
    """
    import re
    pattern = f"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, output, re.DOTALL | re.IGNORECASE)
    if match:
        return True, match.group(1).strip()
    return False, ""


def extractAnswerNumberedList(output: str, ignoreNoTrailing: bool = False) -> List[str]:
    """
    Extract numbered list from output.
    Expected format:
    1. Item one
    2. Item two
    3. Item three

    Returns list of items without numbers.
    """
    import re

    # Remove any leading/trailing whitespace
    output = output.strip()

    # Find all numbered items (handles "1.", "1)", etc.)
    items = []
    for line in output.split('\n'):
        line = line.strip()
        # Match lines starting with number followed by . or )
        match = re.match(r'^\d+[\.)]\s*(.*)', line)
        if match:
            item = match.group(1).strip()
            if item:  # Only add non-empty items
                items.append(item)

    return items


def removeNumberFromOutput(output: str) -> str:
    """
    Remove leading "1. " or "1) " from string.
    Examples:
      "1. hi" -> "hi"
      "144. wow" -> "wow"
    """
    import re
    return re.sub(r'^\d+[\.)]\s*', '', output.strip())
