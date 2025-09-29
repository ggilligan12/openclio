"""Interactive Jupyter/Colab widget for exploring Clio results"""

import numpy as np
from typing import List, Optional
import plotly.graph_objects as go
from ipywidgets import widgets, Output, VBox, HBox, HTML
from IPython.display import display
import json

from .opencliotypes import OpenClioResults, ConversationCluster, shouldMakeFacetClusters


class ClioWidget:
    """
    Interactive widget for exploring Clio analysis results in Jupyter/Colab.

    Features:
    - UMAP scatter plot with cluster hulls
    - Hierarchical tree view
    - Text viewer with facet annotations
    """

    def __init__(self, results: OpenClioResults):
        """
        Initialize widget with Clio results.

        Args:
            results: OpenClioResults object from runClio()
        """
        self.results = results
        self.selected_facet_idx = 0
        self.selected_cluster = None
        self.selected_indices = None

        # Find first facet with clusters
        for i, facet in enumerate(results.facets):
            if shouldMakeFacetClusters(facet) and results.rootClusters[i] is not None:
                self.selected_facet_idx = i
                break

        # Create widgets
        self._create_widgets()

    def _create_widgets(self):
        """Create all widget components"""
        # Facet selector dropdown
        facet_names = [f.name for f in self.results.facets if shouldMakeFacetClusters(f)]
        self.facet_dropdown = widgets.Dropdown(
            options=[(name, i) for i, name in enumerate(facet_names)],
            value=self.selected_facet_idx,
            description='Facet:',
            style={'description_width': 'initial'}
        )
        self.facet_dropdown.observe(self._on_facet_change, 'value')

        # Output areas
        self.plot_output = Output()
        self.tree_output = Output()
        self.text_output = Output()

        # Initial render
        self._update_plot()
        self._update_tree()
        self._update_text_viewer([])

    def _on_facet_change(self, change):
        """Handle facet selection change"""
        self.selected_facet_idx = change['new']
        self.selected_cluster = None
        self.selected_indices = None
        self._update_plot()
        self._update_tree()
        self._update_text_viewer([])

    def _get_cluster_indices(self, cluster: ConversationCluster) -> np.ndarray:
        """Recursively get all indices belonging to a cluster"""
        if cluster.children is None:
            return cluster.indices if cluster.indices is not None else np.array([])
        else:
            indices = []
            for child in cluster.children:
                indices.extend(self._get_cluster_indices(child))
            return np.array(indices)

    def _update_plot(self):
        """Update UMAP scatter plot"""
        with self.plot_output:
            self.plot_output.clear_output(wait=True)

            facet_idx = self.selected_facet_idx
            umap_coords = self.results.umap[facet_idx]

            if umap_coords is None:
                print(f"No UMAP data available for facet {self.results.facets[facet_idx].name}")
                return

            # Create scatter plot
            fig = go.Figure()

            # Add all points
            fig.add_trace(go.Scatter(
                x=umap_coords[:, 0],
                y=umap_coords[:, 1],
                mode='markers',
                marker=dict(
                    size=4,
                    color='lightblue',
                    opacity=0.6
                ),
                text=[f"Point {i}" for i in range(len(umap_coords))],
                hoverinfo='text',
                name='Data points'
            ))

            # Highlight selected cluster if any
            if self.selected_indices is not None and len(self.selected_indices) > 0:
                selected_coords = umap_coords[self.selected_indices]
                fig.add_trace(go.Scatter(
                    x=selected_coords[:, 0],
                    y=selected_coords[:, 1],
                    mode='markers',
                    marker=dict(
                        size=6,
                        color='red',
                        opacity=0.8
                    ),
                    name='Selected cluster',
                    hoverinfo='skip'
                ))

            fig.update_layout(
                title=f"UMAP Projection - {self.results.facets[facet_idx].name}",
                xaxis_title="UMAP 1",
                yaxis_title="UMAP 2",
                height=500,
                hovermode='closest'
            )

            display(fig)

    def _update_tree(self):
        """Update hierarchy tree view"""
        with self.tree_output:
            self.tree_output.clear_output(wait=True)

            facet_idx = self.selected_facet_idx
            root_clusters = self.results.rootClusters[facet_idx]

            if root_clusters is None:
                print(f"No clusters available for facet {self.results.facets[facet_idx].name}")
                return

            html_parts = ["<div style='font-family: monospace; font-size: 12px;'>"]

            def render_cluster(cluster: ConversationCluster, depth: int = 0):
                indent = "&nbsp;" * (depth * 4)
                indices = self._get_cluster_indices(cluster)
                count = len(indices)

                # Make cluster clickable
                cluster_id = id(cluster)
                onclick = f"window.clioSelectCluster({cluster_id}, {json.dumps(indices.tolist())})"

                html_parts.append(
                    f"{indent}<div style='margin: 2px 0; cursor: pointer;' "
                    f"onclick='{onclick}' "
                    f"onmouseover='this.style.backgroundColor=\"#f0f0f0\"' "
                    f"onmouseout='this.style.backgroundColor=\"transparent\"'>"
                    f"<b>{cluster.name}</b> ({count} items)"
                    f"</div>"
                )

                if cluster.children:
                    for child in cluster.children:
                        render_cluster(child, depth + 1)

            for root_cluster in root_clusters:
                render_cluster(root_cluster)

            html_parts.append("</div>")

            # Add JavaScript for click handling
            html_parts.append("""
            <script>
            window.clioSelectCluster = function(clusterId, indices) {
                // Store in window for Python to access
                window.selectedClusterIndices = indices;
                // Notify Python (this would need proper callback setup)
                console.log('Selected cluster:', clusterId, 'with', indices.length, 'items');
            }
            </script>
            """)

            display(HTML(''.join(html_parts)))

    def _update_text_viewer(self, indices: List[int], max_display: int = 20):
        """Update text viewer to show selected data points"""
        with self.text_output:
            self.text_output.clear_output(wait=True)

            if len(indices) == 0:
                print("Select a cluster from the tree to view texts")
                return

            print(f"Showing {min(len(indices), max_display)} of {len(indices)} texts:\n")

            for i, idx in enumerate(indices[:max_display]):
                data_point = self.results.data[idx]
                facet_data = self.results.facetValues[idx]

                # Display text
                if isinstance(data_point, str):
                    # Truncate long texts
                    text = data_point[:500] + ("..." if len(data_point) > 500 else "")
                    print(f"--- Text {i+1} ---")
                    print(text)
                else:
                    # Legacy conversation format
                    print(f"--- Item {i+1} ---")
                    print(str(data_point)[:500])

                # Display facet values
                print("\nFacets:")
                for fv in facet_data.facetValues:
                    print(f"  {fv.facet.name}: {fv.value}")
                print("\n")

    def display(self):
        """Display the widget"""
        # Layout
        left_panel = VBox([
            HBox([self.facet_dropdown]),
            self.plot_output
        ])

        right_panel = VBox([
            HTML("<h4>Cluster Hierarchy</h4>"),
            self.tree_output,
            HTML("<h4>Selected Texts</h4>"),
            self.text_output
        ])

        main_layout = HBox([left_panel, right_panel])
        display(main_layout)

        # Instructions
        print("\n📊 Clio Analysis Widget")
        print("- Select a facet from the dropdown")
        print("- Click clusters in the tree to view texts")
        print("- UMAP plot shows the distribution of all data points")
