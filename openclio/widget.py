"""Interactive Jupyter/Colab widget for exploring Clio results"""

import numpy as np
from typing import List, Optional
import plotly.graph_objects as go
from ipywidgets import widgets, Output, VBox, HBox, HTML
from IPython.display import display, clear_output
import json

from .opencliotypes import OpenClioResults, ConversationCluster, shouldMakeFacetClusters


def test_widget_components():
    """Test if widget components work - call this to debug display issues"""
    print("Testing widget components...\n")

    # Test 1: Basic ipywidgets
    print("1. Testing basic ipywidgets...")
    try:
        test_button = widgets.Button(description="Test Button")
        test_dropdown = widgets.Dropdown(options=['A', 'B', 'C'], description='Test:')
        test_output = Output()
        with test_output:
            print("Output widget works!")
        print("✓ Basic ipywidgets work")
        display(VBox([test_dropdown, test_button, test_output]))
    except Exception as e:
        print(f"✗ Basic ipywidgets failed: {e}\n")
        return

    print("\n2. Testing Plotly with embedded plotly.js...")
    try:
        import plotly.offline as pyo
        from IPython.display import HTML as IPyHTML

        # Initialize plotly offline mode (loads plotly.js)
        pyo.init_notebook_mode(connected=False)

        fig = go.Figure(data=[go.Scatter(x=[1, 2, 3], y=[4, 5, 6], mode='markers', marker=dict(size=10, color='blue'))])
        fig.update_layout(title="Test Plot - Should Be Interactive", width=400, height=300)

        # Include full plotly.js library (not CDN)
        html_str = pyo.plot(fig, include_plotlyjs=True, output_type='div')
        print(f"✓ Generated Plotly div with embedded JS ({len(html_str)} chars)")
        print("   Plot should appear below:")
        display(IPyHTML(html_str))
    except Exception as e:
        print(f"✗ Plotly rendering failed: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n3. Testing HTML in Output widget...")
    try:
        html_output = Output()
        with html_output:
            from IPython.display import HTML as IPyHTML
            display(IPyHTML("<h4 style='color: blue;'>Test HTML</h4><p>This is a test</p>"))
        print("✓ HTML in Output widget - check if HTML appears below")
        display(html_output)
    except Exception as e:
        print(f"✗ HTML in Output widget failed: {e}")
        return

    print("\n4. Testing Plotly with different renderers...")
    try:
        import plotly.io as pio
        print(f"Available renderers: {list(pio.renderers)}")
        print(f"Default renderer: {pio.renderers.default}")

        fig = go.Figure(data=[go.Scatter(x=[1, 2, 3], y=[4, 5, 6], mode='markers+lines', marker=dict(size=10, color='green'))])
        fig.update_layout(title="Test Plot Direct Show", width=400, height=300)

        print("✓ Calling fig.show() directly:")
        fig.show()
    except Exception as e:
        print(f"✗ Plotly show() failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n✓ Tests complete!")


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

        # Output areas for all components
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
        """Update UMAP scatter plot using matplotlib (static but works in widgets)"""
        self.plot_output.clear_output(wait=True)

        with self.plot_output:
            facet_idx = self.selected_facet_idx
            umap_coords = self.results.umap[facet_idx]

            if umap_coords is None:
                print(f"No UMAP data available for facet {self.results.facets[facet_idx].name}")
                return

            # Use matplotlib for reliable rendering in widgets
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(8, 6))

            # Plot all points
            ax.scatter(umap_coords[:, 0], umap_coords[:, 1],
                      c='lightblue', s=20, alpha=0.6, label='Data points')

            # Highlight selected cluster if any
            if self.selected_indices is not None and len(self.selected_indices) > 0:
                selected_coords = umap_coords[self.selected_indices]
                ax.scatter(selected_coords[:, 0], selected_coords[:, 1],
                          c='red', s=40, alpha=0.8, label='Selected cluster', edgecolors='darkred')

            ax.set_xlabel('UMAP 1')
            ax.set_ylabel('UMAP 2')
            ax.set_title(f"UMAP Projection - {self.results.facets[facet_idx].name}")
            ax.legend()
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.show()
            plt.close()

    def _update_tree(self):
        """Update hierarchy tree view"""
        self.tree_output.clear_output(wait=True)

        with self.tree_output:
            facet_idx = self.selected_facet_idx
            root_clusters = self.results.rootClusters[facet_idx]

            if root_clusters is None:
                print(f"No clusters available for facet {self.results.facets[facet_idx].name}")
                return

            tree_widgets = []

            def render_cluster(cluster: ConversationCluster, depth: int = 0):
                indices = self._get_cluster_indices(cluster)
                count = len(indices)
                indent = "  " * depth

                # Create button for this cluster
                button_text = f"{indent}{'└─' if depth > 0 else '▶'} {cluster.name} ({count})"
                btn = widgets.Button(
                    description=button_text,
                    layout=widgets.Layout(width='100%', margin='1px'),
                    button_style='',
                    tooltip=f'Click to view {count} texts',
                    style={'button_color': '#f8f9fa' if depth == 0 else '#ffffff'}
                )

                # Store indices with button for callback
                def on_click(b, cluster_indices=indices.tolist()):
                    self.selected_indices = np.array(cluster_indices)
                    self._update_plot()
                    self._update_text_viewer(cluster_indices)

                btn.on_click(on_click)
                tree_widgets.append(btn)

                # Render children
                if cluster.children:
                    for child in cluster.children:
                        render_cluster(child, depth + 1)

            for root_cluster in root_clusters:
                render_cluster(root_cluster)

            # Display all buttons
            tree_box = VBox(tree_widgets, layout=widgets.Layout(overflow_y='auto', max_height='400px'))
            display(tree_box)

    def _update_text_viewer(self, indices: List[int], max_display: int = 20):
        """Update text viewer to show selected data points"""
        self.text_output.clear_output(wait=True)

        with self.text_output:
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
        # Check if we have any facets with clusters
        has_clusters = any(
            shouldMakeFacetClusters(f) and self.results.rootClusters[i] is not None
            for i, f in enumerate(self.results.facets)
        )

        if not has_clusters:
            print("⚠️  No clusters were generated. This might happen with very small datasets.")
            print(f"   Total data points: {len(self.results.data)}")
            print(f"   Facets analyzed: {', '.join([f.name for f in self.results.facets])}")
            print("\n" + "="*80)
            print("Facet values extracted (showing first 5):")
            print("="*80)
            for i, facet_data in enumerate(self.results.facetValues[:min(5, len(self.results.facetValues))]):
                text = self.results.data[i]
                print(f"\n📄 Text {i+1}: {text[:150]}{'...' if len(text) > 150 else ''}")
                print("-" * 80)
                for fv in facet_data.facetValues:
                    print(f"   • {fv.facet.name}: {fv.value}")
            print("\n" + "="*80)
            print(f"✓ Analysis complete! {len(self.results.facetValues)} texts processed.")
            return

        try:
            self._display_layout()
        except Exception as e:
            print(f"\n⚠️  Widget rendering failed: {e}")
            print("Falling back to text-based display...\n")
            self._display_text_fallback()

    def _display_layout(self):
        """Display the widget layout"""
        # Instructions BEFORE widget (so widget is last output)
        print("📊 Clio Analysis Widget")
        print("- Select a facet from the dropdown")
        print("- Click clusters in the tree to view texts")
        print("- UMAP plot shows the distribution of all data points\n")

        # Debug: check what we have
        print(f"DEBUG: plot_output type: {type(self.plot_output)}")
        print(f"DEBUG: tree_output type: {type(self.tree_output)}")
        print(f"DEBUG: facet_dropdown type: {type(self.facet_dropdown)}")

        # Layout - use Output widgets for all components
        left_panel = VBox([
            HBox([self.facet_dropdown]),
            self.plot_output
        ])

        right_panel = VBox([
            widgets.Label("Cluster Hierarchy"),
            self.tree_output,
            widgets.Label("Selected Texts"),
            self.text_output
        ])

        main_layout = HBox([left_panel, right_panel])

        print(f"DEBUG: About to display main_layout: {type(main_layout)}")

        # MUST be last - no print statements after this!
        display(main_layout)

    def _display_text_fallback(self):
        """Display results as formatted text when widgets don't work"""
        print("="*80)
        print("CLIO ANALYSIS RESULTS")
        print("="*80)
        print(f"\nTotal texts analyzed: {len(self.results.data)}")
        print(f"Facets: {', '.join([f.name for f in self.results.facets])}\n")

        # Show cluster summaries for each facet
        for facet_idx, facet in enumerate(self.results.facets):
            if not shouldMakeFacetClusters(facet):
                continue

            print("\n" + "="*80)
            print(f"FACET: {facet.name}")
            print("="*80)

            root_clusters = self.results.rootClusters[facet_idx]
            if root_clusters:
                print(f"\nFound {len(root_clusters)} top-level clusters:\n")
                for i, cluster in enumerate(root_clusters):
                    indices = self._get_cluster_indices(cluster)
                    print(f"{i+1}. {cluster.name} ({len(indices)} texts)")
                    if cluster.children:
                        for child in cluster.children[:5]:  # Show first 5 children
                            child_indices = self._get_cluster_indices(child)
                            print(f"   └─ {child.name} ({len(child_indices)} texts)")
                        if len(cluster.children) > 5:
                            print(f"   └─ ... and {len(cluster.children) - 5} more")
            else:
                print("No clusters generated for this facet")

        # Show sample facet values
        print("\n" + "="*80)
        print("SAMPLE FACET VALUES (first 5 texts)")
        print("="*80)
        for i, facet_data in enumerate(self.results.facetValues[:min(5, len(self.results.facetValues))]):
            text = self.results.data[i]
            print(f"\n📄 Text {i+1}: {text[:150]}{'...' if len(text) > 150 else ''}")
            print("-" * 80)
            for fv in facet_data.facetValues:
                print(f"   • {fv.facet.name}: {fv.value}")

        print("\n" + "="*80)
        print("✓ Analysis complete!")
        print("="*80)
