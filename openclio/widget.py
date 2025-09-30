"""Interactive widget for exploring Clio results using anywidget"""

import anywidget
import traitlets
import json
import base64
from io import BytesIO
from typing import List, Optional
import numpy as np

from .opencliotypes import OpenClioResults, ConversationCluster, shouldMakeFacetClusters


class ClioWidget(anywidget.AnyWidget):
    """
    Interactive widget for exploring Clio analysis results.

    Uses anywidget for cross-platform compatibility (Jupyter, Colab, VSCode).
    """

    # Widget state synchronized between Python and JavaScript
    _facets = traitlets.List([]).tag(sync=True)
    _selected_facet_idx = traitlets.Int(0).tag(sync=True)
    _plot_data = traitlets.Unicode("").tag(sync=True)  # Base64 encoded plot image
    _clusters = traitlets.List([]).tag(sync=True)
    _texts = traitlets.List([]).tag(sync=True)

    # Simple HTML/JS for the widget (no external files needed)
    _esm = """
    function render({ model, el }) {
        // Create container
        el.innerHTML = `
            <div style="display: flex; gap: 20px; font-family: sans-serif;">
                <div style="flex: 1;">
                    <div style="margin-bottom: 10px;">
                        <label>Facet: </label>
                        <select id="facet-select" style="padding: 5px; font-size: 14px;"></select>
                    </div>
                    <div id="plot-container" style="width: 100%; max-width: 600px;"></div>
                </div>
                <div style="flex: 1; overflow-y: auto; max-height: 600px;">
                    <h4>Cluster Hierarchy</h4>
                    <div id="cluster-tree" style="font-size: 13px;"></div>
                    <h4 style="margin-top: 20px;">Selected Texts</h4>
                    <div id="text-viewer" style="font-size: 12px;"></div>
                </div>
            </div>
        `;

        const facetSelect = el.querySelector('#facet-select');
        const plotContainer = el.querySelector('#plot-container');
        const clusterTree = el.querySelector('#cluster-tree');
        const textViewer = el.querySelector('#text-viewer');

        // Populate facet dropdown
        function updateFacetDropdown() {
            const facets = model.get('_facets');
            facetSelect.innerHTML = facets.map((f, i) =>
                `<option value="${i}">${f}</option>`
            ).join('');
            facetSelect.value = model.get('_selected_facet_idx');
        }

        // Update plot
        function updatePlot() {
            const plotData = model.get('_plot_data');
            if (plotData) {
                plotContainer.innerHTML = `<img src="${plotData}" style="max-width: 100%; height: auto;" />`;
            } else {
                plotContainer.innerHTML = '<p>No plot available</p>';
            }
        }

        // Update cluster tree
        function updateClusters() {
            const clusters = model.get('_clusters');
            if (clusters && clusters.length > 0) {
                clusterTree.innerHTML = clusters.map((cluster, i) =>
                    `<div style="margin: 5px 0; padding: 5px; cursor: pointer; border: 1px solid #ddd; border-radius: 3px;"
                          onmouseover="this.style.backgroundColor='#f0f0f0'"
                          onmouseout="this.style.backgroundColor='white'"
                          onclick="selectCluster(${i})">
                        ${cluster.name} (${cluster.count} items)
                    </div>`
                ).join('');
            } else {
                clusterTree.innerHTML = '<p>No clusters available</p>';
            }
        }

        // Update text viewer
        function updateTexts() {
            const texts = model.get('_texts');
            if (texts && texts.length > 0) {
                textViewer.innerHTML = texts.map((text, i) =>
                    `<div style="margin: 10px 0; padding: 10px; background: #f9f9f9; border-radius: 3px;">
                        <strong>Text ${i + 1}:</strong><br/>
                        ${text.substring(0, 200)}${text.length > 200 ? '...' : ''}
                    </div>`
                ).join('');
            } else {
                textViewer.innerHTML = '<p>Select a cluster to view texts</p>';
            }
        }

        // Global function for cluster selection
        window.selectCluster = function(idx) {
            model.send({ type: 'select_cluster', cluster_idx: idx });
        };

        // Event listeners
        facetSelect.addEventListener('change', (e) => {
            model.set('_selected_facet_idx', parseInt(e.target.value));
            model.save_changes();
        });

        // Initial render
        updateFacetDropdown();
        updatePlot();
        updateClusters();
        updateTexts();

        // Listen for changes
        model.on('change:_facets', updateFacetDropdown);
        model.on('change:_plot_data', updatePlot);
        model.on('change:_clusters', updateClusters);
        model.on('change:_texts', updateTexts);
    }

    export default { render };
    """

    def __init__(self, results: OpenClioResults):
        """
        Initialize widget with Clio results.

        Args:
            results: OpenClioResults object from runClio()
        """
        self.results = results
        self.selected_facet_idx = 0
        self.selected_cluster_indices = None

        # Find first facet with clusters
        for i, facet in enumerate(results.facets):
            if shouldMakeFacetClusters(facet) and results.rootClusters[i] is not None:
                self.selected_facet_idx = i
                break

        # Set initial state
        self._facets = [f.name for f in results.facets if shouldMakeFacetClusters(f)]
        self._selected_facet_idx = self.selected_facet_idx

        # Generate initial plot
        self._update_plot()
        self._update_clusters()

        super().__init__()

        # Listen for custom messages from JavaScript
        self.on_msg(self._handle_custom_msg)

    def _handle_custom_msg(self, content, buffers):
        """Handle messages from JavaScript frontend"""
        msg_type = content.get('type')

        if msg_type == 'select_cluster':
            cluster_idx = content.get('cluster_idx')
            self._on_cluster_selected(cluster_idx)

    def _update_plot(self):
        """Generate plot as base64-encoded PNG"""
        import matplotlib.pyplot as plt

        facet_idx = self.selected_facet_idx
        umap_coords = self.results.umap[facet_idx]

        if umap_coords is None:
            self._plot_data = ""
            return

        # Create matplotlib plot
        fig, ax = plt.subplots(figsize=(8, 6))

        # Plot all points
        ax.scatter(umap_coords[:, 0], umap_coords[:, 1],
                  c='lightblue', s=20, alpha=0.6, label='Data points')

        # Highlight selected cluster if any
        if self.selected_cluster_indices is not None and len(self.selected_cluster_indices) > 0:
            selected_coords = umap_coords[self.selected_cluster_indices]
            ax.scatter(selected_coords[:, 0], selected_coords[:, 1],
                      c='red', s=40, alpha=0.8, label='Selected', edgecolors='darkred')

        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.set_title(f"UMAP Projection - {self.results.facets[facet_idx].name}")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Convert to base64
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        self._plot_data = f"data:image/png;base64,{img_base64}"

        plt.close(fig)

    def _update_clusters(self):
        """Update cluster list"""
        facet_idx = self.selected_facet_idx
        root_clusters = self.results.rootClusters[facet_idx]

        if root_clusters is None:
            self._clusters = []
            return

        # Flatten cluster hierarchy for simple display
        cluster_list = []

        def add_cluster(cluster, depth=0):
            indices = self._get_cluster_indices(cluster)
            cluster_list.append({
                'name': '  ' * depth + cluster.name,
                'count': len(indices),
                'indices': indices.tolist()
            })
            if cluster.children:
                for child in cluster.children:
                    add_cluster(child, depth + 1)

        for root_cluster in root_clusters:
            add_cluster(root_cluster)

        self._clusters = cluster_list

    def _get_cluster_indices(self, cluster: ConversationCluster) -> np.ndarray:
        """Recursively get all indices belonging to a cluster"""
        if cluster.children is None:
            return cluster.indices if cluster.indices is not None else np.array([])
        else:
            indices = []
            for child in cluster.children:
                indices.extend(self._get_cluster_indices(child))
            return np.array(indices)

    def _on_cluster_selected(self, cluster_idx):
        """Handle cluster selection"""
        if cluster_idx < len(self._clusters):
            cluster = self._clusters[cluster_idx]
            self.selected_cluster_indices = np.array(cluster['indices'])

            # Update plot with highlighted cluster
            self._update_plot()

            # Update texts
            texts = []
            for idx in cluster['indices'][:10]:  # Show first 10
                if idx < len(self.results.data):
                    texts.append(self.results.data[idx])

            self._texts = texts

    @traitlets.observe('_selected_facet_idx')
    def _on_facet_change(self, change):
        """Handle facet selection change"""
        self.selected_facet_idx = change['new']
        self.selected_cluster_indices = None
        self._update_plot()
        self._update_clusters()
        self._texts = []


def test_widget_components():
    """Test if anywidget works in this environment"""
    print("Testing anywidget...")

    try:
        import anywidget
        import traitlets

        class TestWidget(anywidget.AnyWidget):
            _value = traitlets.Int(0).tag(sync=True)
            _esm = """
            function render({ model, el }) {
                el.innerHTML = '<div style="padding: 20px; background: #e3f2fd; border-radius: 5px;"><h3>✓ anywidget works!</h3><p>The widget system is functional.</p></div>';
            }
            export default { render };
            """

        print("✓ anywidget imported successfully")
        print("Creating test widget...")

        test = TestWidget()
        print("✓ Test widget created")
        print("\nWidget should appear below:")

        return test

    except Exception as e:
        print(f"✗ anywidget test failed: {e}")
        import traceback
        traceback.print_exc()
        return None
