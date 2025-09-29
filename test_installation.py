#!/usr/bin/env python3
"""Quick test to verify OpenClio installation"""

def test_imports():
    """Test that all key modules can be imported"""
    print("Testing OpenClio installation...\n")

    try:
        import openclio
        print("✓ openclio module imported")
    except ImportError as e:
        print(f"✗ Failed to import openclio: {e}")
        return False

    # Test key exports
    exports_to_test = [
        'VertexLLMInterface',
        'LLMInterface',
        'systemPromptFacets',
        'runClio',
        'Facet',
        'OpenClioConfig',
        'ClioWidget',
    ]

    all_good = True
    for export in exports_to_test:
        if hasattr(openclio, export):
            print(f"✓ {export} available")
        else:
            print(f"✗ {export} NOT FOUND")
            all_good = False

    print()

    if all_good:
        print("✅ All tests passed! OpenClio is ready to use.")
        print("\nExample usage:")
        print("""
import openclio
from sentence_transformers import SentenceTransformer

llm = openclio.VertexLLMInterface(
    model_name="gemini-1.5-flash",
    project_id="your-project-id"
)

embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

results = openclio.runClio(
    facets=openclio.systemPromptFacets,
    llm=llm,
    embeddingModel=embedding_model,
    data=["You are a helpful assistant...", ...],
    outputDirectory="./output",
    displayWidget=True,
)
        """)
    else:
        print("❌ Some tests failed. Try reinstalling:")
        print("  pip install --upgrade --force-reinstall git+https://github.com/ggilligan12/openclio.git")

    return all_good

if __name__ == "__main__":
    test_imports()
