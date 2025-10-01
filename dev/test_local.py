#!/usr/bin/env python3
"""
Quick local test script for OpenClio development.
Run with: python test_local.py
"""

import sys
import os

# Ensure we're using local development version
sys.path.insert(0, os.path.dirname(__file__))

def test_import():
    """Test basic import"""
    print("Testing import...")
    try:
        import openclio
        from sentence_transformers import SentenceTransformer
        from pydantic import BaseModel, Field
        print("✓ Imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


def test_custom_facets():
    """Test defining custom facets"""
    print("\nTesting custom facet definition...")
    try:
        from pydantic import BaseModel, Field
        import openclio

        class TestFacets(BaseModel):
            topic: str = Field(description="Main topic")
            sentiment: str = Field(description="Positive, negative, or neutral")

        metadata = {
            "topic": openclio.FacetMetadata(
                name="Topic",
                summaryCriteria="Cluster by topic"
            )
        }

        print("✓ Custom facets defined successfully")
        print(f"  Fields: {list(TestFacets.model_fields.keys())}")
        print(f"  Metadata: {list(metadata.keys())}")
        return True
    except Exception as e:
        print(f"✗ Custom facets failed: {e}")
        return False


def test_basic_extraction():
    """Test basic facet extraction (requires GCP auth and quota)"""
    print("\nTesting basic extraction (requires GCP auth)...")

    # Check for required environment
    project_id = os.environ.get("GCP_PROJECT_ID")
    if not project_id:
        print("⚠ Skipping (set GCP_PROJECT_ID env var to test)")
        return True

    try:
        import openclio
        from sentence_transformers import SentenceTransformer

        # Minimal test data
        test_data = [
            "You are a helpful assistant that answers questions about Python programming.",
            "You are a creative writing coach that helps with story development.",
        ]

        embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

        print(f"  Using project: {project_id}")
        print(f"  Processing {len(test_data)} items...")

        results = openclio.runClio(
            facetSchema=openclio.SystemPromptFacets,
            facetMetadata=openclio.systemPromptFacetMetadata,
            embeddingModel=embedding_model,
            data=test_data,
            outputDirectory="./test_output",
            project_id=project_id,
            model_name="gemini-1.5-flash-002",
            displayWidget=False,  # No widget in CLI
            llmBatchSize=2,
            verbose=False,
        )

        # Check results
        assert len(results.facetValues) == len(test_data), "Wrong number of results"
        assert len(results.facets) > 0, "No facets extracted"

        print("✓ Basic extraction successful")
        print(f"  Extracted {len(results.facets)} facets from {len(test_data)} items")

        # Show first result
        if results.facetValues:
            print("\n  First result:")
            for fv in results.facetValues[0].facetValues:
                print(f"    {fv.facet.name}: {fv.value[:50]}...")

        return True

    except Exception as e:
        print(f"✗ Extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_vertex_schema_conversion():
    """Test Pydantic to Vertex AI schema conversion"""
    print("\nTesting schema conversion...")
    try:
        from pydantic import BaseModel, Field
        import openclio

        class TestSchema(BaseModel):
            text_field: str = Field(description="A text field")
            number_field: int = Field(description="A number")
            list_field: list[str] = Field(description="A list of strings")

        schema = openclio.pydantic_to_vertex_schema(TestSchema)

        assert "properties" in schema
        assert "text_field" in schema["properties"]
        assert schema["properties"]["text_field"]["type"] == "STRING"
        assert schema["properties"]["number_field"]["type"] == "INTEGER"
        assert schema["properties"]["list_field"]["type"] == "ARRAY"

        print("✓ Schema conversion successful")
        print(f"  Converted {len(schema['properties'])} fields")
        return True

    except Exception as e:
        print(f"✗ Schema conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("=" * 60)
    print("OpenClio Local Development Tests")
    print("=" * 60)

    results = []

    # Basic tests (no GCP needed)
    results.append(("Import", test_import()))
    results.append(("Custom Facets", test_custom_facets()))
    results.append(("Schema Conversion", test_vertex_schema_conversion()))

    # GCP tests (optional)
    results.append(("Basic Extraction", test_basic_extraction()))

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")

    total = len(results)
    passed = sum(1 for _, p in results if p)

    print(f"\n{passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
