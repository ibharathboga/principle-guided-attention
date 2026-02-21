"""
PGA Claude v0 — Integration Tests.

Tests every stage of the PGA pipeline individually and end-to-end.

Run:
    cd pga-claude.v0
    pip install -r requirements.txt
    python -m pytest test_pipeline.py -v
"""

from __future__ import annotations

import asyncio
import warnings

import pytest
import torch

# Import from the package
from .config import PGAConfig
from .errors import (
    BufferColdStartWarning,
    IncompleteInformationError,
    PrincipleExtractionError,
)
from .models import (
    DecodedResult,
    Observation,
    PipelineTrace,
    PrincipleMatrix,
    StateVector,
    UserQuery,
)
from .observation_encoder import MockLLMClient, ObservationEncoder
from .epistemic_buffer import EpistemicBuffer
from .principle_engine import PrincipleExtractor
from .pga_layer import PGALayer
from .clarity_decoder import ClarityDecoder
from .pipeline import PGAPipeline


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Fixtures
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@pytest.fixture
def config():
    return PGAConfig(
        d_model=32,
        n_heads=4,
        n_layers=1,
        buffer_capacity=64,
        retrieval_top_k=4,
        svd_rank=4,
        entropy_threshold=6.0,
        chroma_collection_name="test_collection",
    )


@pytest.fixture
def strict_config():
    """Config with very low entropy threshold (will trigger guard)."""
    return PGAConfig(
        d_model=32,
        n_heads=4,
        n_layers=1,
        entropy_threshold=0.01,
        chroma_collection_name="test_strict_collection",
    )


@pytest.fixture
def sample_state_vector():
    return StateVector(
        tensor=torch.randn(32),
        observations=[
            Observation(name="mass", value=100.0, unit="kg", certainty=0.95),
            Observation(name="velocity", value=10.0, unit="m/s", certainty=0.90),
        ],
        source_certainty=0.925,
        entropy_level=1.5,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Test 1: Pydantic Model Validation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class TestModels:
    def test_observation_creation(self):
        obs = Observation(name="mass", value=100.0, unit="kg", certainty=0.95)
        assert obs.name == "mass"
        assert obs.value == 100.0
        assert obs.certainty == 0.95

    def test_observation_certainty_bounds(self):
        with pytest.raises(Exception):
            Observation(name="x", value=1.0, certainty=1.5)  # > 1.0
        with pytest.raises(Exception):
            Observation(name="x", value=1.0, certainty=-0.1)  # < 0.0

    def test_state_vector_tensor_coercion(self):
        sv = StateVector(tensor=[1.0, 2.0, 3.0])
        assert isinstance(sv.tensor, torch.Tensor)
        assert sv.tensor.dtype == torch.float32

    def test_state_vector_auto_id(self):
        sv1 = StateVector(tensor=torch.zeros(4))
        sv2 = StateVector(tensor=torch.zeros(4))
        assert sv1.id != sv2.id  # UUIDs should be unique

    def test_user_query_requires_text(self):
        with pytest.raises(Exception):
            UserQuery(raw_text="")  # min_length=1

    def test_principle_matrix_creation(self):
        pm = PrincipleMatrix(
            W_P=torch.eye(8),
            extraction_method="svd",
            explained_variance_ratio=0.95,
            rank=4,
        )
        assert pm.W_P.shape == (8, 8)
        assert pm.rank == 4


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Test 2: ObservationEncoder (Stage 1)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class TestObservationEncoder:
    @pytest.mark.asyncio
    async def test_encode_physics_text(self, config):
        encoder = ObservationEncoder(config, llm_client=MockLLMClient())
        sv = await encoder.encode("A bridge handles heavy load with steel")
        assert isinstance(sv, StateVector)
        assert sv.tensor.shape == (config.d_model,)
        assert len(sv.observations) > 0
        obs_names = [o.name for o in sv.observations]
        assert "structure_type" in obs_names  # "bridge" keyword
        assert "applied_force" in obs_names  # "load" keyword

    @pytest.mark.asyncio
    async def test_encode_produces_unit_vector(self, config):
        encoder = ObservationEncoder(config)
        sv = await encoder.encode("mass and velocity measurement")
        norm = torch.norm(sv.tensor).item()
        assert abs(norm - 1.0) < 0.01  # Should be L2-normalized

    @pytest.mark.asyncio
    async def test_encode_unknown_text_fallback(self, config):
        encoder = ObservationEncoder(config)
        sv = await encoder.encode("xyzzy qwerty foobar")
        assert len(sv.observations) == 1
        assert sv.observations[0].source == "hash_fallback"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Test 3: EpistemicBuffer (Stage 2)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class TestEpistemicBuffer:
    @pytest.mark.asyncio
    async def test_store_and_retrieve(self, config, sample_state_vector):
        buffer = EpistemicBuffer(config)
        entry_id = await buffer.store(sample_state_vector)
        assert entry_id == sample_state_vector.id
        assert buffer.size == 1

        # Retrieve with the same vector as query
        results = await buffer.retrieve(sample_state_vector, top_k=1)
        assert len(results) == 1
        assert results[0].id == sample_state_vector.id

    @pytest.mark.asyncio
    async def test_cold_start_warning(self, config):
        buffer = EpistemicBuffer(config)
        query = StateVector(tensor=torch.randn(config.d_model))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            results = await buffer.retrieve(query)
            assert len(results) == 0
            assert any(issubclass(x.category, BufferColdStartWarning) for x in w)

    @pytest.mark.asyncio
    async def test_metadata_storage(self, config, sample_state_vector):
        buffer = EpistemicBuffer(config)
        await buffer.store(sample_state_vector)
        meta = await buffer.get_metadata(sample_state_vector.id)
        assert meta is not None
        assert abs(meta.source_certainty - 0.925) < 0.01
        assert meta.observation_count == 2

    @pytest.mark.asyncio
    async def test_buffer_reset(self, config, sample_state_vector):
        buffer = EpistemicBuffer(config)
        await buffer.store(sample_state_vector)
        assert buffer.size == 1
        buffer.reset()
        assert buffer.size == 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Test 4: PrincipleExtractor (Stage 3)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class TestPrincipleExtractor:
    def test_extract_with_observations(self, config):
        extractor = PrincipleExtractor(config)
        query = StateVector(tensor=torch.randn(config.d_model))
        retrieved = [
            StateVector(tensor=torch.randn(config.d_model)) for _ in range(5)
        ]
        pm = extractor.extract(query, retrieved)
        assert isinstance(pm, PrincipleMatrix)
        assert pm.W_P.shape == (config.d_model, config.d_model)
        assert pm.extraction_method == "svd"
        assert pm.explained_variance_ratio > 0.0
        assert pm.rank > 0

    def test_extract_cold_start_fallback(self, config):
        extractor = PrincipleExtractor(config)
        query = StateVector(tensor=torch.randn(config.d_model))
        pm = extractor.extract(query, retrieved=[])
        assert pm.extraction_method == "identity_fallback"
        assert pm.W_P.shape == (config.d_model, config.d_model)

    def test_svd_explained_variance_bounded(self, config):
        extractor = PrincipleExtractor(config)
        query = StateVector(tensor=torch.randn(config.d_model))
        retrieved = [
            StateVector(tensor=torch.randn(config.d_model)) for _ in range(10)
        ]
        pm = extractor.extract(query, retrieved)
        assert 0.0 <= pm.explained_variance_ratio <= 1.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Test 5: PGALayer (Stage 4)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class TestPGALayer:
    def test_forward_shape(self, config):
        layer = PGALayer(config.d_model, config.n_heads)
        x = torch.randn(2, 5, config.d_model)  # (batch=2, seq=5, d=32)
        pm = PrincipleMatrix(W_P=torch.eye(config.d_model))

        result = layer(x, pm)
        assert result.output_tensor.shape == (2, 5, config.d_model)
        assert result.attention_weights.shape == (2, config.n_heads, 5, 5)

    def test_forward_with_svd_principle(self, config):
        layer = PGALayer(config.d_model, config.n_heads)
        extractor = PrincipleExtractor(config)

        query = StateVector(tensor=torch.randn(config.d_model))
        retrieved = [
            StateVector(tensor=torch.randn(config.d_model)) for _ in range(3)
        ]
        pm = extractor.extract(query, retrieved)

        x = torch.randn(1, 4, config.d_model)
        result = layer(x, pm)
        assert result.output_tensor.shape == (1, 4, config.d_model)
        assert result.principle_applied is True

    def test_attention_weights_sum_to_one(self, config):
        layer = PGALayer(config.d_model, config.n_heads)
        x = torch.randn(1, 3, config.d_model)
        pm = PrincipleMatrix(W_P=torch.eye(config.d_model))

        result = layer(x, pm)
        # Check that attention weights per head sum to 1 across key dimension
        sums = result.attention_weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Test 6: ClarityDecoder (Stage 5)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class TestClarityDecoder:
    def test_decode_passes_within_threshold(self, config):
        decoder = ClarityDecoder(config)
        essence = torch.randn(config.d_model) * 5.0  # Strong signal
        pm = PrincipleMatrix(
            W_P=torch.eye(config.d_model),
            explained_variance_ratio=0.9,
            rank=4,
        )
        trace = PipelineTrace(
            extracted_observations=[
                Observation(name="mass", value=100.0, unit="kg")
            ]
        )
        result = decoder.decode(essence, pm, trace)
        assert isinstance(result, DecodedResult)
        assert result.confidence > 0
        assert trace.clarity_passed is True

    def test_decode_raises_on_high_entropy(self, strict_config):
        decoder = ClarityDecoder(strict_config)
        # Near-uniform tensor → high entropy
        essence = torch.ones(strict_config.d_model) * 0.01
        pm = PrincipleMatrix(W_P=torch.eye(strict_config.d_model))
        trace = PipelineTrace()

        with pytest.raises(IncompleteInformationError) as exc_info:
            decoder.decode(essence, pm, trace)

        assert exc_info.value.global_entropy > strict_config.entropy_threshold
        assert len(exc_info.value.missing_dimensions) > 0
        assert trace.clarity_passed is False


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Test 7: Full Pipeline Integration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class TestPipeline:
    @pytest.mark.asyncio
    async def test_ingest_and_process(self, config):
        pipeline = PGAPipeline(config)

        # Ingest some observations
        await pipeline.ingest("A bridge with heavy load and steel structure")
        await pipeline.ingest("Concrete mass under gravity tension")
        assert pipeline.buffer.size == 2

        # Process a query
        query = UserQuery(raw_text="bridge load steel tension")
        result = await pipeline.process(query)

        assert isinstance(result, DecodedResult)
        assert result.confidence > 0
        assert result.trace.trace_id != ""
        assert result.trace.total_duration_ms > 0
        assert "observer" in result.trace.stage_durations_ms
        assert "buffer" in result.trace.stage_durations_ms
        assert "principle" in result.trace.stage_durations_ms
        assert "pga_layers" in result.trace.stage_durations_ms
        assert "decoder" in result.trace.stage_durations_ms

    @pytest.mark.asyncio
    async def test_pipeline_cold_start(self, config):
        """Pipeline should work even with an empty buffer."""
        pipeline = PGAPipeline(config)
        query = UserQuery(raw_text="bridge load")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", BufferColdStartWarning)
            result = await pipeline.process(query)
        assert isinstance(result, DecodedResult)
        assert result.trace.principle_method == "identity_fallback"

    @pytest.mark.asyncio
    async def test_pipeline_feedback_loop(self, config):
        """After processing a query, the buffer should grow (feedback)."""
        pipeline = PGAPipeline(config)
        await pipeline.ingest("mass velocity pressure temperature")
        initial_size = pipeline.buffer.size

        query = UserQuery(raw_text="mass pressure")
        await pipeline.process(query)

        # Buffer should have grown by 1 (the essence was stored back)
        assert pipeline.buffer.size == initial_size + 1

    @pytest.mark.asyncio
    async def test_pipeline_trace_populated(self, config):
        """Verify all trace fields are populated after a full run."""
        pipeline = PGAPipeline(config)
        await pipeline.ingest("steel bridge load")

        query = UserQuery(raw_text="bridge steel")
        result = await pipeline.process(query)

        trace = result.trace
        assert trace.query_id == query.id
        assert trace.raw_input == query.raw_text
        assert len(trace.extracted_observations) > 0
        assert trace.encoded_state_vector_id != ""
        assert trace.principle_method in ("svd", "identity_fallback")
        assert trace.clarity_passed is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
