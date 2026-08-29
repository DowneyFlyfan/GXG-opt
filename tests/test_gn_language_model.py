import torch
import pytest
from torch import nn

from baseline_kronecker_ggn import KroneckerGGN
from gn_training import nlp_mc_ggn_update
from kronecker_ggn_common.layer_registry import LayerRegistry
from low_rank_corrected_kronecker_ggn import LowRankCorrectedKroneckerGGN
from kronecker_ggn_common.curvature_operator import FunctionalCurvatureBatch, GGNLinearOperator
from models import DecoderTransformer


class TinyCausalLanguageModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(11, 5)
        self.projection = nn.Linear(5, 11, bias=False)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.projection(self.embedding(token_ids))


def test_nlp_mc_ggn_update_captures_linear_factors_and_exact_residual_operator():
    model = TinyCausalLanguageModel()
    registry = LayerRegistry(model)
    token_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    targets = torch.tensor([[2, 3, 4], [5, 6, 7]])

    update = nlp_mc_ggn_update(
        model,
        (token_ids, targets),
        registry,
        include_exact_residual_operator=True,
        seed=19,
    )

    assert update.curvature_mode == "mc_ggn"
    assert set(update.factors) == {"projection"}
    assert set(update.ggn_operators) == {"projection"}
    factor = update.factors["projection"]
    assert factor.activation.shape == (5, 5)
    assert factor.output.shape == (11, 11)
    assert torch.isfinite(factor.activation).all()
    assert torch.isfinite(factor.output).all()


def test_nlp_exact_residual_operator_uses_the_configured_small_curvature_subbatch():
    model = TinyCausalLanguageModel()
    registry = LayerRegistry(model)
    token_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    targets = torch.tensor([[2, 3, 4], [5, 6, 7]])

    update = nlp_mc_ggn_update(
        model,
        (token_ids, targets),
        registry,
        include_exact_residual_operator=True,
        seed=19,
        residual_batch_size=1,
        residual_sequence_length=2,
    )

    operator = update.ggn_operators["projection"]
    residual_tokens, = operator.batch.args
    assert residual_tokens.shape == (1, 2)


def test_language_model_gn_specs_select_the_retained_gpt_and_distinct_optimizers():
    from gn_experiment import LANGUAGE_MODEL_GN_TASK, build_gn_optimizer, language_model_task

    assert LANGUAGE_MODEL_GN_TASK.identifier == "nlp_gpt_12x512"
    assert LANGUAGE_MODEL_GN_TASK.model == "gpt_12x512"
    assert LANGUAGE_MODEL_GN_TASK.micro_batch_size == 4
    assert LANGUAGE_MODEL_GN_TASK.gradient_accumulation == 2

    baseline = build_gn_optimizer(TinyCausalLanguageModel(), "kronecker_ggn")
    corrected = build_gn_optimizer(
        TinyCausalLanguageModel(), "low_rank_corrected_kronecker_ggn"
    )

    assert isinstance(baseline, KroneckerGGN)
    assert isinstance(corrected, LowRankCorrectedKroneckerGGN)
    assert baseline.config.curvature_mode == "mc_ggn"
    assert corrected.config.correction_rank > 0
    assert corrected.config.correction_oversampling >= 7
    assert corrected.config.lanczos_steps >= 8
    assert corrected.config.lanczos_tolerance <= 0.02
    assert corrected.config.correction_warmup_steps > 0

    batch_eight = language_model_task(micro_batch_size=8, gradient_accumulation=1)
    assert (batch_eight.micro_batch_size, batch_eight.gradient_accumulation) == (8, 1)
    assert (LANGUAGE_MODEL_GN_TASK.micro_batch_size, LANGUAGE_MODEL_GN_TASK.gradient_accumulation) == (4, 2)


def test_gn_artifact_paths_keep_language_model_metrics_and_results_separate(tmp_path):
    from gn_experiment import artifact_paths

    paths = artifact_paths(tmp_path, "kronecker_ggn")

    assert paths.metric == tmp_path / "metrics/nlp/nlp_gpt_12x512__kronecker_ggn.jsonl"
    assert paths.result == tmp_path / "results/nlp/nlp_gpt_12x512__kronecker_ggn.json"
    assert paths.checkpoint == tmp_path / ".cache/nlp/checkpoints/nlp_gpt_12x512__kronecker_ggn.checkpoint.pt"


def test_language_model_gn_update_uses_captured_curvature_instead_of_fallback():
    from gn_experiment import build_gn_optimizer

    model = TinyCausalLanguageModel()
    optimizer = build_gn_optimizer(model, "kronecker_ggn")
    batch = (
        torch.tensor([[1, 2, 3], [4, 5, 6]]),
        torch.tensor([[2, 3, 4], [5, 6, 7]]),
    )
    optimizer.update_curvature(
        lambda current_model, current_batch, registry: nlp_mc_ggn_update(
            current_model,
            current_batch,
            registry,
            include_exact_residual_operator=False,
            seed=23,
        ),
        batch,
    )
    logits = model(batch[0])
    torch.nn.functional.cross_entropy(logits.reshape(-1, 11), batch[1].reshape(-1)).backward()
    optimizer.step()

    assert optimizer.last_direction_stats.layers["projection"].used_curvature
    assert optimizer.last_direction_stats.layers["projection"].fallback_reason is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for SDP forward-AD coverage")
def test_attention_language_model_ggn_matvec_forces_the_math_sdp_kernel():
    model = DecoderTransformer(width=16, heads=2, layers=2, vocabulary_size=23).cuda()
    registry = LayerRegistry(model)
    tokens = torch.tensor([[1, 2, 3, 4]], device="cuda")
    targets = torch.tensor([[2, 3, 4, 5]], device="cuda")
    batch = FunctionalCurvatureBatch(
        args=(tokens,),
        loss_fn=lambda logits: torch.nn.functional.cross_entropy(
            logits.reshape(-1, 23), targets.reshape(-1)
        ),
    )
    operator = GGNLinearOperator(model, registry, batch)
    layer = registry.by_id("blocks.0.feedforward_in")

    product = operator.matvec(layer.layer_id, torch.randn_like(layer.weight))

    assert product.shape == layer.weight.shape
    assert torch.isfinite(product).all()
