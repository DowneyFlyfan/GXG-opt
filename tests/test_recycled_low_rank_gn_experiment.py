import pytest
import torch


def test_controlled_contract_preserves_the_paper_statistical_batches():
    from recycled_low_rank_gn_experiment import validate_kron_rpcg_contract

    contract = validate_kron_rpcg_contract(
        physical_batch_size=8,
        sequence_length=1024,
        outer_effective_batch_size=3904,
        curvature_batch_size=64,
    )

    assert contract == {
        "sequence_length": 1024,
        "physical_batch_size": 8,
        "gradient_accumulation": 488,
        "final_physical_batch_size": 8,
        "outer_effective_batch_size": 3904,
        "outer_effective_tokens": 3_997_696,
        "curvature_batch_size": 64,
        "curvature_accumulation": 8,
        "final_curvature_batch_size": 8,
        "line_search_effective_batch_size": 3904,
    }


def test_controlled_contract_records_an_exact_final_partial_batch():
    from recycled_low_rank_gn_experiment import validate_kron_rpcg_contract

    contract = validate_kron_rpcg_contract(
        physical_batch_size=12,
        sequence_length=1024,
        outer_effective_batch_size=3904,
        curvature_batch_size=64,
    )

    assert contract["gradient_accumulation"] == 326
    assert contract["final_physical_batch_size"] == 4


def test_controlled_contract_can_match_gradient_and_curvature_accumulation():
    from recycled_low_rank_gn_experiment import validate_kron_rpcg_contract

    contract = validate_kron_rpcg_contract(
        physical_batch_size=12,
        sequence_length=1024,
        outer_effective_batch_size=3904,
        curvature_batch_size=3904,
    )

    assert contract["gradient_accumulation"] == 326
    assert contract["curvature_accumulation"] == 326
    assert contract["final_curvature_batch_size"] == 4


def test_sample_weighted_curvature_average_matches_the_combined_batch():
    from kronecker_ggn_common.curvature_operator import (
        FunctionalCurvatureBatch,
        GGNFullOperator,
    )
    from recycled_low_rank_gn_experiment import SampleWeightedGGNOperator

    model = torch.nn.Linear(1, 1, bias=False).double()
    inputs = torch.arange(1.0, 5.0, dtype=torch.float64).reshape(-1, 1)
    targets = torch.zeros_like(inputs)

    def make_batch(selected_inputs, selected_targets):
        return FunctionalCurvatureBatch(
            args=(selected_inputs,),
            loss_fn=lambda output: 0.5
            * (output - selected_targets).square().mean(),
        )

    split = (
        GGNFullOperator(model, make_batch(inputs[:3], targets[:3])),
        GGNFullOperator(model, make_batch(inputs[3:], targets[3:])),
    )
    weighted = SampleWeightedGGNOperator(split, sample_counts=(3, 1))
    combined = GGNFullOperator(model, make_batch(inputs, targets))
    vector = torch.ones(combined.numel, dtype=torch.float64)

    assert torch.allclose(weighted.matvec(vector), combined.matvec(vector))


def test_outer_curvature_collection_keeps_vectorized_physical_batches():
    from recycled_low_rank_gn_experiment import (
        _outer_gradient_and_curvature_batches,
    )

    class TinyLanguageModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(8, 4)
            self.output = torch.nn.Linear(4, 8)

        def forward(self, token_ids):
            return self.output(self.embedding(token_ids))

    model = TinyLanguageModel()
    batches = (
        (torch.tensor([[0, 1], [1, 2], [2, 3]]), torch.tensor([[1, 2], [2, 3], [3, 4]])),
        (torch.tensor([[3, 4], [4, 5]]), torch.tensor([[4, 5], [5, 6]])),
    )

    _, curvature_batches, _ = _outer_gradient_and_curvature_batches(
        model,
        batches,
        curvature_batch_size=4,
        device=torch.device("cpu"),
    )

    assert [batch.args[0].shape[0] for batch in curvature_batches] == [3, 1]


def test_outer_collection_skips_unused_full_second_moment_diagonal():
    from recycled_low_rank_gn_experiment import (
        _outer_gradient_and_curvature_batches,
    )

    class TinyLanguageModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(8, 4)
            self.output = torch.nn.Linear(4, 8)

        def forward(self, token_ids):
            return self.output(self.embedding(token_ids))

    batches = (
        (torch.tensor([[0, 1], [1, 2]]), torch.tensor([[1, 2], [2, 3]])),
    )

    _, _, second_moment = _outer_gradient_and_curvature_batches(
        TinyLanguageModel(),
        batches,
        curvature_batch_size=2,
        device=torch.device("cpu"),
        collect_second_moment=False,
    )

    assert second_moment is None


def test_exact_batch_stream_preserves_leftovers_without_dropping_examples():
    from recycled_low_rank_gn_experiment import ExactBatchStream

    raw_batches = iter(
        (
            (torch.tensor([[0], [1], [2]]), torch.tensor([[10], [11], [12]])),
            (torch.tensor([[3], [4], [5]]), torch.tensor([[13], [14], [15]])),
        )
    )
    stream = ExactBatchStream(raw_batches)

    first = stream.take(4)
    second = stream.take(2)

    assert torch.cat([batch[0] for batch in first + second]).flatten().tolist() == list(range(6))
    assert [batch[0].shape[0] for batch in first] == [3, 1]
    assert [batch[0].shape[0] for batch in second] == [2]
    assert stream.samples_consumed == 6
    assert stream.raw_batches_consumed == 2


def test_direction_line_search_selects_the_best_candidate_and_leaves_it_applied():
    from kronecker_ggn_common.curvature_operator import FunctionalCurvatureBatch
    from recycled_low_rank_gn_experiment import held_out_direction_line_search

    model = torch.nn.Linear(1, 1, bias=False).double()
    with torch.no_grad():
        model.weight.zero_()
    inputs = torch.ones(2, 1, dtype=torch.float64)
    targets = torch.ones(2, 1, dtype=torch.float64)
    batch = FunctionalCurvatureBatch(
        args=(inputs,),
        loss_fn=lambda output: 0.5 * (output - targets).square().mean(),
    )

    result = held_out_direction_line_search(
        model,
        direction=torch.tensor([2.0], dtype=torch.float64),
        batches=(batch,),
        search_range=3,
    )

    assert result.step_size == pytest.approx(0.5)
    assert result.loss == pytest.approx(0.0)
    assert model.weight.item() == pytest.approx(1.0)


def test_direction_line_search_can_tune_a_larger_optimizer_specific_scale():
    from kronecker_ggn_common.curvature_operator import FunctionalCurvatureBatch
    from recycled_low_rank_gn_experiment import held_out_direction_line_search

    model = torch.nn.Linear(1, 1, bias=False).double()
    with torch.no_grad():
        model.weight.zero_()
    inputs = torch.ones(1, 1, dtype=torch.float64)
    targets = torch.full((1, 1), 2.0, dtype=torch.float64)
    batch = FunctionalCurvatureBatch(
        args=(inputs,),
        loss_fn=lambda output: 0.5 * (output - targets).square().mean(),
    )

    result = held_out_direction_line_search(
        model,
        direction=torch.tensor([1.0], dtype=torch.float64),
        batches=(batch,),
        search_range=4,
        initial_step_scale=4.0,
    )

    assert result.step_size == pytest.approx(2.0)
    assert model.weight.item() == pytest.approx(2.0)


def test_progressive_line_search_uses_full_held_out_data_for_the_two_finalists():
    from kronecker_ggn_common.curvature_operator import FunctionalCurvatureBatch
    from recycled_low_rank_gn_experiment import held_out_direction_line_search

    model = torch.nn.Linear(1, 1, bias=False).double()
    with torch.no_grad():
        model.weight.zero_()
    batches = tuple(
        FunctionalCurvatureBatch(
            args=(torch.ones(1, 1, dtype=torch.float64),),
            loss_fn=lambda output: 0.5 * (output - 2.0).square().mean(),
        )
        for _ in range(3)
    )

    result = held_out_direction_line_search(
        model,
        direction=torch.tensor([1.0], dtype=torch.float64),
        batches=batches,
        search_range=4,
        initial_step_scale=4.0,
        screening_sequences=1,
        finalists=2,
    )

    assert result.step_size == pytest.approx(2.0)
    assert result.loss == pytest.approx(0.0)
    assert result.screening_sequences == 1
    assert result.finalist_sequences == 3


def test_direction_line_search_restores_parameters_if_evaluation_fails():
    from kronecker_ggn_common.curvature_operator import FunctionalCurvatureBatch
    from recycled_low_rank_gn_experiment import held_out_direction_line_search

    model = torch.nn.Linear(1, 1, bias=False).double()
    with torch.no_grad():
        model.weight.fill_(3.0)
    batch = FunctionalCurvatureBatch(
        args=(torch.ones(1, 1, dtype=torch.float64),),
        loss_fn=lambda output: (_ for _ in ()).throw(RuntimeError("failure")),
    )

    with pytest.raises(RuntimeError, match="failure"):
        held_out_direction_line_search(
            model,
            direction=torch.tensor([-1.0], dtype=torch.float64),
            batches=(batch,),
            search_range=2,
        )

    assert model.weight.item() == pytest.approx(3.0)


def test_direction_line_search_can_reject_every_nonzero_candidate():
    from kronecker_ggn_common.curvature_operator import FunctionalCurvatureBatch
    from recycled_low_rank_gn_experiment import held_out_direction_line_search

    model = torch.nn.Linear(1, 1, bias=False).double()
    with torch.no_grad():
        model.weight.fill_(1.0)
    inputs = torch.ones(1, 1, dtype=torch.float64)
    batch = FunctionalCurvatureBatch(
        args=(inputs,), loss_fn=lambda output: 0.5 * (output - 1.0).square().mean()
    )

    result = held_out_direction_line_search(
        model,
        direction=torch.tensor([10.0], dtype=torch.float64),
        batches=(batch,),
        search_range=3,
        include_zero_step=True,
    )

    assert result.step_size == 0.0
    assert result.loss == pytest.approx(0.0)
    assert model.weight.item() == pytest.approx(1.0)


def test_checkpoint_restores_model_and_recycled_basis(tmp_path):
    from recycled_low_rank_gn import LowRankResidualCorrection, RecycledResidualState
    from recycled_low_rank_gn_experiment import (
        load_kron_rpcg_checkpoint,
        save_kron_rpcg_checkpoint,
    )

    model = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1.0, 2.0]]))
    recycled = RecycledResidualState(rank=1, refresh_interval=4)
    recycled.correction = LowRankResidualCorrection(
        basis=torch.tensor([[1.0], [0.0]]),
        projected_matrix=torch.tensor([[0.5]]),
    )
    recycled.built_at_step = 3
    path = tmp_path / "checkpoint.pt"
    save_kron_rpcg_checkpoint(
        path,
        model,
        recycled,
        completed_outer_steps=4,
        consumed_training_batches=20,
        elapsed_seconds=12.5,
    )
    restored_model = torch.nn.Linear(2, 1, bias=False)
    restored = RecycledResidualState(rank=1, refresh_interval=4)

    counters = load_kron_rpcg_checkpoint(path, restored_model, restored)

    assert counters == (4, 20, 12.5)
    assert torch.equal(restored_model.weight, model.weight)
    assert restored.built_at_step == 3
    assert restored.correction is not None
    assert torch.equal(restored.correction.basis, recycled.correction.basis)
