"""Проверяет детерминированные нагрузки и матрицу экспериментальных запусков.

Проверяются матрица из 24 запусков и 12 условий, суммарные объемы входных и
выходных токенов, воспроизводимость по seed, состав запросов и правила общего
префикса, а также расчет границы блока, исключающий последний неподходящий
блок при точном совпадении с границей последовательности.
"""

from taei.nano_dataset.workload import (
    build_run_matrix,
    build_workload,
    validate_shared_prefix_boundary,
)


def test_matrix_is_exactly_24_runs_and_12_conditions():
    matrix = build_run_matrix()
    assert len(matrix) == 24
    assert len({row["condition_id"] for row in matrix}) == 12
    assert {row["run_id"] for row in matrix} == {f"R{i:02}" for i in range(1, 25)}
    assert len({row["run_order"] for row in matrix}) == 24
    assert {row["seed"] for row in matrix} == {42, 43}
    assert len({row["pair_id"] for row in matrix}) == 6
    assert len({row["split_group_id"] for row in matrix}) == 3
    for family in ("S", "L", "M"):
        assert (
            len(
                {
                    row["split_group_id"]
                    for row in matrix
                    if row["workload_id"] == family
                }
            )
            == 1
        )
    assert sum(1 for _ in matrix for _ in range(32)) == 768


def test_matrix_logical_work_totals_match_contract():
    from taei.nano_dataset.workload import build_workload

    matrix = build_run_matrix()
    prompts = outputs = 0
    for run in matrix:
        workload = build_workload(run["workload_id"], run["seed"])
        prompts += workload["summary"]["logical_prompt_tokens"]
        outputs += workload["summary"]["requested_output_token_total"]
    assert prompts == 835_584
    assert outputs == 245_760


def test_workloads_are_deterministic_and_obey_shared_prefix_rules():
    for name, expected_count, expected_length in (
        ("S", 32, 128),
        ("L", 0, 2048),
        ("M", 16, 128),
    ):
        a = build_workload(name, seed=42)
        b = build_workload(name, seed=42)
        assert a == b
        assert len(a["requests"]) == 32
        assert a["summary"]["short_count"] == expected_count
        assert len(a["requests"][0]["input_token_ids"]) == expected_length
        expected_output = 512 if name == "L" else 128
        assert {
            r["requested_output_tokens"]
            for r in a["requests"]
            if r["workload_class"] == "short"
        } == ({expected_output} if expected_count else set())
    assert {
        r["requested_output_tokens"] for r in build_workload("L", 42)["requests"]
    } == {512}
    assert {
        r["requested_output_tokens"]
        for r in build_workload("M", 42)["requests"]
        if r["workload_class"] == "long"
    } == {512}
    mixed = build_workload("M", seed=42)["requests"]
    long = [r for r in mixed if r["workload_class"] == "long"]
    assert len(long) == 16
    assert len({tuple(r["input_token_ids"][:1024]) for r in long}) == 1
    assert len({tuple(r["input_token_ids"][1024:]) for r in long}) == 16


def test_mixed_summary_never_labels_heterogeneous_output_lengths_uniform():
    workload = build_workload("M", 42)
    assert "requested_output_tokens_each" not in workload["summary"]
    assert workload["summary"]["requested_output_token_counts"] == [128, 512] * 16


def test_shared_prefix_boundary_uses_sequence_tail_not_prefix_tail():
    assert validate_shared_prefix_boundary(
        block_size=16, prefix_tokens=32, sequence_tokens=64
    ) == {
        "eligible_blocks": 2,
        "excluded_tail_tokens": 0,
        "reason": "upstream_lookup_excludes_last_full_block",
    }
    assert (
        validate_shared_prefix_boundary(
            block_size=16, prefix_tokens=32, sequence_tokens=33
        )["eligible_blocks"]
        == 2
    )
    assert (
        validate_shared_prefix_boundary(
            block_size=16, prefix_tokens=32, sequence_tokens=32
        )["eligible_blocks"]
        == 1
    )
