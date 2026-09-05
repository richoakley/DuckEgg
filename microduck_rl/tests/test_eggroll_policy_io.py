"""Exact deployed-policy import, forward, adaptation, and EGGROLL tests."""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from mjlab_microduck.eggroll.policy import (
    OutputLayerPolicy,
    PostTrainingPolicyConfig,
)
from mjlab_microduck.eggroll.policy_io import (
    EXPECTED_WIDTHS,
    export_adapted_policy,
    import_deployed_policy,
    numpy_actions,
    onnx_actions,
)


def make_policy(
    path: Path, *, seed: int = 7, widths: tuple[int, ...] = EXPECTED_WIDTHS
) -> Path:
    rng = np.random.default_rng(seed)
    initializers = [
        numpy_helper.from_array(np.zeros((1, 61), np.float32), name="mean"),
        numpy_helper.from_array(np.ones((1, 61), np.float32), name="denominator"),
    ]
    nodes = [
        helper.make_node("Sub", ["obs", "mean"], ["centered"]),
        helper.make_node("Div", ["centered", "denominator"], ["normalized"]),
    ]
    input_name = "normalized"
    for index, (in_width, out_width) in enumerate(pairwise(widths)):
        weight_name = f"weight_{index}"
        bias_name = f"bias_{index}"
        output_name = "actions" if index == len(widths) - 2 else f"linear_{index}"
        initializers.extend(
            (
                numpy_helper.from_array(
                    rng.normal(0.0, 0.02, (out_width, in_width)).astype(np.float32),
                    name=weight_name,
                ),
                numpy_helper.from_array(
                    rng.normal(0.0, 0.02, out_width).astype(np.float32),
                    name=bias_name,
                ),
            )
        )
        nodes.append(
            helper.make_node(
                "Gemm",
                [input_name, weight_name, bias_name],
                [output_name],
                transB=1,
            )
        )
        if index < len(widths) - 2:
            elu_name = f"elu_{index}"
            nodes.append(helper.make_node("Elu", [output_name], [elu_name], alpha=1.0))
            input_name = elu_name
    graph = helper.make_graph(
        nodes,
        "deployed-policy-test",
        [helper.make_tensor_value_info("obs", TensorProto.FLOAT, [1, 61])],
        [helper.make_tensor_value_info("actions", TensorProto.FLOAT, [1, 14])],
        initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, path)
    return path


def test_exact_import_matches_onnx_runtime(tmp_path: Path) -> None:
    source = make_policy(tmp_path / "source.onnx")
    policy = import_deployed_policy(source)
    observations = np.random.default_rng(1).normal(size=(8, 61)).astype(np.float32)
    expected = numpy_actions(policy, observations)
    actual = onnx_actions(policy.source_model, observations)
    np.testing.assert_allclose(actual, expected, rtol=1.0e-5, atol=1.0e-6)
    assert policy.metadata()["trainable_parameters"] == 1_806


def test_import_discovers_hidden_layers_without_assuming_standup_widths(
    tmp_path: Path,
) -> None:
    source = make_policy(tmp_path / "small.onnx", widths=(61, 32, 16, 14))
    policy = import_deployed_policy(source)
    assert policy.widths == (61, 32, 16, 14)
    observations = np.random.default_rng(3).normal(size=(4, 61)).astype(np.float32)
    np.testing.assert_allclose(
        onnx_actions(policy.source_model, observations),
        numpy_actions(policy, observations),
        rtol=1.0e-5,
        atol=1.0e-6,
    )


def test_import_rejects_a_nearby_but_different_graph(tmp_path: Path) -> None:
    source = make_policy(tmp_path / "source.onnx")
    model = onnx.load(source)
    model.graph.node[3].op_type = "Relu"
    model.graph.node[3].ClearField("attribute")
    onnx.save(model, source)
    try:
        import_deployed_policy(source)
    except ValueError as error:
        assert "Unsupported deployed policy graph" in str(error)
    else:
        raise AssertionError("Importer accepted a different activation graph")


def test_adapted_export_changes_only_output_and_remains_equivalent(
    tmp_path: Path,
) -> None:
    source = make_policy(tmp_path / "source.onnx")
    policy = import_deployed_policy(source)
    weight = policy.output_weight + np.float32(0.001)
    bias = policy.output_bias - np.float32(0.002)
    output = tmp_path / "adapted.onnx"
    maximum_error = export_adapted_policy(
        policy, output_weight=weight, output_bias=bias, output_path=output
    )
    adapted = import_deployed_policy(output)
    for original, changed in zip(policy.layers[:-1], adapted.layers[:-1], strict=True):
        np.testing.assert_array_equal(original.weight, changed.weight)
        np.testing.assert_array_equal(original.bias, changed.bias)
    np.testing.assert_array_equal(adapted.output_weight, weight)
    np.testing.assert_array_equal(adapted.output_bias, bias)
    assert maximum_error < 1.0e-5


def test_output_layer_policy_is_antithetic_and_updates(tmp_path: Path) -> None:
    deployed = import_deployed_policy(make_policy(tmp_path / "source.onnx"))
    policy = OutputLayerPolicy(
        deployed,
        PostTrainingPolicyConfig(sigma=0.01, learning_rate=0.003, rank=2, seed=11),
    )
    observations = jnp.ones((2, 61), dtype=jnp.float32)
    candidate = np.asarray(policy.candidate_actions(observations, generation=0))
    base = np.asarray(policy.base_actions(observations))
    source_before = np.asarray(policy.source_actions(observations))
    np.testing.assert_allclose(candidate.mean(axis=0), base[0], atol=2.0e-6)
    before = policy.output_parameters()
    policy.update(np.asarray([1.0, 0.0], dtype=np.float32), generation=0)
    after = policy.output_parameters()
    assert policy.trainable_parameter_count == 1_806
    assert not np.array_equal(before[0], after[0])
    assert not np.array_equal(before[1], after[1])
    np.testing.assert_array_equal(policy.source_actions(observations), source_before)
    assert not np.array_equal(policy.base_actions(observations), source_before)


def test_output_layer_checkpoint_state_round_trips_scalar_optimizer_leaves(
    tmp_path: Path,
) -> None:
    deployed = import_deployed_policy(make_policy(tmp_path / "source.onnx"))
    config = PostTrainingPolicyConfig(sigma=0.015, learning_rate=0.003, rank=4, seed=11)
    source = OutputLayerPolicy(deployed, config)
    source.update(np.asarray([1.0, 0.0], dtype=np.float32), generation=0)
    expected = source.output_parameters()

    restored = OutputLayerPolicy(deployed, config)
    restored.load_state_dict(source.state_dict())
    actual = restored.output_parameters()

    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])
