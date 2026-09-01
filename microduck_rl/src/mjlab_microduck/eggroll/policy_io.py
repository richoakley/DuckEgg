"""Strict interchange with the deployed Microduck PPO ONNX actor.

Post-training starts from the exact artifact loaded by ``robotd``.  This module
therefore accepts one deliberately narrow graph contract and rejects anything
it cannot reproduce exactly instead of guessing at exporter conventions.
"""

from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import ModelProto, checker, numpy_helper

EXPECTED_NODE_TYPES = (
    "Sub",
    "Div",
    "Gemm",
    "Elu",
    "Gemm",
    "Elu",
    "Gemm",
    "Elu",
    "Gemm",
)
EXPECTED_WIDTHS = (61, 512, 256, 128, 14)


@dataclass(frozen=True)
class LinearLayer:
    """One deployed ``Gemm(transB=1)`` layer."""

    weight_name: str
    bias_name: str
    weight: np.ndarray
    bias: np.ndarray


@dataclass(frozen=True)
class DeployedPolicy:
    """Validated deployed actor and the bytes required for a drop-in export."""

    source_sha256: str
    source_model: bytes
    input_name: str
    output_name: str
    normalizer_mean: np.ndarray
    normalizer_denominator: np.ndarray
    layers: tuple[LinearLayer, ...]
    widths: tuple[int, ...]

    @property
    def output_weight(self) -> np.ndarray:
        return self.layers[-1].weight

    @property
    def output_bias(self) -> np.ndarray:
        return self.layers[-1].bias

    def metadata(self) -> dict[str, Any]:
        return {
            "source_sha256": self.source_sha256,
            "input_name": self.input_name,
            "output_name": self.output_name,
            "widths": list(self.widths),
            "activation": "elu",
            "trainable_scope": "output-layer",
            "trainable_parameters": int(
                self.output_weight.size + self.output_bias.size
            ),
        }


def _shape(value_info: onnx.ValueInfoProto) -> tuple[int | str, ...]:
    result: list[int | str] = []
    for dimension in value_info.type.tensor_type.shape.dim:
        if dimension.HasField("dim_value"):
            result.append(int(dimension.dim_value))
        elif dimension.HasField("dim_param"):
            result.append(str(dimension.dim_param))
        else:
            result.append("")
    return tuple(result)


def _attribute(node: onnx.NodeProto, name: str, default: Any = None) -> Any:
    for attribute in node.attribute:
        if attribute.name == name:
            return onnx.helper.get_attribute_value(attribute)
    return default


def _float32_initializer(
    initializers: dict[str, onnx.TensorProto],
    name: str,
    shape: tuple[int, ...],
) -> np.ndarray:
    if name not in initializers:
        raise ValueError(f"ONNX graph is missing initializer {name!r}")
    value = np.asarray(numpy_helper.to_array(initializers[name]))
    if value.dtype != np.float32 or value.shape != shape:
        raise ValueError(
            f"Initializer {name!r} must be float32 shaped {shape}, "
            f"got {value.dtype} {value.shape}"
        )
    if not np.isfinite(value).all():
        raise ValueError(f"Initializer {name!r} contains non-finite values")
    return np.array(value, dtype=np.float32, copy=True)


def _float32_initializer_value(
    initializers: dict[str, onnx.TensorProto], name: str
) -> np.ndarray:
    if name not in initializers:
        raise ValueError(f"ONNX graph is missing initializer {name!r}")
    value = np.asarray(numpy_helper.to_array(initializers[name]))
    if value.dtype != np.float32 or not np.isfinite(value).all():
        raise ValueError(f"Initializer {name!r} must contain finite float32 values")
    return np.array(value, dtype=np.float32, copy=True)


def import_deployed_policy(path: Path) -> DeployedPolicy:
    """Load the exact supported PPO actor graph from a trusted local file."""

    return import_deployed_policy_bytes(path.read_bytes())


def import_deployed_policy_bytes(raw: bytes) -> DeployedPolicy:
    """Validate deployed-policy bytes embedded in a trusted checkpoint."""

    model = onnx.load_model_from_string(raw)
    checker.check_model(model)
    if model.ir_version != 8:
        raise ValueError(f"Expected deployed ONNX IR version 8, got {model.ir_version}")
    opsets = [(entry.domain, entry.version) for entry in model.opset_import]
    if opsets != [("", 18)]:
        raise ValueError(f"Expected only default ONNX opset 18, got {opsets}")
    node_types = tuple(node.op_type for node in model.graph.node)
    tail = node_types[2:]
    structurally_valid = (
        node_types[:2] == ("Sub", "Div")
        and len(tail) >= 1
        and tail[-1] == "Gemm"
        and all(
            node_type == ("Gemm" if index % 2 == 0 else "Elu")
            for index, node_type in enumerate(tail)
        )
    )
    if not structurally_valid:
        raise ValueError(
            "Unsupported deployed policy graph; expected Sub/Div followed by "
            f"Gemm/ELU hidden layers and a final Gemm, got {node_types}"
        )
    if len(model.graph.input) != 1 or len(model.graph.output) != 1:
        raise ValueError("Deployed policy must have exactly one input and one output")
    input_info = model.graph.input[0]
    output_info = model.graph.output[0]
    input_shape = _shape(input_info)
    output_shape = _shape(output_info)
    if input_shape != (1, EXPECTED_WIDTHS[0]):
        raise ValueError(f"Expected policy input [1, 61], got {input_shape}")
    if output_shape != (1, EXPECTED_WIDTHS[-1]):
        raise ValueError(f"Expected policy output [1, 14], got {output_shape}")
    if input_info.type.tensor_type.elem_type != onnx.TensorProto.FLOAT:
        raise ValueError("Policy input must be float32")
    if output_info.type.tensor_type.elem_type != onnx.TensorProto.FLOAT:
        raise ValueError("Policy output must be float32")

    nodes = list(model.graph.node)
    if nodes[0].input[0] != input_info.name:
        raise ValueError("Normalizer Sub must consume the graph input directly")
    if nodes[1].input[0] != nodes[0].output[0]:
        raise ValueError("Normalizer Div must consume the centered observation")
    for index in range(2, len(nodes)):
        if nodes[index].input[0] != nodes[index - 1].output[0]:
            raise ValueError(
                f"Policy graph is not a direct chain at node {index} "
                f"({nodes[index].op_type})"
            )
    initializers = {value.name: value for value in model.graph.initializer}
    mean = _float32_initializer(
        initializers, nodes[0].input[1], (1, EXPECTED_WIDTHS[0])
    ).reshape(EXPECTED_WIDTHS[0])
    denominator = _float32_initializer(
        initializers, nodes[1].input[1], (1, EXPECTED_WIDTHS[0])
    ).reshape(EXPECTED_WIDTHS[0])
    if np.any(denominator <= 0.0):
        raise ValueError("Observation-normalizer denominator must be positive")

    layers: list[LinearLayer] = []
    widths = [EXPECTED_WIDTHS[0]]
    gemm_nodes = [node for node in nodes if node.op_type == "Gemm"]
    for index, node in enumerate(gemm_nodes):
        if len(node.input) != 3:
            raise ValueError(f"Gemm layer {index} must have weight and bias inputs")
        if _attribute(node, "transA", 0) != 0:
            raise ValueError(f"Gemm layer {index} must use transA=0")
        if _attribute(node, "transB", 0) != 1:
            raise ValueError(f"Gemm layer {index} must use transB=1")
        if float(_attribute(node, "alpha", 1.0)) != 1.0:
            raise ValueError(f"Gemm layer {index} must use alpha=1")
        if float(_attribute(node, "beta", 1.0)) != 1.0:
            raise ValueError(f"Gemm layer {index} must use beta=1")
        weight_name, bias_name = node.input[1], node.input[2]
        weight = _float32_initializer_value(initializers, weight_name)
        bias = _float32_initializer_value(initializers, bias_name)
        if weight.ndim != 2 or weight.shape[1] != widths[-1]:
            raise ValueError(
                f"Gemm layer {index} weight must be [out,{widths[-1]}], "
                f"got {weight.shape}"
            )
        out_width = int(weight.shape[0])
        if bias.shape != (out_width,):
            raise ValueError(
                f"Gemm layer {index} bias must be {(out_width,)}, got {bias.shape}"
            )
        widths.append(out_width)
        layers.append(
            LinearLayer(
                weight_name=weight_name,
                bias_name=bias_name,
                weight=weight,
                bias=bias,
            )
        )
    if widths[-1] != EXPECTED_WIDTHS[-1]:
        raise ValueError(
            f"Final structurally discovered output width must be 14, got {widths[-1]}"
        )
    for index, node in enumerate(node for node in nodes if node.op_type == "Elu"):
        if float(_attribute(node, "alpha", 1.0)) != 1.0:
            raise ValueError(f"ELU layer {index} must use alpha=1")
    if gemm_nodes[-1].output[0] != output_info.name:
        raise ValueError("Final Gemm must produce the graph output directly")
    referenced_initializers = {
        nodes[0].input[1],
        nodes[1].input[1],
        *(name for layer in layers for name in (layer.weight_name, layer.bias_name)),
    }
    if set(initializers) != referenced_initializers:
        raise ValueError("Deployed policy has unexpected or unused initializers")

    return DeployedPolicy(
        source_sha256=hashlib.sha256(raw).hexdigest(),
        source_model=raw,
        input_name=input_info.name,
        output_name=output_info.name,
        normalizer_mean=mean,
        normalizer_denominator=denominator,
        layers=tuple(layers),
        widths=tuple(widths),
    )


def numpy_actions(
    policy: DeployedPolicy,
    observations: np.ndarray,
    *,
    output_weight: np.ndarray | None = None,
    output_bias: np.ndarray | None = None,
) -> np.ndarray:
    """Evaluate the imported contract without ONNX Runtime."""

    value = np.asarray(observations, dtype=np.float32)
    if value.ndim != 2 or value.shape[1] != EXPECTED_WIDTHS[0]:
        raise ValueError(f"Observations must be [batch, 61], got {value.shape}")
    value = (value - policy.normalizer_mean) / policy.normalizer_denominator
    for index, layer in enumerate(policy.layers):
        is_output = index == len(policy.layers) - 1
        weight = (
            output_weight if is_output and output_weight is not None else layer.weight
        )
        bias = output_bias if is_output and output_bias is not None else layer.bias
        value = value @ np.asarray(weight, dtype=np.float32).T + np.asarray(
            bias, dtype=np.float32
        )
        if index != len(policy.layers) - 1:
            negative = value <= 0.0
            value[negative] = np.expm1(value[negative])
            value = value.astype(np.float32, copy=False)
    if not np.isfinite(value).all():
        raise FloatingPointError("Deployed policy produced non-finite actions")
    return np.asarray(value, dtype=np.float32)


def onnx_actions(model_bytes: bytes, observations: np.ndarray) -> np.ndarray:
    """Evaluate a serialized actor with deterministic CPU ONNX Runtime."""

    options = ort.SessionOptions()
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        model_bytes, sess_options=options, providers=["CPUExecutionProvider"]
    )
    input_meta = session.get_inputs()[0]
    output_meta = session.get_outputs()[0]
    observations = np.asarray(observations, dtype=np.float32)
    # Production exports use a static batch of one. Preserve that exact runtime
    # contract and evaluate larger verification batches one row at a time.
    if input_meta.shape[0] == 1 and observations.shape[0] != 1:
        rows = [
            session.run([output_meta.name], {input_meta.name: row[None]})[0][0]
            for row in observations
        ]
        return np.asarray(rows, dtype=np.float32)
    return np.asarray(
        session.run([output_meta.name], {input_meta.name: observations})[0],
        dtype=np.float32,
    )


def export_adapted_policy(
    policy: DeployedPolicy,
    *,
    output_weight: np.ndarray,
    output_bias: np.ndarray,
    output_path: Path,
) -> float:
    """Replace only the deployed output layer and prove runtime equivalence."""

    weight = np.asarray(output_weight, dtype=np.float32)
    bias = np.asarray(output_bias, dtype=np.float32)
    if (
        weight.shape != policy.output_weight.shape
        or bias.shape != policy.output_bias.shape
    ):
        raise ValueError("Adapted output parameters do not match the deployed layer")
    if not np.isfinite(weight).all() or not np.isfinite(bias).all():
        raise ValueError("Adapted output parameters contain non-finite values")

    model: ModelProto = copy.deepcopy(onnx.load_model_from_string(policy.source_model))
    replacements = {
        policy.layers[-1].weight_name: numpy_helper.from_array(
            weight, name=policy.layers[-1].weight_name
        ),
        policy.layers[-1].bias_name: numpy_helper.from_array(
            bias, name=policy.layers[-1].bias_name
        ),
    }
    for index, initializer in enumerate(model.graph.initializer):
        replacement = replacements.get(initializer.name)
        if replacement is not None:
            model.graph.initializer[index].CopyFrom(replacement)
    checker.check_model(model)
    rng = np.random.default_rng(20260830)
    observations = rng.normal(size=(32, EXPECTED_WIDTHS[0])).astype(np.float32)
    expected = numpy_actions(
        policy,
        observations,
        output_weight=weight,
        output_bias=bias,
    )
    serialized = model.SerializeToString()
    actual = onnx_actions(serialized, observations)
    maximum_error = float(np.max(np.abs(expected - actual)))
    if maximum_error >= 1.0e-5:
        raise ValueError(f"Adapted ONNX runtime error {maximum_error:.3g} exceeds 1e-5")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.write_bytes(serialized)
    temporary_path.replace(output_path)
    return maximum_error
