"""Check orchestration separately from the pending single-output validators."""

import warnings
from pathlib import Path

import pandas as pd
import pytest

import debiaspy
from debiaspy import validate_flows

WRAPPERS = [
    (
        "validate_flow_overall_methods",
        "validate_flow_overall",
        {
            "comparisons": "all",
            "drop_zeros": False,
            "return_joined": False,
        },
    ),
    ("validate_flow_residual_methods", "validate_flow_residuals", {"top_n": 10}),
    (
        "validate_flow_distribution_methods",
        "validate_flow_distribution",
        {
            "comparisons": "all",
            "weight_by": "benchmark_origin_total",
            "return_origin_level": True,
            "return_od_level": False,
        },
    ),
    (
        "validate_flow_residual_structure_methods",
        "validate_flow_residual_structure",
        {
            "comparison": "adjusted_vs_benchmark",
        },
    ),
]


@pytest.fixture
def inputs():
    raw = pd.DataFrame(
        {"origin": ["A", "B"], "destination": ["X", "Y"], "flow": [0, 5]}
    )
    return {
        "method_b": raw.assign(flow_adj=[1, 4]),
        "method_a": raw.assign(flow_adj=[2, 3]),
    }, raw


@pytest.mark.parametrize("wrapper_name,core_name,defaults", WRAPPERS)
def test_defaults_delegate_each_method_and_preserve_result_identity(
    monkeypatch, inputs, wrapper_name, core_name, defaults
):
    outputs, benchmark = inputs
    calls = []
    results = {name: object() for name in outputs}

    def validator(**kwargs):
        calls.append(kwargs)
        return results[kwargs["method_name"]]

    monkeypatch.setattr(validate_flows, core_name, validator, raising=False)
    wrapper = getattr(debiaspy, wrapper_name)
    assert wrapper_name in debiaspy.__all__
    actual = wrapper(outputs, benchmark)
    assert list(actual) == ["method_b", "method_a"]
    assert len(calls) == 2
    for call, method in zip(calls, outputs, strict=True):
        assert call.pop("adj_df") is outputs[method]
        assert call.pop("benchmark_od_df") is benchmark
        assert call == {"method_name": method, **defaults}
        assert actual[method] is results[method]


@pytest.mark.parametrize(
    "wrapper_name,core_name,overrides",
    [
        (
            "validate_flow_overall_methods",
            "validate_flow_overall",
            {
                "comparisons": ["raw_vs_adjusted", "adjusted_vs_benchmark"],
                "drop_zeros": True,
                "return_joined": True,
                "na_rm": False,
                "by_source": True,
            },
        ),
        ("validate_flow_residual_methods", "validate_flow_residuals", {"top_n": 2}),
        (
            "validate_flow_distribution_methods",
            "validate_flow_distribution",
            {
                "comparisons": "raw_vs_benchmark",
                "weight_by": "none",
                "return_origin_level": False,
                "return_od_level": True,
                "epsilon": 1e-6,
            },
        ),
        (
            "validate_flow_residual_structure_methods",
            "validate_flow_residual_structure",
            {
                "comparison": "raw_vs_adjusted",
                "spatial_role": "destination",
                "residual_aggregation": "sum",
                "local_moran": True,
                "local_moran_seed": 7,
            },
        ),
    ],
)
def test_explicit_options_and_custom_columns_are_forwarded(
    monkeypatch, inputs, wrapper_name, core_name, overrides
):
    outputs, benchmark = inputs
    calls = []
    options = {
        **overrides,
        "flow_col_adj": "corrected",
        "flow_col_mpd": "raw",
        "flow_col_bench": "bench",
    }
    monkeypatch.setattr(
        validate_flows, core_name, lambda **kwargs: calls.append(kwargs), raising=False
    )
    getattr(debiaspy, wrapper_name)(outputs, benchmark, **options)
    for call in calls:
        for key, value in options.items():
            assert call[key] == value


@pytest.mark.parametrize(
    "selection,expected",
    [
        (None, ["method_b", "method_a"]),
        ("method_a", ["method_a"]),
        (["method_a", "method_b"], ["method_a", "method_b"]),
        (("method_b",), ["method_b"]),
        ([], []),
    ],
)
def test_structure_selection_order_and_optional_dataframe_identity(
    monkeypatch, inputs, selection, expected
):
    outputs, benchmark = inputs
    neighbors = pd.DataFrame({"area": ["A"], "neighbor": ["B"]})
    calls = []

    def validator(**kwargs):
        calls.append(kwargs)
        return kwargs["method_name"]

    monkeypatch.setattr(
        validate_flows, "validate_flow_residual_structure", validator, raising=False
    )
    result = debiaspy.validate_flow_residual_structure_methods(
        outputs, benchmark, methods=selection, area_neighbors=neighbors
    )
    assert list(result) == expected
    assert [call["method_name"] for call in calls] == expected
    assert all(call["area_neighbors"] is neighbors for call in calls)


@pytest.mark.parametrize(
    "selection,exception,message",
    [
        (["missing", "other"], ValueError, "Missing: missing, other"),
        (["method_a", "method_a"], ValueError, "duplicate"),
        ([""], ValueError, "non-empty string"),
        ([1], ValueError, "non-empty string"),
        ({"method_a"}, TypeError, "ordered sequence"),
    ],
)
def test_structure_selection_is_validated_before_dispatch(
    inputs, selection, exception, message
):
    outputs, benchmark = inputs
    with pytest.raises(exception, match=message):
        debiaspy.validate_flow_residual_structure_methods(
            outputs, benchmark, methods=selection
        )


@pytest.mark.parametrize("wrapper_name,core_name,defaults", WRAPPERS)
def test_missing_core_reports_dependency_and_method(
    monkeypatch, inputs, wrapper_name, core_name, defaults
):
    outputs, benchmark = inputs
    monkeypatch.delattr(validate_flows, core_name, raising=False)
    with pytest.raises(
        NotImplementedError, match=f"method_b.*{core_name}.*not yet available"
    ):
        getattr(debiaspy, wrapper_name)(outputs, benchmark)
    assert getattr(debiaspy, wrapper_name)({}, benchmark) == {}


def test_empty_selection_does_not_require_core(monkeypatch, inputs):
    outputs, benchmark = inputs
    monkeypatch.delattr(
        validate_flows, "validate_flow_residual_structure", raising=False
    )
    assert (
        debiaspy.validate_flow_residual_structure_methods(
            outputs, benchmark, methods=[]
        )
        == {}
    )


@pytest.mark.parametrize("wrapper_name,core_name,defaults", WRAPPERS)
@pytest.mark.parametrize("exception", [ValueError, TypeError, KeyError])
def test_core_errors_retain_cause_and_identify_failing_method(
    monkeypatch, inputs, wrapper_name, core_name, defaults, exception
):
    outputs, benchmark = inputs
    failure = exception("`adj_df` must contain: flow_adj")
    calls = []

    def validator(**kwargs):
        calls.append(kwargs["method_name"])
        if kwargs["method_name"] == "method_a":
            raise failure
        return {"summary": pd.DataFrame({"method": ["method_b"]})}

    monkeypatch.setattr(validate_flows, core_name, validator, raising=False)
    with pytest.raises(exception, match="method_a.*flow_adj") as caught:
        getattr(debiaspy, wrapper_name)(outputs, benchmark)
    assert caught.value.__cause__ is failure
    assert calls == ["method_b", "method_a"]


@pytest.mark.parametrize("wrapper_name,core_name,defaults", WRAPPERS)
def test_core_warnings_propagate_without_altering_results(
    monkeypatch, inputs, wrapper_name, core_name, defaults
):
    outputs, benchmark = inputs

    def validator(**kwargs):
        warnings.warn("Missing benchmark rows", UserWarning, stacklevel=2)
        return None

    monkeypatch.setattr(validate_flows, core_name, validator, raising=False)
    with pytest.warns(UserWarning, match="Missing benchmark rows") as caught:
        result = getattr(debiaspy, wrapper_name)(outputs, benchmark)
    assert len(caught) == 2
    assert result == {"method_b": None, "method_a": None}


@pytest.mark.parametrize("wrapper_name,core_name,defaults", WRAPPERS)
@pytest.mark.parametrize("key", ["method_name", "adj_df"])
def test_mapping_derived_arguments_cannot_be_overridden(
    inputs, wrapper_name, core_name, defaults, key
):
    outputs, benchmark = inputs
    with pytest.raises(TypeError, match=f"method_b.*{key}.*derived"):
        getattr(debiaspy, wrapper_name)(outputs, benchmark, **{key: "override"})


@pytest.mark.parametrize("wrapper_name,core_name,defaults", WRAPPERS)
def test_invalid_method_collection_is_rejected(
    inputs, wrapper_name, core_name, defaults
):
    _, benchmark = inputs
    wrapper = getattr(debiaspy, wrapper_name)
    with pytest.raises(TypeError, match="mapping"):
        wrapper([], benchmark)
    with pytest.raises(ValueError, match="non-empty strings"):
        wrapper({"": benchmark}, benchmark)
    with pytest.raises(TypeError, match="bad.*DataFrame"):
        wrapper({"bad": None}, benchmark)


def test_default_call_traces_match_executed_r_wrappers(monkeypatch, inputs):
    """The fixture captures calls, not numerical validation results."""
    outputs, benchmark = inputs
    rows = []
    for wrapper_name, core_name, _ in WRAPPERS:

        def capture(_wrapper=wrapper_name, **kwargs):
            method = kwargs.pop("method_name")
            kwargs.pop("adj_df")
            kwargs.pop("benchmark_od_df")
            for key, value in kwargs.items():
                text = str(value).upper() if isinstance(value, bool) else str(value)
                rows.append(
                    {
                        "wrapper": _wrapper,
                        "method": method,
                        "argument": key,
                        "value": text,
                    }
                )
            return None

        monkeypatch.setattr(validate_flows, core_name, capture, raising=False)
        getattr(debiaspy, wrapper_name)(outputs, benchmark)
    expected = pd.read_csv(
        Path(__file__).parent / "fixtures" / "flow_method_r_calls.csv", dtype=str
    )
    actual = pd.DataFrame(rows)
    keys = ["wrapper", "method", "argument"]
    pd.testing.assert_frame_equal(
        actual.sort_values(keys).reset_index(drop=True),
        expected.sort_values(keys).reset_index(drop=True),
    )
