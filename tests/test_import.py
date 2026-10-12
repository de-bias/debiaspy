import debiaspy


def test_package_imports() -> None:
    assert debiaspy.__version__ == "0.0.0"
    assert callable(debiaspy.measure_bias)
    assert callable(debiaspy.measure_bias_distribution)
    assert callable(debiaspy.validate_bias_residual_structure)
