from src.llm_eval.config import load_yaml_config
from src.llm_eval.normalization import MetricRegistry


def test_minmax_normalization():
    cfg = load_yaml_config("src/llm_eval/config/defaults.yaml", "src/llm_eval/config/metrics.yaml")
    reg = MetricRegistry(cfg)
    res = reg.normalize("binary_acc", 1.0, {})
    assert 99.0 <= res.normalized <= 100.0
    res0 = reg.normalize("binary_acc", 0.0, {})
    assert 0.0 <= res0.normalized <= 1.0


def test_binary_normalization_for_label_only_match():
    cfg = load_yaml_config("src/llm_eval/config/defaults.yaml", "src/llm_eval/config/metrics.yaml")
    reg = MetricRegistry(cfg)
    res1 = reg.normalize("label_only_match", 1.0, {})
    assert res1.normalized == 100.0
    res0 = reg.normalize("label_only_match", 0.0, {})
    assert res0.normalized == 0.0


if __name__ == "__main__":
    test_minmax_normalization()
    test_binary_normalization_for_label_only_match()
    print("All tests passed!")