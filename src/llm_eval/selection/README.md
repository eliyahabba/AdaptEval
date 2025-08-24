### Selection

Select informative questions for a new model.

Interfaces:
- `QuestionSelector.select(model_profile, k, matrix_df) -> list[str]`

Implementations:
- `TinyBenchmarksSelector`: unified IRT implementation. Uses `py-irt` if available for full statistical modeling, with a fallback to a fast heuristic method. Builds a representative subset of questions.
- `NaiveVarianceSelector`: chooses highest-variance questions across models.
- `MITVSelector`: interview-style with difficulty levels and optional diversity.

Run examples
------------

Python:
```python
import pandas as pd
from llm_eval.matrix import MatrixStorage
from llm_eval.selection import TinyBenchmarksSelector, ModelProfile

df = MatrixStorage("data/processed/matrix.parquet").load()
sel = TinyBenchmarksSelector()
qs = sel.select(ModelProfile(model_name="new"), 5, df)
print(qs)
```

CLI:
```bash
# The 'irt' method now points to the TinyBenchmarks implementation
llm-eval select --method irt --k 5 --model-name new --matrix data/processed/matrix.parquet --out out/selection.json
```


