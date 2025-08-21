### Selection

Select informative questions for a new model.

Interfaces:
- `QuestionSelector.select(model_profile, k, matrix_df) -> list[str]`

Implementations:
- `IRT2PLSelector`: heuristic 2PL, uses cold-start prior for θ and Fisher information.
- `NaiveVarianceSelector`: chooses highest-variance questions across models.
 - `MITVSelector`: interview-style with difficulty levels and optional diversity.
 - `PyIRTSelector` (optional extras): trains 2PL via py-irt/pyro/torch.

Run examples
------------

Python:
```python
import pandas as pd
from src.llm_eval.selection import IRT2PLSelector, NaiveVarianceSelector, ModelProfile

df = pd.read_csv("examples/tiny_dataset.csv")
if "normalized_score" not in df.columns:
    df = df.assign(normalized_score=(df["raw_score"].astype(float) * 100.0))
sel = IRT2PLSelector()
qs = sel.select(ModelProfile(model_name="new"), 5, df)
print(qs)
```

CLI:
```bash
llm-eval select --method irt --k 5 --model-name new --matrix data/processed/matrix.parquet --out out/selection.json
```


