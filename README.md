# Usage
```python
import sys
sys.path.append(path_to_pjm_repo)

from pjm import from_pretrained, build_default_alphabet

alphabet = build_default_alphabet()

# Load encoder
embedder = from_pretrained(
    model_type="mmplm",
    alphabet=alphabet,
    checkpoint_path=model_checkpoint_path,
)
```
