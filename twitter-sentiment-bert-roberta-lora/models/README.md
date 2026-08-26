# Model artifacts

## What is here

| Path | Size | Contents |
|---|---:|---|
| `baseline-svm/tfidf_vectorizer.joblib` | 727 KB | Fitted `TfidfVectorizer` (unigrams, fit on train split only) |
| `baseline-svm/svm_linearsvc.joblib` | 842 KB | Fitted `LinearSVC` (`max_iter=5000`, one-vs-rest) |
| `roberta-lora-r32/adapter_model.safetensors` | 6.8 MB | LoRA adapter weights, `r=32`, best configuration |
| `roberta-lora-r32/adapter_config.json` | 1 KB | PEFT config — targets `query` and `value`, `lora_alpha=16`, `lora_dropout=0.1` |
| `roberta-lora-r32/tokenizer.json` + `tokenizer_config.json` | 3.4 MB | RoBERTa byte-level BPE tokenizer |
| `roberta-lora-r32/base_config.json` | 1 KB | Config of the fine-tuned `roberta-base` classification head |

## What is NOT here, and why

The **full fine-tuned RoBERTa checkpoint** (`model.safetensors`, ~498 MB) is not committed. GitHub rejects any single file over **100 MB**, and Git LFS on a free account only gives 1 GB of storage and 1 GB of monthly bandwidth — a 500 MB checkpoint burns through both fast.

Two better options if you want to publish it:

1. **Hugging Face Hub** (recommended, free, purpose-built for model weights):
   ```bash
   pip install huggingface_hub
   huggingface-cli login
   huggingface-cli upload <your-username>/roberta-tweeteval-sentiment ./path/to/checkpoint
   ```
   Then link the Hub repo from the main README.

2. **Reproduce it.** Run section 4.3 of `notebooks/final_twitter_code.ipynb` — roughly 22 minutes on a Colab T4.

## Loading the LoRA adapter

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

base = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=3)
model = PeftModel.from_pretrained(base, "models/roberta-lora-r32")
tokenizer = AutoTokenizer.from_pretrained("models/roberta-lora-r32")
```

Label mapping: `0 = negative`, `1 = neutral`, `2 = positive`.
