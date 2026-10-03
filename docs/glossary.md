# Glossary

Plain-English definitions of the terms used in this project.

| Term | Meaning |
|---|---|
| **Synthetic data** | Artificial data created by a model, meant to behave like real data without being real records. |
| **Turbofan** | The type of jet engine used on most airliners. |
| **Sensor** | An instrument on the engine that records one quantity, such as a temperature or a pressure. |
| **Cycle** | One flight, from take-off to landing. Each row of the data is one engine at one cycle. |
| **Run-to-failure** | A record that follows an engine from new until a part fails. |
| **NASA C-MAPSS / FD001** | A public set of simulated engine records from NASA. FD001 is the subset with one operating condition and one type of wear. |
| **Ps30, T50, P30** | The three sensors studied here: compressor outlet static pressure, low-pressure turbine outlet temperature, and compressor outlet total pressure. |
| **Model** | A program that learns patterns from data and can then make predictions or new data. |
| **Training** | Showing a model real data repeatedly so it learns the patterns. |
| **Epoch** | One full pass of the model over all the training data. |
| **Loss** | A score of how wrong the model's guesses are during training. Lower is better. |
| **Checkpoint** | A saved copy of a trained model, so it can be used later without retraining. |
| **Transformer** | A model that reads a sequence (here, recent cycles) and predicts what comes next. |
| **Input window** | How many past cycles the Transformer reads before predicting the next one (31 here). |
| **Attention** | How much the Transformer weighs each past cycle when making a prediction. |
| **Diffusion model** | A model that learns to remove random noise; it creates new data by starting from noise and cleaning it step by step. |
| **Noise schedule** | How much noise is added at each step of the diffusion process. |
| **Fine-tuning** | Lightly adjusting an already trained model for a specific goal. |
| **LoRA** (low-rank adaptation) | A fine-tuning method that keeps the original model frozen and trains a small add-on (here 1.44% of the size). |
| **Adapter** | The saved LoRA add-on, applied on top of the original model. |
| **Distribution** | The spread of values a sensor takes: which values are common and which are rare. |
| **KS test** (Kolmogorov–Smirnov) | A statistical test that measures the biggest gap between two distributions. |
| **p-value** | The chance of seeing a difference as large as the one observed if both datasets truly came from the same source. 0.05 or higher counts as "no meaningful difference". |
| **Similarity** | 100% minus the KS statistic: how closely the two distributions match. |
| **Correlation** | Whether two sensors tend to rise and fall together. |
| **PCA** (principal component analysis) | A way to squeeze several sensors into a two-dimensional picture, so both datasets can be compared on one map. |
| **RUL** (remaining useful life) | How many cycles an engine has left before it fails. |
| **Seed** (random seed) | A fixed starting number for random choices, so the same "random" result can be reproduced exactly. |
| **Replay** | Playing back a recording of a real training run, sped up, instead of running it live. |
