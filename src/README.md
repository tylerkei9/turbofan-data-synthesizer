# Source code

The core of the project, in three parts:

| Folder | In plain words |
|---|---|
| `training/` | The two AI models. `Transformer/` predicts an engine's next reading from its recent history; `Diffusion/` turns random noise into realistic readings and supports LoRA fine-tuning. |
| `features/` | Tools that edit artificial data: shift a sensor's values, fill gaps in a timeline, or let a model continue an engine's record. |
| `validate/` | Statistical tests that compare artificial data with real data. |
| `config/` | Settings files for the models (what data to use, how long to train, how big the model is). |

How these fit together: [docs/how-it-works.md](../docs/how-it-works.md). Commands to run them:
[docs/running-the-code.md](../docs/running-the-code.md).
