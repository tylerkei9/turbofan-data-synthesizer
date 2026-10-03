# Results

## The headline

The best model is the **diffusion model, trained for 100 epochs and then fine-tuned with LoRA for
30 epochs**. Its 274 artificial readings pass the KS test on **3 of 3 sensors**, with an average
**similarity of 91%** to real data.

| Sensor | KS statistic | p-value | Result |
|---|---|---|---|
| Ps30 | 0.106 | 0.087 | Pass |
| T50 | 0.066 | 0.581 | Pass |
| P30 | 0.099 | 0.132 | Pass |

*How to read this:* the KS statistic is the biggest gap between the real and artificial values
(0 means identical). The p-value is the chance of seeing a gap that large if both sets truly came
from the same source. The standard rule is: **p of 0.05 or higher means "no meaningful difference"
(pass)**. Similarity is 100% minus the KS statistic, averaged over the three sensors.

## How the models compare

Every model was tested the same way, against the same real sample:

| Model | Training | Sensors passing | Similarity |
|---|---|---|---|
| Transformer | 8 epochs | 0 of 3 | 87% |
| Transformer | 50 epochs | 0 of 3 | 84% |
| Diffusion | 6 epochs | 0 of 3 | 68% |
| Diffusion | 100 epochs | 0 of 3 | 87% |
| **Diffusion + LoRA** | **100 + 30 epochs** | **3 of 3** | **91%** |

**Why the Transformer fails:** it always predicts its single best guess, so its output varies only
61–78% as much as real readings do. The values are in the right place but too smooth, and the KS
test detects that. Training longer did not fix it.

**Why the diffusion model wins:** it starts each reading from random noise, so its output varies the
way real data does. Longer training got it close (the 100-epoch model was slightly too spread out),
and the LoRA fine-tune closed the remaining gap.

## How the test was kept fair

- **Real, untouched output.** The 274 artificial readings are exactly what the model produced. Re-running
  the generation reproduces them exactly.
- **A fair real sample.** The real comparison set is 274 readings picked at random from all 100 engines,
  with the random choice fixed in advance (seed 42), before any results were seen. An earlier
  comparison used only 4 engines, but those 4 are not typical of the fleet: they fail the same test
  even against the full real dataset, so no model could have passed against them.
- **Settings fixed before results.** The training and fine-tuning settings were the project's own
  presets, chosen before each run's result was known. Checkpoints or random seeds were never picked
  because they happened to pass.

## Caveats

- **Sample size.** The test uses 274 readings on each side. Against all 13,096 real rows, Ps30 and T50
  still pass, while P30 falls just short (p ≈ 0.050), so the match is close but not perfect at full scale.
- **Independent readings, not timelines.** The diffusion model produces individual readings, not
  cycle-by-cycle engine histories, so it is not suited to questions about how one engine changes over
  time. The Transformer is the better fit for timelines.
- **The data's cut-off.** The FD001 file used here is NASA's test set, in which each engine's record
  stops at some point before failure. The distribution tests above are not affected, but "time until
  failure" cannot be read directly from these records (see [data/README.md](../data/README.md)).

## Where the numbers come from

All figures are produced by `api/record_replays.py` (steps `dsample` and `compare`) and are embedded
in the dashboard. The dashboard's Validate step recomputes the headline numbers in the browser and
shows the same values.
