# How it works

This guide explains the whole process without assuming any technical background. Words in
*italics* are defined in the [glossary](glossary.md).

## The problem

A jet engine (a *turbofan*) carries dozens of *sensors* that record temperature, pressure and speed
every flight *cycle*. Engineers use these records to predict when a part will need maintenance.
Teaching software to make those predictions needs a lot of records, but real ones are limited and
often private. The goal here is to create **artificial records that behave like real ones**, and to
check that they really do.

## The data

The real data comes from NASA: simulated records of 100 jet engines (the *FD001* set). Each row is
one engine at one cycle, with three operating settings and 21 sensor readings. The dashboard and the
tests focus on three sensors:

| Short name | What it measures |
|---|---|
| Ps30 | Static pressure at the high-pressure compressor outlet |
| T50 | Temperature at the low-pressure turbine outlet |
| P30 | Total pressure at the high-pressure compressor outlet |

Details and the exact source are in [data/README.md](../data/README.md).

## The two models

The project tries two different kinds of AI model and compares them.

**1. The Transformer: "predict the next reading."**
It reads an engine's recent history (the last 31 cycles) and predicts the next cycle. To create a
whole artificial engine, it starts from a few real cycles and keeps predicting, feeding each new
prediction back in. This is good at smooth, realistic-looking timelines. Its weakness: it always
gives its single best guess, so its output is *too smooth*. Real sensors are noisier, and the tests
notice.

**2. The Diffusion model: "remove the noise."**
During training, real readings are gradually buried in random static, and the model learns to tell
the static apart from the signal. To create new data, it starts from pure static and removes a little
of it, step by step (1,000 steps), until realistic readings remain. Because it starts from random
static each time, its output naturally varies like real data does. It produces individual readings
rather than whole engine timelines.

## The six steps

**1. Import.** Load the real records, match up the columns, and check for missing values or sensors
that never change.

**2. Train.** Show the model the real records over and over (each full pass is an *epoch*). The
diffusion model used here trained for 100 epochs, about 30 minutes on a laptop. The trained model is
saved as a *checkpoint* file.

**3. Fine-tune.** Instead of retraining the whole model, attach a small add-on (*LoRA*) and train only
that. Here it changes just 1.44% of the model's numbers, yet it noticeably improves how well the
output matches the real data.

**4. Generate.** Ask the model for new, artificial readings. The demo uses 274 of them.

**5. Edit (optional).** Deliberately change the artificial data to create test cases, for example
raise one sensor's values. Every edit creates a new version; the original is kept.

**6. Validate.** Compare artificial and real readings with standard statistical tests:

- the *KS test*, which checks whether two sets of values could come from the same source;
- *correlation*, which checks whether sensors move together the same way;
- *PCA*, a picture that places both datasets on one map to see whether they overlap.

A sensor **passes** when the KS test finds no meaningful difference. How the final result was
reached, and how fair the comparison is, are covered in [results.md](results.md).

## How the demo works

The dashboard does not train models live in your browser (that would take about 40 minutes). Instead,
the real training runs were **recorded** when they happened, and the demo **replays** them, sped up:
every loss value, progress step and log line is the real one. The final tests (step 6) are computed
in your browser from the bundled data. The [dashboard guide](dashboard-guide.md) shows what each
screen means.
