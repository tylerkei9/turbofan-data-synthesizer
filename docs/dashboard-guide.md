# Dashboard guide

Open `dashboard/index.html` in a web browser (Chrome, Edge, Safari or Firefox). It works offline and
needs no installation. It is designed for a laptop or desktop screen; on a phone it scrolls.

## The opening screen (the "theater")

An animation walks through the six stages, with a short description of each at the top. Along the
bottom:

- **Pause / Play** stops or restarts the animation.
- **The result line** summarizes the final test: 3 of 3 sensors pass, 91% similar to real data.
- **Skip to the result** jumps straight to the final test.
- **Run the full demo** plays every stage in order (about 90 seconds).
- **My role** (next to the title) shows the author's role on the project.

## Inside the demo

A bar at the top lists the stages: Import, Train, Generate, Edit, Fine-tune, Validate. Click any
stage you have reached to go back to it. **Theater** returns to the opening screen.

Each stage has two columns:

- **Left:** what goes in and comes out, the run's progress, the result card, and simple controls.
- **Right:** charts. Hover over a **(?)** next to a chart for a one-sentence explanation.

| Stage | What you see |
|---|---|
| Import | How many rows and engines were loaded, and real vs artificial readings for one engine. |
| Train | The loss dropping epoch by epoch, how noise is added during training, and the model's size. |
| Generate | The model turning random noise into realistic readings, step by step, and the final distribution compared with real data. |
| Edit | A tool to shift one sensor's values and preview the effect. Each change creates a new version. |
| Fine-tune | How small the LoRA add-on is compared with the model, and its loss over 30 epochs. |
| Validate | The verdict, a pass or fail per sensor, distribution and PCA charts, and the Transformer vs diffusion comparison. |

## What is real and what is replayed

- Train, Fine-tune and Generate **replay recorded real runs** of this code, sped up. Their settings are
  locked to the recorded configuration, so every replay is the real run.
- Validate and Edit **compute live in your browser** from the bundled data.
- Engine and Sensor buttons only change which data the charts display.
