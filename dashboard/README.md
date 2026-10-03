# Dashboard

Open **`index.html`** in a web browser to see the project. It is one self-contained file: it works
offline, without a server or installation.

- **Run the full demo** plays the whole process (about 90 seconds); **Skip to the result** jumps to the
  final test.
- What each screen shows: [docs/dashboard-guide.md](../docs/dashboard-guide.md).

The other files here are the data the page displays (`demo_data.js`, `replays.json`,
`ckpt_meta.json`), already embedded inside `index.html`. After regenerating them with
`api/record_replays.py`, run `python dashboard/embed_data.py` to update the page
(see [docs/running-the-code.md](../docs/running-the-code.md)).

To show a GitHub link in the dashboard, search `index.html` for `GITHUB_URL = '';` and put the
repository address between the quotes.
