# Data

## `FD001.csv`: real engine records

Simulated jet-engine records from NASA's **C-MAPSS** turbofan degradation data set, subset **FD001**
(one operating condition, one type of wear: high-pressure compressor degradation).

- 13,096 rows, 100 engines, 26 columns: engine number, cycle number, 3 operating settings
  (`Altitude`, `Mach_number`, `TRA`) and 21 sensors (columns `1` to `21`).
- This file is NASA's **test** portion of FD001 (`test_FD001.txt`), with column names added. In the
  test portion, each engine's record stops at some point **before** failure; the number of cycles left
  at that point is published separately by NASA (`RUL_FD001.txt`, not included here). Engine record
  lengths range from 31 to 303 cycles.

**Source:** NASA Prognostics Center of Excellence, Prognostics Data Repository (C-MAPSS
"Turbofan Engine Degradation Simulation" data set).
**Reference:** A. Saxena, K. Goebel, D. Simon and N. Eklund, "Damage Propagation Modeling for
Aircraft Engine Run-to-Failure Simulation," International Conference on Prognostics and Health
Management (PHM), 2008.

## `synthetic_demo.csv`: sample artificial data

An artificial dataset in the same column layout, kept as a ready-made example for the server's
"load bundled synthetic data" option. The dashboard's results use freshly generated model output
instead (see [docs/results.md](../docs/results.md)).

## Columns used in the dashboard

| Column | Sensor | Meaning |
|---|---|---|
| `11` | Ps30 | Static pressure at the high-pressure compressor outlet |
| `4` | T50 | Total temperature at the low-pressure turbine outlet |
| `7` | P30 | Total pressure at the high-pressure compressor outlet |
