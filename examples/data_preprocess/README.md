# Usage

The processed data will be saved in the `data/train` and `data/test` directories.
The naming convention is `<domain>__<dataset_name>_<dataset_size>`, where `<domain>` is one of `math`, `codegen`, `logic`, `simulation`, `tableqa` for now.

## Math
```bash
python examples/data_preprocess/math/bigmath_preview_filtered_mar21.py --train-sample-size 10000
```

## Code

## Logic

## Simulation
```bash
python examples/data_preprocess/simulation/codeio.py --train-sample-size 5000 --test-sample-size 500
```

## Table
```bash
python examples/data_preprocess/table/multihier.py
```