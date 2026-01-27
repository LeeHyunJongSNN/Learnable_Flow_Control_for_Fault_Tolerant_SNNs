# Supplementary

This folder contains **(1) SNN training code based on PyTorch/SpikingJelly**, **(2) weight/connectivity fault injection**, **(3) input fragmentation methods (algorithmic / learnable / dynamic step selection)**, and **(4) (optional) an example of VHDL generation via Spiker+**.

- VHDL-related files are under `SpikerVHDL/`. They are **VHDL sources for running an SNN on FPGA** (plus minimal simulation scripts / I/O examples). (Per-file details are intentionally omitted.)
- This README focuses on **Python and shell scripts**.

---

## 0) Quick start

This repo uses **relative imports**, so it is recommended to run scripts from inside this folder:

```bash
cd Supplementary
```

Each script uses `argparse`, so you can check available options with `--help`:

```bash
python3 simple_snn.py --help
python3 vgg_snn.py --help
python3 resnet_snn.py --help
```

---

## 1) Environment / dependencies

### Required (training/experiments)
- Python 3.x
- `torch`, `torchvision`
- `spikingjelly` (SNN models / neurons)
- `numpy`
- `scikit-learn` (e.g., confusion matrix)
- `matplotlib` (some scripts may also use `seaborn` for plots)

### Optional (logging/plotting)
- `pandas` (used in `addon_metrics_wrapper.py`)

### Optional (VHDL generation)
- `snntorch` (dependency of Spiker+)
- `spikerplus` (install via pip or use a local checkout)

> **Note (headless servers without a GUI)**
> Some scripts force `matplotlib.use("TkAgg")`.
> On headless environments (SSH, no display), this can fail.
> - Either set `export MPLBACKEND=Agg` before running, or
> - change/comment out that line (e.g., switch to `Agg`).

---

## 2) Main scripts (training/evaluation)

### 2.1 MLP SNN (MNIST/FMNIST/UCIHAR)
- **File:** `simple_snn.py`
- **Data:**
  - Images: `--data_path propdata/MNIST` or `propdata/FMNIST`
  - Time-series (UCIHAR): `--dataset sequential --data_path <UCIHAR folder>`

Example:
```bash
# (Example) MNIST, train without faults
python3 simple_snn.py \
  --data_path propdata/MNIST \
  --Fault False \
  --num_epochs 10 \
  --batch_size 100
```

### 2.2 VGG SNN (CIFAR10/100)
- **File:** `vgg_snn.py`
- **Data:** `--data_path propdata/CIFAR10` or `propdata/CIFAR100`
- **Depth:** `--vgg_depth {7,11,15}`

Example:
```bash
python3 vgg_snn.py \
  --data_path propdata/CIFAR10 \
  --vgg_depth 11 \
  --num_epochs 50 \
  --train_batch_size 100 --test_batch_size 100
```

### 2.3 ResNet SNN (CIFAR10/100/Tiny-ImageNet)
- **File:** `resnet_snn.py`
- **Depth:** `--resnet_depth {18,34}`

Example:
```bash
python3 resnet_snn.py \
  --data_path propdata/CIFAR100 \
  --resnet_depth 34 \
  --num_epochs 50
```

---

## 3) Fault injection

The three SNN scripts (`simple_snn.py`, `vgg_snn.py`, `resnet_snn.py`) share the following options:

- `--Fault True/False` : enable/disable fault injection
- `--fault_type stuck|random|connectivity`
- `--fault_dist sporadic|clustered`
- `--fault_ratio 0.0~1.0`
- `--fault_start_epoch N` : apply faults starting from epoch N

Example:
```bash
# (Example) stuck-at faults, fault_ratio=0.1, applied from epoch 5
python3 simple_snn.py \
  --data_path propdata/MNIST \
  --Fault True \
  --fault_type stuck \
  --fault_dist sporadic \
  --fault_ratio 0.1 \
  --fault_start_epoch 5
```

Fault injection is handled by `FaultManager` in `fault_injection.py`.

---

## 4) Input fragmentation

This codebase supports three main flows:

1) **Algorithmic fragmentation (non-learnable)**: utilities such as `batch_dynamic_fragments(...)` in `algorithmic_fragmentation.py`
2) **Learnable fragmentation (fixed number of steps)**: `GlobalMultiLineFrags` in `learnable_fragmentation.py`
3) **Dynamic step selection (choose among candidates)**: `DynamicGlobalStaticMultiLineFrags` (and variants) in `learnable_fragmentation.py`

### Important default
In `simple_snn.py`, `vgg_snn.py`, and `resnet_snn.py`, the default of **`--Dynamic` is set to `True`**.
That means running a script with no extra flags may enter the “dynamic step selection fragmentation” path.

- If you want a fully plain SNN (**no fragmentation at all**), it is recommended to explicitly set:

```bash
python3 simple_snn.py --Dynamic False --Learnable False --Frag False
```

### 4.1 Algorithmic fragmentation
```bash
# (Example) enable algorithmic fragmentation
python3 simple_snn.py \
  --Frag True \
  --Dynamic False \
  --Learnable False
```

### 4.2 Learnable fragmentation (fixed `num_steps`)
```bash
python3 simple_snn.py \
  --Learnable True \
  --Dynamic False \
  --Frag False \
  --num_steps 2
```

### 4.3 Dynamic step selection fragmentation
```bash
python3 simple_snn.py \
  --Dynamic True \
  --Learnable False \
  --Frag False
```

---

## 5) (Optional) Robustness benchmark flags

You can enable the following flags in the three SNN scripts:

- `--ECOC True` : use ECOC head/loss (`benchmarks.py`)
- `--Soft True` : soft-SNN style processing
- `--Routing True` : fault-map-based routing
- `--Astrocyte True`, `--Falvolt True`, `--LIFA True` : additional correction/regularizer modules

Example:
```bash
python3 vgg_snn.py \
  --data_path propdata/CIFAR10 \
  --ECOC True
```

---

## 6) Automated fault-ratio sweep

### 6.1 Run via shell script (recommended)
- **File:** `run_sweep.sh`
- Internally calls `auto_sweep.py`.

```bash
chmod +x run_sweep.sh

# (Example) sweep fault_ratio 0.0~0.5 for VGG SNN
SWEEP_PY=./auto_sweep.py \
SCRIPT=./vgg_snn.py \
DATA_PATH=propdata/CIFAR10 \
DEVICES=0,1 \
MAX_PROCS=4 \
R_START=0.0 R_STOP=0.5 R_STEP=0.1 \
OPTIONS=baseline,ecoc,soft,routing,frag \
./run_sweep.sh
```

### 6.2 Run directly in Python
- **File:** `auto_sweep.py`
- Summary:
  - Iterates over fault ratios **sequentially** (to avoid memory blow-ups)
  - Runs **parallel processes** within each fault-ratio group
  - Writes results to CSV (option × fault_ratio pivot table)

Example:
```bash
python3 auto_sweep.py \
  --script ./simple_snn.py \
  --fault_type stuck \
  --fault_ratio_start 0.0 --fault_ratio_stop 0.5 --fault_ratio_step 0.1 \
  --options baseline,ecoc,soft,routing,frag \
  --devices 0,1 --max_procs 4 \
  --epochs 50 --batch_size 100 \
  --data_path propdata/MNIST \
  --results_dir results_auto_sweep
```

> Note: `auto_sweep.py` parses accuracy from stdout logs with a pattern like:
> `Total correctly classified ...: correct/total`

---

## 7) Extra metric logging/plotting (wrapper)

- **File:** `addon_metrics_wrapper.py`
- What it does:
  - Runs a target training script (e.g., `simple_snn.py`) as a wrapped subprocess
  - Monkey-patches the optimizer step to log gradient/weight statistics to CSV

Example:
```bash
python3 addon_metrics_wrapper.py \
  --metrics_csv mnist_fault.csv \
  --limit 1.0 --sat_eps 1e-3 \
  --print_every 10 --plot --fig_dir figs \
  --target simple_snn.py -- \
  --data_path propdata/MNIST --batch_size 100 --num_epochs 15 \
  --Fault True --fault_ratio 0.3
```

---

## 8) (Optional) Training → VHDL generation example

- **File:** `vhdl_simple_snn.py`
- Flow:
  1) Train a small SNN (MNIST)
  2) (Optional) apply fragmentation + fault injection
  3) Generate VHDL via Spiker+ (`spikerplus`)

Example:
```bash
python3 vhdl_simple_snn.py \
  --data_dir ./data \
  --out_dir ./out_run \
  --n_cycles 8 \
  --hidden 128 \
  --epochs 5 \
  --frag_method combo --overlap \
  --fault_ratio 0.05 --fault_type stuck --fault_start_epoch 2 \
  --weights_bw 8 --neurons_bw 12 --fp_dec 6 \
  --functional_vhdl False --interface False \
  --spiker_root ./Spiker-public/spiker
```

> Note: In this script, **fragmentation/encoding is performed as input preprocessing** (CPU/host-side),
> while the generated VHDL corresponds to the **SNN accelerator (Spiker+ network)**.

---

## 9) VHDL folder (brief)

- **Folder:** `SpikerVHDL/`
- Contents: VHDL sources for running an SNN on FPGA + testbench/I-O examples
  - e.g., `network.vhd`, `network_tb.vhd`
  - example I/O: `in_spikes.txt`, `out_spikes.txt`
  - `sim_script.tcl` (simple run/quit script)

Compilation/simulation steps depend on your environment (Questa/ModelSim/Vivado, etc.),
so this folder mainly provides the file set and examples.

---

## 10) File overview (Python/Shell-focused)

- `simple_snn.py` : MLP-SNN training/evaluation for MNIST/FMNIST/UCIHAR (fault, fragmentation, benchmark flags)
- `vgg_snn.py` : VGG-SNN training/evaluation for CIFAR10/100
- `resnet_snn.py` : ResNet-SNN training/evaluation for CIFAR10/100 (and optional datasets)
- `simple_dnn.py`, `vgg_dnn.py`, `resnet_dnn.py` : DNN baselines (can also serve as weak models)
- `algorithmic_fragmentation.py` : algorithmic fragmentation / saliency maps / temporal fragment generation utilities
- `learnable_fragmentation.py` : learnable fragmentation modules (fixed-step / dynamic step selection)
- `fault_injection.py` : fault-map generation/application (`FaultManager`)
- `benchmarks.py` : ECOC/soft/routing/correction methods and benchmark components
- `surrogate_encoders.py` : differentiable Poisson/Bernoulli spike encoders
- `utils.py` : helpers (tdBN, ZBiasAdder, CIFAR transforms, sequence→2D autofold, etc.)
- `auto_sweep.py` : automated fault-ratio sweep (parallel runs + CSV logging)
- `run_sweep.sh` : shell wrapper for sweeps
- `addon_metrics_wrapper.py` : wrapper for metric logging/plotting
- `vhdl_simple_snn.py` : (optional) example: train → generate VHDL via Spiker+
