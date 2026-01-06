# Upper bounds for qubit Pauli channels (`python/upper_bounds.py`)

This repository contains `upper_bounds.py` in the folder **python**, a script to **compute and plot (and save) multiple upper bounds** on the quantum capacity \(Q\) for several **single-qubit Pauli channel families**. The script generates PNG plots under `./plots/`.

The script supports both:
- **closed-form / non-SDP bounds** (fast), and
- **SDP-based bounds** (slower), enabled by computing an approximate-degradability parameter \(\varepsilon\) using **CVXPY**.

---

## What the script does

`upper_bounds.py` evaluates upper bounds on \(Q\) for three Pauli channel families:

1. **Independent (symmetric)**: \(p_X = p_Z = p\) (hence \(p_Y = p^2\))
2. **Depolarizing**: \(p_I = 1-p,\; p_X=p_Y=p_Z=p/3\)
3. **Skewed independent**: \(p_X=p,\; p_Z=p/9\) (hence \(p_Y=p^2/9\))

It saves three plots:
- `plots/upper_bounds_ind.png`
- `plots/upper_bounds_dep.png`
- `plots/upper_bounds_skw.png`

The `plots/` directory is created automatically by the script.

---

## Bounds implemented

### Fast (no SDP)
These run with `compute_eps = False`.

- **EA/2 bound** (Entanglement-assisted / 2):
  \[
  Q(\mathcal N) \le 1 - \frac{1}{2} H(p_I,p_X,p_Y,p_Z),
  \]
  where \(H(\cdot)\) is the Shannon entropy of the Pauli error distribution.

- **Entanglement-breaking (EB) cutoff**:
  for unital qubit channels (including Pauli channels), if
  \[
  |\lambda_X| + |\lambda_Y| + |\lambda_Z| \le 1,
  \]
  then the channel is entanglement-breaking and \(Q(\mathcal N)=0\).
  The script computes \(\lambda_X,\lambda_Y,\lambda_Z\) from \((p_I,p_X,p_Y,p_Z)\).

- **SSC convex-envelope bounds (closed-form)**
  - Implemented for **independent symmetric** (BB84-type expression)
  - Implemented for **depolarizing**

> Note: For the **skewed** independent channel, the script does not implement a special closed-form SSC curve; it will still compute EA/2 and EB (and tightest among computed).

### Optional SDP-based (slow)
These run with `compute_eps = True`.

- **Approximate degradability (AD) bound**:
  The script computes \(\varepsilon\) via an SDP (CVXPY) and plugs it into a published AD upper bound form.
  This is implemented for:
  - **independent symmetric** (BB84-type AD expression)
  - **depolarizing** (depolarizing AD expression)

> Performance note: the SDP is solved **once per p-grid point per channel**, so use a coarse p-grid.

---

## Requirements

### Python
- Python 3.10+ recommended (works on Python 3.12 as well)

### Python packages 
`python3 -m pip install numpy matplotlib cvxpy`

To verify:

`python3 -c "import numpy, matplotlib; print('ok')"`

`python3 -c "import cvxpy as cp; print('cvxpy:', cp.__version__)"`

### Solvers (important for SDPs)

CVXPY calls an external solver to solve the SDP. This script defaults to:

`eps_solver = "SCS"`

SCS = Splitting Conic Solver (a common first-order conic solver that supports SDPs).

Check which solvers CVXPY sees on your machine:

`python3 -c "import cvxpy as cp; print(cp.installed_solvers())"`

If SCS is missing, reinstall CVXPY or install SCS explicitly:
`python3 -m pip install scs`

### Running

From the directory containing upper_bounds.py:

`python3 upper_bounds.py`

Then view output images:

`open plots/upper_bounds_ind.png`

`open plots/upper_bounds_dep.png`

`open plots/upper_bounds_skw.png`

# qec-induced — Induced logical channel & hashing bound (Julia + Rust)

This project computes the **induced logical Pauli channel** of a stabilizer code under an **i.i.d. Pauli (depolarizing) physical channel**, and evaluates the **hashing bound**
\[
\mathrm{HB} \;=\; \frac{k - H(\bar p)}{n}.
\]
It:

- builds a **full stabilizer tableau** \((H, L_X, L_Z, G)\) from an independent stabilizer set \(S\) via **Symplectic Gram–Schmidt (SGS)** (Julia),
- uses **bit-packing** \((u|v)\) into `UInt64` words + **hardware popcount**,
- calls a **Rust** kernel (Rayon parallelism) for the hot inner loop (\(t\)-coset sums + ML argmax + accumulation),
- supports **parallel sweeps** over the depolarizing parameter \(p\).

---

## Repository Layout

```
qec-induced/
├─ README.md
├─ rust_kernel/
│  ├─ Cargo.toml
│  └─ src/
│     └─ lib.rs
└─ julia/
   ├─ Project.toml
   └─ src/
      ├─ QECInduced.jl
      ├─ Symplectic.jl
      ├─ SGS.jl
      ├─ Bitpack.jl
      ├─ Induced.jl
      └─ ParallelSweep.jl
```

> The Julia code looks for the Rust library via the environment variable **`QEC_RUST_LIB`**. You can keep any folder name; examples assume `rust_kernel/`.

---

## Ubuntu Installation

### Install Rust (rustup)
```bash
# Install rustup (interactive; choose default toolchain)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
# Activate in this shell
source ~/.cargo/env
# Verify
rustc --version
cargo --version
```

### Install Julia (juliaup)
```bash
# Install juliaup (official installer)
curl -fsSL https://install.julialang.org | sh
# Optionally re-source your shell as prompted
juliaup --version
# Install latest Julia and set as default
juliaup add release
juliaup default release
# Verify
julia --version
```

> Alternative: download binaries from https://julialang.org/downloads/ and add `julia` to your `PATH`.

---

## macOS Installation

### Install Rust (Homebrew + rustup)
```bash
# If you don't have Homebrew yet: https://brew.sh
brew install rustup-init
rustup-init -y
# Activate in this shell
source ~/.cargo/env
rustc --version
cargo --version
```

### Install Julia (Homebrew + juliaup)
```bash
brew install juliaup
juliaup add release
juliaup default release
julia --version
```

> Alternative: download the `.dmg` from https://julialang.org/downloads/ and drag to Applications.

---

## Build the Rust Kernel

From the repo root:
```bash
cd rust_kernel
cargo build --release
```

Shared library paths:
- **Linux:** `rust_kernel/target/release/librust_kernel.so`
- **macOS:** `rust_kernel/target/release/librust_kernel.dylib`
- **Windows:** `rust_kernel\target\release\rust_kernel.dll`

Tell Julia where it is:
```bash
# Linux
export QEC_RUST_LIB="$(pwd)/target/release/librust_kernel.so"

# macOS (zsh/bash)
export QEC_RUST_LIB="$(pwd)/target/release/librust_kernel.dylib"

# Windows (PowerShell)
# $env:QEC_RUST_LIB="C:\path\to\rust_kernel\target\release\rust_kernel.dll"
```

**Optional: Optimize Rust Release Profile** (add to `rust_kernel/Cargo.toml`)
```toml
[profile.release]
opt-level = 3
lto = "fat"
codegen-units = 1
strip = true
```

---

## Instantiate the Julia Environment

From the repo root:
```bash
cd julia
julia --project -e 'using Pkg; Pkg.precompile(); using QECInduced; println("OK")'
```

---

## Run the Demo (End-to-End)

```bash
# Ensure QEC_RUST_LIB is set (see “Build the Rust Kernel”)
julia --project -e 'using QECInduced; QECInduced.demo()'
```

Expected output:
- tiny demo tableau (n=1, k=1),
- `pbar` shape and sum,
- small grid `[p  hashing_bound]`.

---

## Running Your Own Computation

### Minimal Example

Create `myrun.jl` in `julia/`:
```julia
using QECInduced

# Example: n = 1, no stabilizers (k=1)
S = zeros(Bool, 0, 2)  # rows are (u|v), length 2n

# Build tableau via SGS
H, Lx, Lz, G = QECInduced.tableau_from_stabilizers(S)

# Single p
pbar, hb = QECInduced.induced_channel_and_hashing_bound(H, Lx, Lz, G; p=0.1)
println("pbar size = ", size(pbar), ", sum = ", sum(pbar))
println("Hashing bound = ", hb)

# Parallel sweep (set threads with JULIA_NUM_THREADS)
grid = QECInduced.sweep_depolarizing_grid(H, Lx, Lz, G; p_min=0.0, p_max=0.2, step=0.05, threads=4)
println("grid:\n", grid)
```

Run it:
```bash
export JULIA_NUM_THREADS=4   # choose threads
julia --project myrun.jl
```

### 3-Qubit Repetition Code Example

```julia
using QECInduced

# 3-qubit Z-type repetition code (bit-flip code)
# Pauli vector format is (u | v) with length 2n
#  - X on qubit i -> u_i = 1, v_i = 0
#  - Z on qubit i -> u_i = 0, v_i = 1

n = 3
S = falses(2, 2n)  # 2 stabilizers, 2n columns

# Stabilizers: Z1 Z2  and  Z2 Z3
# Row 1: (u=000 | v=110)
S[1, n+1] = true  # v1
S[1, n+2] = true  # v2

# Row 2: (u=000 | v=011)
S[2, n+2] = true  # v2
S[2, n+3] = true  # v3

# Ensure it's a plain Bool matrix
S = Matrix{Bool}(S)

# Build tableau/logicals
H, Lx, Lz, G = QECInduced.tableau_from_stabilizers(S)

@show size(H)  # (r, 2n)
@show size(Lx) # (k, 2n)
@show size(Lz) # (k, 2n)
@show size(G)  # (r, 2n)

# k should be 1 for the 3-qubit repetition code
@assert size(Lx, 1) == 1 && size(Lz, 1) == 1 "Expected k=1 logical qubit"

# Depolarizing channel with probability p
p = 0.10

# Call the public wrapper: it expects keyword `p::Float64`
pbar, hashing = QECInduced.induced_channel_and_hashing_bound(H, Lx, Lz, G; p=p)

@show size(pbar)
@show hashing
```

---

## Public API Summary

- **Build Tableau from Stabilizers**
  ```julia
  H, Lx, Lz, G = QECInduced.tableau_from_stabilizers(S::AbstractMatrix{Bool})
  ```
  `S` is \(r\times 2n\), rows are binary symplectic \((u|v)\).

- **Single-p Induced Channel & Hashing Bound**
  ```julia
  pbar::Matrix{Float64}, hb::Float64 =
      QECInduced.induced_channel_and_hashing_bound(H, Lx, Lz, G; p=0.1)
  ```

- **Parallel Sweep over p**
  ```julia
  grid::Matrix{Float64} =
      QECInduced.sweep_depolarizing_grid(H, Lx, Lz, G; p_min=0.0, p_max=0.2, step=0.005, threads=8)
  ```
  Returns a 2-column matrix `[p  hashing_bound]`.

---

## Performance Notes

- The hotspot is the coset sum over \(t\in\{0,1\}^r\).
- The Rust kernel:
  - uses **bit-packing** and **popcount** per 64 sites,
  - parallelizes over syndromes \(s\) with **Rayon**,
  - avoids storing the full \(P_s\) cube (computes argmax on the fly and accumulates).
- Outer-level parallelism: sweep many \(p\) values with threads (and/or launch multiple Julia processes).
- For very large \(r\), consider group-transform techniques (e.g., Walsh–Hadamard over the stabilizer subgroup) — not implemented here but compatible with this design.

---

## Troubleshooting

- **“Rust library not found”**  
  Set `QEC_RUST_LIB` to the absolute path of the built library (see “Build the Rust Kernel”). Example (Linux):
  ```bash
  export QEC_RUST_LIB="$(pwd)/rust_kernel/target/release/librust_kernel.so"
  ```
- **Mismatched architectures**  
  Ensure 64-bit Julia and Rust target match.
- **Missing build tools**  
  - Ubuntu: `sudo apt-get install build-essential` is helpful.  
  - macOS: install Xcode Command Line Tools: `xcode-select --install`.
- **Performance**  
  - Use `cargo build --release`.  
  - Increase `JULIA_NUM_THREADS`.  
  - Keep inputs as `Bool` and let the library bit-pack internally.

---

## License

MIT

---



