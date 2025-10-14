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

Here \(n=3, r=2\). Use \((u|v)\) of length \(2n=6\).  
\(Z_1Z_2 = (000|110)\), \(Z_2Z_3 = (000|011)\).

```julia
using QECInduced

n = 3
S = falses(2, 2n)
S[1, n+1] = true; S[1, n+2] = true   # Z1Z2
S[2, n+2] = true; S[2, n+3] = true   # Z2Z3

H, Lx, Lz, G = QECInduced.tableau_from_stabilizers(S)

pbar, hb = QECInduced.induced_channel_and_hashing_bound(H, Lx, Lz, G; p=0.1)
println("HB = ", hb)

grid = QECInduced.sweep_depolarizing_grid(H, Lx, Lz, G; p_min=0.0, p_max=0.3, step=0.05, threads=4)
println(grid)
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



