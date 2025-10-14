# Induced logical channel & hashing bound (Julia + Rust)

This repo computes the **induced logical channel** of a stabilizer code under an **i.i.d. Pauli channel** (e.g. depolarizing), using a **full stabilizer tableau** `(H, L_X, L_Z, G)` derived from a given set of stabilizers `S`. It then returns the **hashing bound** `(k - H( p̄ )) / n` for the induced channel.

Performance highlights:
- **Symplectic Gram–Schmidt (SGS)** in Julia to construct a full tableau adapted to `S`
- **Bit-packed** Pauli vectors (`UInt64` words) + hardware **popcount**
- Heavy inner loops in **Rust** (Rayon parallelism)
- **Parallel** sweeps over the depolarizing parameter grid

## Quick start

### 1) Build the Rust kernel

```bash
cd rust-kernel
cargo build --release

