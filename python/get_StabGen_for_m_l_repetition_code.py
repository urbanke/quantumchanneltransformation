#!/usr/bin/env python3
# -*- coding: utf-8 -*-

def stabilizers_full_repetition(m: int, ell: int):
    """
    Construct stabilizer generators for the CSS/Shor-style full repetition code:
      - Arrange n = m*ell qubits as ell blocks of size m: B1, B2, ..., Bell.
      - Z-type checks: within each block, Z_i Z_{i+1} for i=1..m-1.
      - X-type checks: between adjacent blocks, put X on all qubits of both blocks.

    Returns:
        List[str]: each string is length n over alphabet {I,X,Z}, one per generator.
    """
    if m < 2 or ell < 2:
        raise ValueError("Use m >= 2 and ell >= 2 for a nontrivial full repetition code.")
    n = m * ell
    gens = []

    def I_string():
        return ["I"] * n

    # Z-type intrablock pairs (protect X flips)
    for b in range(ell):
        block_start = b * m
        for i in range(m - 1):
            s = I_string()
            s[block_start + i] = "Z"
            s[block_start + i + 1] = "Z"
            gens.append("".join(s))

    # X-type interblock checks (protect Z flips)
    for b in range(ell - 1):
        s = I_string()
        # Block b
        for j in range(m):
            s[b * m + j] = "X"
        # Block b+1
        for j in range(m):
            s[(b + 1) * m + j] = "X"
        gens.append("".join(s))

    return gens


if __name__ == "__main__":
    # Shor [[9,1,3]]: m = 3 (within-block repetition), ell = 3 (across-block repetition)
    m, ell = 1, 1
    gens = stabilizers_full_repetition(m, ell)

    # Print literal Python list of strings (copy-paste ready)
    print('[' + ', '.join(f'"{s}"' for s in gens) + ']')

