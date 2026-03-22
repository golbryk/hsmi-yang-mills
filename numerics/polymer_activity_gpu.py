"""
Krok 0.2: Weryfikacja Lemma 3.1 — aktywności polimerów via Monte Carlo na GPU.

Lemma 3.1 (mass_gap_rigorous.tex):
    |a({p})| <= (g^4 / Delta^2) * |gamma|
gdzie Delta = Delta_min = 2, |gamma| = liczba plakiet.

Metoda:
- Pojedyncza plakieta gamma = {p}, link variables U_ell in SU(2)
- a({p}) = integral [ exp(-V_f(p, A^f)) - 1 ] dmu_{C_f}(A^f)
- W aproksymacji gaussowskiej z C_f = 1/Delta:
    V_f(p, A^f) = g^2 * |F_p(A^f)|^2
    F_p ~ suma komutatorów, F_p^2 ~ (g * A^f)^2 * 4 linki

RAW CUDA kernel: próbkowanie A^f z miary gaussowskiej N(0, 1/Delta),
obliczenie V_f, całkowanie na GPU.

Architektura: ZERO pętli CPU. Wszystko w jednym kernel launch.
- grid: N_g^2 wartości g^2 x N_size wartości |gamma|
- każdy wątek: N_samples próbek, redukuje lokalnie
"""

import os, sys
os.environ.setdefault('CUDA_PATH', '/usr/local/cuda-13.1')
os.environ['PATH'] = os.environ.get('PATH', '') + ':/usr/local/cuda-13.1/bin'

import numpy as np
import cupy as cp
import json
import time

# -----------------------------------------------------------------------
# RAW CUDA KERNEL — całe obliczenie MC w jednym kernel launch
# -----------------------------------------------------------------------

KERNEL_CODE = r"""
#include <curand_kernel.h>
#include <math.h>

// Losowanie z N(0, sigma^2) przy użyciu Box-Muller
__device__ float randn(curandState* state, float sigma) {
    float u1 = curand_uniform(state);
    float u2 = curand_uniform(state);
    // unikamy log(0)
    if (u1 < 1e-37f) u1 = 1e-37f;
    float z = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265f * u2);
    return sigma * z;
}

// Kernel: dla każdej pary (g2_idx, size_idx) oblicza |a(gamma)| przez MC
// Wynik: d_result[g2_idx * n_sizes + size_idx] = |a(gamma)|
extern "C" __global__ void polymer_activity_mc(
    const float*  d_g2,          // [n_g2] wartości g^2
    const int*    d_sizes,        // [n_sizes] rozmiary polimerów |gamma|
    int           n_g2,
    int           n_sizes,
    int           n_samples,      // próbki MC per wątek
    int           n_links_per_plaq, // = 4 (SU(2), d=4)
    float         delta_min,       // = 2.0
    float*        d_result,        // [n_g2 * n_sizes] wyniki |a|
    float*        d_bound,         // [n_g2 * n_sizes] granice (g^4/delta^2)*|gamma|
    unsigned long long seed
) {
    int g_idx   = blockIdx.x * blockDim.x + threadIdx.x;
    int sz_idx  = blockIdx.y * blockDim.y + threadIdx.y;
    if (g_idx >= n_g2 || sz_idx >= n_sizes) return;

    float g2     = d_g2[g_idx];
    int   gamma  = d_sizes[sz_idx];   // liczba plakiet
    float sigma  = rsqrtf(delta_min); // sigma = 1/sqrt(Delta) = odch. std. A^f

    // Inicjalizacja generatora losowego
    curandState state;
    curand_init(seed, (long long)g_idx * n_sizes + sz_idx, 0, &state);

    // Monte Carlo: estymacja E[exp(-V_f) - 1]
    // V_f(p, A^f) = g^2 * F_p^2
    // F_p = sum_{links} A^f_link * (coeff), aproksymacja: F_p^2 ~ sum_j (A_j)^2
    // Dla gamma plakiet: V_f = g^2 * sum_{p in gamma} F_p^2
    // Linki per plakieta = 4, su(2) dim = 3 komponenty
    // => 4 * 3 = 12 gaussowskich zmiennych na plakietę

    int n_comp_per_plaq = n_links_per_plaq * 3;  // 12 komponentów na plakietę
    double sum_exp  = 0.0;
    double sum_exp2 = 0.0;

    for (int s = 0; s < n_samples; ++s) {
        float Vf = 0.0f;
        // Dla każdej plakiety w polimerze
        for (int p = 0; p < gamma; ++p) {
            float Fp2 = 0.0f;
            // F_p^2 = (g^2 / 4) * sum_{j=1}^{4} |A_mu(x) - A_nu(x+mu)|^2
            // Uproszczenie: F_p^2 ~ g^2 * (A_1^2 + A_2^2 + A_3^2 + A_4^2) * c
            // gdzie A_j sa niezaleznymi N(0, 1/Delta)
            // Normalizacja: c = 1 (na link, sumujemy 4 linki)
            for (int j = 0; j < n_comp_per_plaq; ++j) {
                float Aj = randn(&state, sigma);
                Fp2 += Aj * Aj;
            }
            // Wklad jednej plakiety do V_f
            // V_f^{(p)} = g^4/(12*Cf^2) * Fp^4 dla kwartyki,
            // ale testujemy dolny rzad: g^2 * Fp^2
            Vf += g2 * Fp2;
        }
        float contrib = expf(-Vf) - 1.0f;
        sum_exp  += (double)contrib;
        sum_exp2 += (double)(contrib * contrib);
    }

    double mean  = sum_exp  / n_samples;
    double mean2 = sum_exp2 / n_samples;
    double var   = mean2 - mean * mean;
    double stderr_val = (n_samples > 1) ? sqrtf((float)(var / (n_samples - 1))) : 0.0f;

    int idx = g_idx * n_sizes + sz_idx;
    d_result[idx] = (float)fabsf((float)mean);       // |a(gamma)| numerycznie
    d_bound[idx]  = (g2 * g2 / (delta_min * delta_min)) * gamma;  // (g^4/Delta^2)*|gamma|
    // Zapisz rowniez stderr w kolejnym elemencie (osobna tablica — ale tu prostej)
    // => pomijamy dla clean interface
}
"""

# -----------------------------------------------------------------------
# Kompilacja i uruchomienie
# -----------------------------------------------------------------------

def compile_kernel():
    """Kompiluj raw CUDA kernel (nvcc, curand)."""
    module = cp.RawModule(code=KERNEL_CODE,
                          backend='nvcc',
                          options=('-O3', '--use_fast_math',
                                   '-arch=sm_120',   # RTX 5070 = sm_120
                                   '-lcurand'))
    return module.get_function('polymer_activity_mc')


def run_polymer_mc(g2_values, size_values, n_samples=1_000_000, seed=12345):
    """
    Uruchom Monte Carlo weryfikację aktywności polimerów na GPU.

    Parametry
    ----------
    g2_values  : list[float] — wartości g² do sprawdzenia
    size_values: list[int]   — rozmiary polimerów |gamma|
    n_samples  : int         — próbek MC per (g², |γ|)
    seed       : int         — ziarno curand

    Zwraca
    ------
    result : np.ndarray [n_g2, n_sizes] — |a(gamma)| numerycznie
    bound  : np.ndarray [n_g2, n_sizes] — (g^4/Delta^2)*|gamma|
    ratio  : np.ndarray [n_g2, n_sizes] — result/bound (powinno być <= 1)
    """
    kernel = compile_kernel()

    n_g2   = len(g2_values)
    n_sz   = len(size_values)
    delta  = 2.0
    n_links = 4

    # Transfer danych na GPU
    d_g2    = cp.array(g2_values, dtype=cp.float32)
    d_sizes = cp.array(size_values, dtype=cp.int32)
    d_res   = cp.zeros(n_g2 * n_sz, dtype=cp.float32)
    d_bnd   = cp.zeros(n_g2 * n_sz, dtype=cp.float32)

    # Grid / block: jeden wątek = jedna para (g², |γ|)
    block = (16, 16, 1)
    grid  = ((n_g2 + 15) // 16, (n_sz + 15) // 16, 1)

    t0 = time.time()
    kernel(grid, block, (
        d_g2, d_sizes,
        np.int32(n_g2), np.int32(n_sz),
        np.int32(n_samples),
        np.int32(n_links),
        np.float32(delta),
        d_res, d_bnd,
        np.uint64(seed)
    ))
    cp.cuda.Device().synchronize()
    elapsed = time.time() - t0

    result = d_res.get().reshape(n_g2, n_sz)
    bound  = d_bnd.get().reshape(n_g2, n_sz)
    ratio  = result / np.maximum(bound, 1e-30)

    print(f"GPU time: {elapsed:.2f}s | "
          f"{n_g2 * n_sz * n_samples / elapsed / 1e9:.2f} Gsamples/s")
    return result, bound, ratio


def main():
    print("=" * 60)
    print("Krok 0.2: Weryfikacja Lemma 3.1 — aktywności polimerów GPU")
    print("=" * 60)

    # Parametry testu
    g2_values   = np.linspace(0.05, 0.55, 20).tolist()
    size_values = [1, 2, 3, 5, 8, 12, 20]
    n_samples   = 2_000_000  # 2M próbek per punkt

    print(f"g² ∈ [{g2_values[0]:.2f}, {g2_values[-1]:.2f}], "
          f"{len(g2_values)} wartości")
    print(f"|γ| ∈ {size_values}")
    print(f"Próbki MC: {n_samples:,} per punkt")
    print(f"Łącznie: {len(g2_values) * len(size_values) * n_samples / 1e9:.2f}G próbek")
    print()

    result, bound, ratio = run_polymer_mc(g2_values, size_values, n_samples)

    # Analiza: czy lemat zachodzi?
    max_ratio = ratio.max()
    violated  = (ratio > 1.0).sum()
    marginal  = (ratio > 0.9).sum()

    print()
    print("Wyniki:")
    print(f"  max(|a(γ)| / bound) = {max_ratio:.4f}  (powinno być ≤ 1)")
    print(f"  Naruszenia (ratio > 1): {violated}")
    print(f"  Bliskie naruszenia (ratio > 0.9): {marginal}")
    print()

    # Tabela dla kluczowych wartości g²
    print("Tabela: max ratio po |γ| dla każdego g²:")
    print(f"{'g²':>8} | {'|γ|=1':>8} {'|γ|=5':>8} {'|γ|=20':>8} | {'max':>8}")
    print("-" * 52)
    for i, g2 in enumerate(g2_values):
        row = ratio[i, :]
        cols = [row[0], row[3], row[6]] if len(row) >= 7 else row[:3].tolist()
        print(f"{g2:8.4f} | {cols[0]:8.4f} {cols[1]:8.4f} {cols[2]:8.4f} | "
              f"{row.max():8.4f}")

    # Zapis wyników
    outfile = os.path.join(os.path.dirname(__file__),
                           '..', 'data', 'polymer_activity_result.json')
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    out = {
        'g2_values':   g2_values,
        'size_values': size_values,
        'result':      result.tolist(),
        'bound':       bound.tolist(),
        'ratio':       ratio.tolist(),
        'max_ratio':   float(max_ratio),
        'violated':    int(violated),
        'lemma_holds': bool(max_ratio <= 1.0),
        'n_samples':   n_samples,
    }
    with open(outfile, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nWyniki zapisane: {outfile}")

    verdict = "✓ LEMAT 3.1 ZWERYFIKOWANY" if max_ratio <= 1.0 else \
              f"✗ NARUSZENIE (max ratio = {max_ratio:.4f})"
    print(f"\n{verdict}")
    return 0 if max_ratio <= 1.0 else 1


if __name__ == '__main__':
    sys.exit(main())
