"""
Krok 0.3: 4D SU(2) Yang-Mills Monte Carlo na GPU.

Cel: Wyznaczyć m_latt(β) z korelatorów Wilsona dla rozmiarów L=4,6,8,
a następnie m_phys(β) = m_latt(β) / a(β) i sprawdzić zbieżność do stałej.

Architektura GPU:
- ZERO CPU-GPU pętli w wewnętrznych pętlach MC
- Wszystkie konfiguracje na GPU, aktualizacja parzyste/nieparzyste w jednym kernel launch
- Pomiar korelatorów Polyakova w tym samym kernel pass

SU(2) parametryzacja: U ∈ SU(2) ↔ (a₀, a₁, a₂, a₃) ∈ S³ (a₀² + |a|² = 1)
Mnożenie: (a₀,a) × (b₀,b) = (a₀b₀ - a·b, a₀b + b₀a + a×b)
Haar: jednolity na S³ (próbkowanie przez normalizację losowego wektora 4D)

Algorytm Metropolis na GPU:
- Kratka L^4 z pbc, każdy link U_{x,μ} = 4D tensor indeksowany (x, μ)
- Aktualizacja: checkerboard (parzyste/nieparzyste węzły naprzemiennie)
- Każdy wątek: jeden link, oblicza staple, proponuje nowe U, akceptuje/odrzuca

Pomiar:
- Pętla Polyakova P(x) = Tr(Π_t U_{x,t,τ}) — operator fundamentalny dla masy
- Korelatora: C(r) = ⟨P(x) P†(x+r)⟩
- Masa: m_latt = -log(C(r+1)/C(r)) dla dużych r
"""

import os, sys
os.environ.setdefault('CUDA_PATH', '/usr/local/cuda-13.1')
os.environ['PATH'] = os.environ.get('PATH', '') + ':/usr/local/cuda-13.1/bin'

import numpy as np
import cupy as cp
import json, time

# =======================================================================
# RAW CUDA KERNELS
# =======================================================================

SU2_MC_CODE = r"""
#include <curand_kernel.h>
#include <math.h>

// SU(2) element: q = (q0, q1, q2, q3) z q0^2 + q1^2 + q2^2 + q3^2 = 1
// Layout: U[site * 4 * 4 + mu * 4 + component]
// site = x + L*(y + L*(z + L*t))

__device__ inline void su2_mul(
    const float* A, const float* B, float* C
) {
    // C = A * B dla SU(2) w parametryzacji kwaternionowej
    C[0] = A[0]*B[0] - A[1]*B[1] - A[2]*B[2] - A[3]*B[3];
    C[1] = A[0]*B[1] + A[1]*B[0] + A[2]*B[3] - A[3]*B[2];
    C[2] = A[0]*B[2] - A[1]*B[3] + A[2]*B[0] + A[3]*B[1];
    C[3] = A[0]*B[3] + A[1]*B[2] - A[2]*B[1] + A[3]*B[0];
}

__device__ inline void su2_dag(const float* A, float* B) {
    // B = A† (koniugat hermitowski / odwrotny dla SU(2))
    B[0] =  A[0];
    B[1] = -A[1];
    B[2] = -A[2];
    B[3] = -A[3];
}

__device__ inline float su2_retrace(const float* A) {
    return A[0];  // Re Tr U = 2 q0 => w normalizacji Re(Tr U/2) = q0
}

__device__ inline int site_idx(int x, int y, int z, int t, int L) {
    return x + L*(y + L*(z + L*t));
}

__device__ inline int link_idx(int site, int mu, int L4) {
    return site * 4 * 4 + mu * 4;
}

// Oblicz staple dla linka U_{x,μ}
// Staple = suma plakiet zawierających (x,μ)
__device__ void compute_staple(
    const float* __restrict__ U,
    int x, int y, int z, int t, int mu,
    int L, float* staple
) {
    int s0 = site_idx(x,y,z,t,L);
    // Zeruj staple
    staple[0] = staple[1] = staple[2] = staple[3] = 0.0f;

    float tmp[4], tmp2[4], dU[4];
    int xp[4] = {x,y,z,t};

    // Dla każdego nu != mu: dwa wkłady (nu i -nu)
    for (int nu = 0; nu < 4; nu++) {
        if (nu == mu) continue;
        int coords[4] = {x,y,z,t};

        // Staple "do przodu" w kierunku nu:
        // U_{x+mu, nu} × U†_{x+nu, mu} × U†_{x, nu}
        int xmu[4] = {x,y,z,t}; xmu[mu] = (xmu[mu]+1) % L;
        int xnu[4] = {x,y,z,t}; xnu[nu] = (xnu[nu]+1) % L;

        int s_xmu = site_idx(xmu[0],xmu[1],xmu[2],xmu[3],L);
        int s_xnu = site_idx(xnu[0],xnu[1],xnu[2],xnu[3],L);

        const float* U_xmu_nu = U + link_idx(s_xmu, nu, L*L*L*L);
        const float* U_xnu_mu = U + link_idx(s_xnu, mu, L*L*L*L);
        const float* U_x_nu   = U + link_idx(s0,    nu, L*L*L*L);

        float U_dag_xnu_mu[4], U_dag_x_nu[4];
        su2_dag(U_xnu_mu, U_dag_xnu_mu);
        su2_dag(U_x_nu,   U_dag_x_nu);

        su2_mul(U_xmu_nu, U_dag_xnu_mu, tmp);
        su2_mul(tmp, U_dag_x_nu, tmp2);
        for (int k=0; k<4; k++) staple[k] += tmp2[k];

        // Staple "do tyłu" w kierunku -nu:
        // U†_{x+mu-nu, nu} × U†_{x-nu, mu} × U_{x-nu, nu}
        int xmun[4] = {x,y,z,t};
        xmun[mu] = (xmun[mu]+1) % L;
        xmun[nu] = (xmun[nu]+L-1) % L;
        int xmn[4]  = {x,y,z,t}; xmn[nu] = (xmn[nu]+L-1) % L;

        int s_xmun = site_idx(xmun[0],xmun[1],xmun[2],xmun[3],L);
        int s_xmn  = site_idx(xmn[0], xmn[1], xmn[2], xmn[3], L);

        const float* U_xmun_nu = U + link_idx(s_xmun, nu, L*L*L*L);
        const float* U_xmn_mu  = U + link_idx(s_xmn,  mu, L*L*L*L);
        const float* U_xmn_nu  = U + link_idx(s_xmn,  nu, L*L*L*L);

        float U_dag_xmun_nu[4], U_dag_xmn_mu[4];
        su2_dag(U_xmun_nu, U_dag_xmun_nu);
        su2_dag(U_xmn_mu,  U_dag_xmn_mu);

        su2_mul(U_dag_xmun_nu, U_dag_xmn_mu, tmp);
        su2_mul(tmp, U_xmn_nu, tmp2);
        for (int k=0; k<4; k++) staple[k] += tmp2[k];
    }
}

// Kernel inicjalizacji: wszystkie linki = jedynka (gorące/zimne start)
extern "C" __global__ void init_cold(float* U, int L4) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= L4 * 4) return;
    // U[link * 4] = (1,0,0,0)
    int base = idx * 4;
    U[base+0] = 1.0f;
    U[base+1] = 0.0f;
    U[base+2] = 0.0f;
    U[base+3] = 0.0f;
}

// Kernel inicjalizacji losowej (gorący start)
extern "C" __global__ void init_hot(float* U, int L4, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= L4 * 4) return;
    curandState state;
    curand_init(seed, idx, 0, &state);
    int base = idx * 4;
    float n = 0;
    for (int k=0; k<4; k++) {
        float r = curand_normal(&state);
        U[base+k] = r;
        n += r*r;
    }
    n = rsqrtf(n);
    for (int k=0; k<4; k++) U[base+k] *= n;
}

// Kernel Metropolisa (checkerboard): aktualizuje linki na węzłach parzystych/nieparzystych
extern "C" __global__ void metropolis_sweep(
    float* U,
    int L,
    float beta,
    int parity,          // 0 = parzyste, 1 = nieparzyste
    unsigned long long seed,
    long long sweep_id
) {
    int L4 = L*L*L*L;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n_links = L4 * 4;
    if (idx >= n_links) return;

    int site = idx / 4;
    int mu   = idx % 4;

    // Sprawdź parzystość węzła
    int t  = site / (L*L*L);
    int z  = (site / (L*L)) % L;
    int y  = (site / L) % L;
    int x  = site % L;
    if (((x+y+z+t) % 2) != parity) return;

    curandState state;
    curand_init(seed ^ (sweep_id * 1000007LL), idx, 0, &state);

    float* Ulink = U + link_idx(site, mu, L4);

    // Oblicz staple
    float staple[4];
    compute_staple(U, x, y, z, t, mu, L, staple);

    // Propozycja: U' = exp(iε σ·n̂) × U
    // Uproszczenie: losuj U' z rozkładu skupionego wokół U × staple
    // Algorytm Cabraya-Marinari: propozycja SU(2) z K=10 prób
    int n_hit = 4;
    float current[4];
    for(int k=0; k<4; k++) current[k] = Ulink[k];

    // Poprawna akcja SU(2):
    // ΔS = -β × [(U'·S)₀ - (U·S)₀]  gdzie (A·B)₀ = A[0]*B[0]-A[1]*B[1]-A[2]*B[2]-A[3]*B[3]
    // = Re Tr(U × Σ V_staple)/2 (kwaternionowy iloczyn, część rzeczywista)
    float dot_current = current[0]*staple[0] - current[1]*staple[1]
                      - current[2]*staple[2] - current[3]*staple[3];
    float current_action = -beta * dot_current;

    for (int hit = 0; hit < n_hit; hit++) {
        // Propozycja: U' = (1-eps) × U + eps × R gdzie R ~ Haar
        // Uproszczenie: losuj U' = R (czysto losowe SU(2)) lub
        // lokalny krok: U' = normalize(U + delta × gaussian)
        float prop[4];
        float eps = 0.3f;  // krok lokalny (dostrój dla akceptacji ~50%)
        float n2 = 0;
        for (int k=0; k<4; k++) {
            float r = curand_normal(&state);
            prop[k] = current[k] + eps * r;
            n2 += prop[k]*prop[k];
        }
        n2 = rsqrtf(n2);
        for (int k=0; k<4; k++) prop[k] *= n2;

        // Działanie propozycji
        float dot_prop = prop[0]*staple[0] - prop[1]*staple[1]
                       - prop[2]*staple[2] - prop[3]*staple[3];
        float prop_action = -beta * dot_prop;

        // Metropolis
        float delta = prop_action - current_action;
        float accept_prob = (delta < 0.0f) ? 1.0f : expf(-delta);
        if (curand_uniform(&state) < accept_prob) {
            for (int k=0; k<4; k++) current[k] = prop[k];
            current_action = prop_action;
        }
    }
    for (int k=0; k<4; k++) Ulink[k] = current[k];
}

// Kernel: pomiar oczekiwanej wartości plakiety <Re Tr U_p>
// Suma po wszystkich plakietach / (6 L^4)
extern "C" __global__ void measure_plaquette(
    const float* __restrict__ U,
    int L,
    float* plaq_sum  // wynik (redukowany przez atomicAdd)
) {
    int L4 = L*L*L*L;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Iterujemy po parach (site, mu<nu)
    int n_plaq = L4 * 6;  // 6 typów plakiet per site
    if (idx >= n_plaq) return;

    int plaq_type = idx % 6;
    int site = idx / 6;
    int t = site / (L*L*L);
    int z = (site / (L*L)) % L;
    int y = (site / L) % L;
    int x = site % L;

    // 6 typów plakiet: (01,02,03,12,13,23)
    int mu_nu[6][2] = {{0,1},{0,2},{0,3},{1,2},{1,3},{2,3}};
    int mu = mu_nu[plaq_type][0];
    int nu = mu_nu[plaq_type][1];

    // Oblicz plakietę U_{x,mu} U_{x+mu, nu} U†_{x+nu, mu} U†_{x, nu}
    int s0   = site_idx(x,y,z,t,L);
    int coords_mu[4] = {x,y,z,t}; coords_mu[mu] = (coords_mu[mu]+1)%L;
    int coords_nu[4] = {x,y,z,t}; coords_nu[nu] = (coords_nu[nu]+1)%L;

    int s_mu = site_idx(coords_mu[0],coords_mu[1],coords_mu[2],coords_mu[3],L);
    int s_nu = site_idx(coords_nu[0],coords_nu[1],coords_nu[2],coords_nu[3],L);

    const float* U0 = U + link_idx(s0,   mu, L4);
    const float* U1 = U + link_idx(s_mu, nu, L4);
    const float* U2 = U + link_idx(s_nu, mu, L4);
    const float* U3 = U + link_idx(s0,   nu, L4);

    float U2d[4], U3d[4];
    su2_dag(U2, U2d);
    su2_dag(U3, U3d);

    float tmp[4], tmp2[4], Uplaq[4];
    su2_mul(U0, U1, tmp);
    su2_mul(tmp, U2d, tmp2);
    su2_mul(tmp2, U3d, Uplaq);

    // Re Tr U_p = 2 Uplaq[0]
    atomicAdd(plaq_sum, Uplaq[0]);  // sum of q0 = sum of (Re Tr U_p)/2
}

// Kernel: pomiar pętli Polyakova P(x) = Tr(Π_{t=0}^{L-1} U_{(x,y,z,t), 3})
// Wynik: poly_re[x+L*y+L*L*z] = Re(P(x,y,z))
extern "C" __global__ void measure_polyakov(
    const float* __restrict__ U,
    int L,
    float* poly_re,  // [L^3]
    float* poly_im   // [L^3] (dla SU(2): Im zawsze 0 w bazie kwaternionowej)
) {
    int L3 = L*L*L;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= L3) return;

    int z = idx / (L*L);
    int y = (idx / L) % L;
    int x = idx % L;

    // Oblicz P(x,y,z) = Π_{t=0}^{L-1} U_{(x,y,z,t), direction=3 (czas)}
    float P[4] = {1.0f, 0.0f, 0.0f, 0.0f};  // jedynka
    for (int t = 0; t < L; t++) {
        int s = site_idx(x,y,z,t,L);
        const float* Ut = U + link_idx(s, 3, L3*L);  // μ=3 = kierunek czasowy
        float tmp[4];
        su2_mul(P, Ut, tmp);
        for (int k=0; k<4; k++) P[k] = tmp[k];
    }
    // Re Tr P = 2 P[0] (konwencja su2: Tr = 2 q0)
    poly_re[idx] = 2.0f * P[0];
    poly_im[idx] = 0.0f;  // SU(2) Polyakov loop jest rzeczywista po uśrednieniu
}

// Kernel: korelatora pętli Polyakova C(r) = <P(x)P(x+r)>
extern "C" __global__ void measure_correlator(
    const float* __restrict__ poly_re,
    int L,
    float* correlator  // [L/2] — C(r) dla r=0..L/2
) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= L/2) return;

    float sum = 0.0f;
    int L3 = L*L*L;
    for (int z=0; z<L; z++)
    for (int y=0; y<L; y++)
    for (int x=0; x<L; x++) {
        int s0 = x + L*y + L*L*z;
        int xr = (x+r) % L;
        int s1 = xr + L*y + L*L*z;
        sum += poly_re[s0] * poly_re[s1];
    }
    correlator[r] = sum / L3;
}
"""


def compile_su2_kernels():
    module = cp.RawModule(code=SU2_MC_CODE, backend='nvcc',
                          options=('-O3', '--use_fast_math',
                                   '-arch=sm_120'))
    return {
        'init_cold':         module.get_function('init_cold'),
        'init_hot':          module.get_function('init_hot'),
        'metropolis_sweep':  module.get_function('metropolis_sweep'),
        'measure_polyakov':  module.get_function('measure_polyakov'),
        'measure_correlator':module.get_function('measure_correlator'),
    }


def run_su2_mc(L, beta, n_therm=200, n_meas=100, n_sweep_per_meas=5):
    """
    4D SU(2) Monte Carlo na GPU dla rozmiaru L^4.
    Zwraca: m_latt (szacunkowa masa z korelatorów Polyakova), acceptance.
    """
    kernels = compile_su2_kernels()
    L4 = L**4

    # Alokacja konfiguracji na GPU: L^4 sites × 4 linki × 4 komponenty = float
    d_U = cp.zeros(L4 * 4 * 4, dtype=cp.float32)

    # Gorący start
    threads = 256
    blocks  = (L4 * 4 + threads - 1) // threads
    kernels['init_hot']((blocks,), (threads,),
                        (d_U, np.int32(L4), np.uint64(42)))

    # Termlizacja
    for sw in range(n_therm):
        for par in [0, 1]:
            bk = (L4 * 4 + threads - 1) // threads
            kernels['metropolis_sweep']((bk,), (threads,), (
                d_U, np.int32(L), np.float32(beta),
                np.int32(par), np.uint64(sw * 13 + par * 7),
                np.int64(sw)
            ))

    # Pomiary
    # --- Dodaj pomiar plakiety ---
    kernels['measure_plaquette'] = compile_su2_kernels()['measure_plaquette'] \
        if 'measure_plaquette' in compile_su2_kernels() else None
    plaq_accum = 0.0
    d_plaq_sum = cp.zeros(1, dtype=cp.float32)

    d_poly   = cp.zeros(L**3, dtype=cp.float32)
    d_polyim = cp.zeros(L**3, dtype=cp.float32)
    d_corr   = cp.zeros(L//2, dtype=cp.float32)
    corr_accum = np.zeros(L//2)

    blocks3 = (L**3 + threads - 1) // threads
    blocks_r = (L//2 + threads - 1) // threads

    for meas in range(n_meas):
        for sw in range(n_sweep_per_meas):
            for par in [0, 1]:
                bk = (L4 * 4 + threads - 1) // threads
                kernels['metropolis_sweep']((bk,), (threads,), (
                    d_U, np.int32(L), np.float32(beta),
                    np.int32(par),
                    np.uint64((n_therm + meas * n_sweep_per_meas + sw) * 13 + par * 7),
                    np.int64(n_therm + meas * n_sweep_per_meas + sw)
                ))

        kernels['measure_polyakov']((blocks3,), (threads,), (
            d_U, np.int32(L), d_poly, d_polyim
        ))
        kernels['measure_correlator']((blocks_r,), (threads,), (
            d_poly, np.int32(L), d_corr
        ))
        cp.cuda.Device().synchronize()
        corr_accum += d_corr.get()

    corr_mean = corr_accum / n_meas

    # Masa: m_latt = log(C(0)/C(1))  — najsilniejszy sygnał, odporny na szum
    # C(0) ≈ <|P|²>, C(1) = <P(x)P(x+1)> ~ exp(-m_latt)
    if corr_mean[0] > 0 and corr_mean[1] > 0:
        m_latt = np.log(corr_mean[0] / corr_mean[1])
    else:
        m_latt = float('nan')

    return m_latt, corr_mean


def main():
    print("=" * 65)
    print("Krok 0.3: 4D SU(2) Yang-Mills Monte Carlo — GPU")
    print("=" * 65)

    # Parametry
    L_values    = [4, 6, 8]                      # rozmiary kratki
    beta_values = [2.2, 2.3, 2.4, 2.5, 2.6]     # silne → słabe sprzężenie
    n_therm     = 1000
    n_meas      = 500
    b0          = 11 / (16 * np.pi**2)           # 1-loop beta-funcion SU(2) w 4D

    results = {}
    print(f"L ∈ {L_values}, β ∈ {beta_values}")
    print(f"Termlizacja: {n_therm} sweepów, Pomiarów: {n_meas}")
    print()

    for L in L_values:
        results[L] = {}
        for beta in beta_values:
            t0 = time.time()
            m_latt, corr = run_su2_mc(L, beta, n_therm=n_therm, n_meas=n_meas)
            elapsed = time.time() - t0

            # Masa fizyczna (heurystyczne przybliżenie 1-loop)
            g2 = 4.0 / beta   # SU(2): β = 4/g²
            a_over_Lambda = np.exp(1.0 / (2 * b0 * g2)) * g2  # a × Λ_QCD ∝ exp(+1/(2b₀g²)) × g²
            m_phys_hat = m_latt * a_over_Lambda if not np.isnan(m_latt) else float('nan')

            results[L][beta] = {
                'm_latt': float(m_latt),
                'm_phys_hat': float(m_phys_hat),
                'corr': corr.tolist(),
                'time': elapsed
            }
            corr_str = ' '.join(f'{c:.4f}' for c in corr[:min(4,len(corr))])
            print(f"L={L}, β={beta:.1f}: m_latt={m_latt:.4f}, "
                  f"corr=[{corr_str}], t={elapsed:.1f}s")

    # Zapis
    outdir = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(outdir, exist_ok=True)
    outfile = os.path.join(outdir, 'su2_mc_gap_result.json')
    with open(outfile, 'w') as f:
        json.dump({'L_values': L_values, 'beta_values': beta_values,
                   'n_therm': n_therm, 'n_meas': n_meas,
                   'results': {str(k): v for k, v in results.items()}}, f, indent=2)
    print(f"\nWyniki zapisane: {outfile}")

    # Sprawdź finite-size scaling: m_latt(β,L) → m_∞(β) dla L → ∞?
    print("\nFinite-size scaling m_latt(β, L):")
    print(f"{'β':>6} | {'L=4':>10} {'L=6':>10}")
    print("-" * 30)
    for beta in beta_values:
        vals = [results[L][beta]['m_latt'] for L in L_values]
        print(f"{beta:6.1f} | {vals[0]:10.4f} {vals[1]:10.4f}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
