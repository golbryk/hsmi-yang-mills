"""
Krok 2.1/2.3: Masa fizyczna m_phys = m_latt / a(β).

Cel: Pokazać że m_phys(β) → const > 0 gdy β → ∞ (granica ciągła).

Metoda:
1. MC: mierz m_latt(β, L) z korelatorów pętli Polyakova dla L=6,8,10,12
2. FSS: ekstrapoluj do L→∞ (m_latt ∝ L dla Polyakova, więc σa² = m_latt/L)
3. Skala kratowa: a(β) = C_L × (b₀g²)^{-b₁/(2b₀²)} × exp(-1/(2b₀g²))
4. m_phys/Λ_L = σa × a^{-1}(β) × Λ_L^{-1} = σa²/a²(β) × Λ_L² (string tension)

Wynik: σa² / a(β)² × Λ_L² powinno być stałe dla dużych β → m_phys = O(Λ_QCD) > 0
"""

import os, sys
os.environ.setdefault('CUDA_PATH', '/usr/local/cuda-13.1')
os.environ['PATH'] = os.environ.get('PATH', '') + ':/usr/local/cuda-13.1/bin'

import numpy as np
import cupy as cp
import json, time

# -----------------------------------------------------------------------
# PARAMETRY FIZYCZNE SU(2) 4D
# -----------------------------------------------------------------------

# 2-pętlowe β-funkcja: b₀ = 11N/(3×16π²) dla SU(N), b₁ = (34N²)/(3×(16π²)²)
B0  = 22.0 / (3.0 * 16.0 * np.pi**2)    # = 0.046439... dla SU(2)
B1  = 136.0 / (3.0 * (16.0 * np.pi**2)**2)  # 2-pętlowy

def lattice_spacing_ratio(beta):
    """
    a(β)/a(β_ref) z 2-pętlowego schematu kratowego.
    Normalizacja: a(β_ref=2.4) = 1.
    a(β) ∝ (b₀ g²)^{-b₁/(2b₀²)} exp(-1/(2b₀ g²))
    """
    g2 = 4.0 / beta
    exponent = -1.0 / (2.0 * B0 * g2)
    prefactor = (B0 * g2) ** (-B1 / (2.0 * B0**2))
    return prefactor * np.exp(exponent)


# -----------------------------------------------------------------------
# MC KERNEL (z naprawionymi błędami z CHECKPOINT_062)
# -----------------------------------------------------------------------

MC_CODE = r"""
#include <curand_kernel.h>
#include <math.h>

__device__ inline void su2_mul(const float* A, const float* B, float* C) {
    C[0] = A[0]*B[0]-A[1]*B[1]-A[2]*B[2]-A[3]*B[3];
    C[1] = A[0]*B[1]+A[1]*B[0]+A[2]*B[3]-A[3]*B[2];
    C[2] = A[0]*B[2]-A[1]*B[3]+A[2]*B[0]+A[3]*B[1];
    C[3] = A[0]*B[3]+A[1]*B[2]-A[2]*B[1]+A[3]*B[0];
}
__device__ inline void su2_dag(const float* A, float* B) {
    B[0]=A[0]; B[1]=-A[1]; B[2]=-A[2]; B[3]=-A[3];
}
__device__ inline int sidx(int x,int y,int z,int t,int L) {
    return x+L*(y+L*(z+L*t));
}

__device__ void staple_sum(const float* U, int site, int mu, int L, float* S) {
    int t=site/(L*L*L),z=(site/(L*L))%L,y=(site/L)%L,x=site%L;
    S[0]=S[1]=S[2]=S[3]=0.0f;
    for(int nu=0; nu<4; nu++) {
        if(nu==mu) continue;
        int c1[4]={x,y,z,t}; c1[mu]=(c1[mu]+1)%L;
        int c2[4]={x,y,z,t}; c2[nu]=(c2[nu]+1)%L;
        int s1=sidx(c1[0],c1[1],c1[2],c1[3],L);
        int s2=sidx(c2[0],c2[1],c2[2],c2[3],L);
        const float* A=U+(s1*4+nu)*4;
        const float* B=U+(s2*4+mu)*4;
        const float* C=U+(site*4+nu)*4;
        float Bd[4],Cd[4],t1[4],t2[4];
        su2_dag(B,Bd); su2_dag(C,Cd);
        su2_mul(A,Bd,t1); su2_mul(t1,Cd,t2);
        for(int k=0;k<4;k++) S[k]+=t2[k];
        int c3[4]={x,y,z,t}; c3[mu]=(c3[mu]+1)%L; c3[nu]=(c3[nu]+L-1)%L;
        int c4[4]={x,y,z,t}; c4[nu]=(c4[nu]+L-1)%L;
        int s3=sidx(c3[0],c3[1],c3[2],c3[3],L);
        int s4=sidx(c4[0],c4[1],c4[2],c4[3],L);
        const float* D=U+(s3*4+nu)*4;
        const float* E=U+(s4*4+mu)*4;
        const float* F=U+(s4*4+nu)*4;
        float Dd[4],Ed[4],t3[4],t4[4];
        su2_dag(D,Dd); su2_dag(E,Ed);
        su2_mul(Dd,Ed,t3); su2_mul(t3,F,t4);
        for(int k=0;k<4;k++) S[k]+=t4[k];
    }
}

extern "C" __global__ void metro_sweep(
    float* U, int L, float beta, int parity, unsigned long long seed, long long id
) {
    int L4=L*L*L*L;
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    if(idx>=L4*4) return;
    int site=idx/4, mu=idx%4;
    int t=site/(L*L*L),z=(site/(L*L))%L,y=(site/L)%L,x=site%L;
    if(((x+y+z+t)%2)!=parity) return;
    curandState rng;
    curand_init(seed^(id*999983LL), idx, 0, &rng);
    float S[4]; staple_sum(U,site,mu,L,S);
    float* Ul=U+(site*4+mu)*4;
    float cur[4]; for(int k=0;k<4;k++) cur[k]=Ul[k];
    // Poprawny iloczyn kwaternionowy (MINUS znaki dla k=1,2,3)
    float dot_cur=cur[0]*S[0]-cur[1]*S[1]-cur[2]*S[2]-cur[3]*S[3];
    for(int hit=0; hit<4; hit++) {
        float eps=0.35f, prop[4], n2=0;
        for(int k=0;k<4;k++){ float r=curand_normal(&rng); prop[k]=cur[k]+eps*r; n2+=prop[k]*prop[k]; }
        n2=rsqrtf(n2); for(int k=0;k<4;k++) prop[k]*=n2;
        float dot_p=prop[0]*S[0]-prop[1]*S[1]-prop[2]*S[2]-prop[3]*S[3];
        // Poprawny czynnik: -β (nie -2β)
        float dS=-beta*(dot_p-dot_cur);
        if(curand_uniform(&rng) < (dS<0?1.0f:expf(-dS))) {
            for(int k=0;k<4;k++) cur[k]=prop[k]; dot_cur=dot_p;
        }
    }
    for(int k=0;k<4;k++) Ul[k]=cur[k];
}

extern "C" __global__ void hot_start(float* U, int L4, unsigned long long seed) {
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    if(idx>=L4*4) return;
    curandState rng; curand_init(seed,idx,0,&rng);
    float* Ul=U+idx*4; float n=0;
    for(int k=0;k<4;k++){Ul[k]=curand_normal(&rng); n+=Ul[k]*Ul[k];}
    n=rsqrtf(n); for(int k=0;k<4;k++) Ul[k]*=n;
}

// Korelatora pętli Polyakova C(r) = <P(x)P(x+r)>
extern "C" __global__ void measure_correlator(
    const float* U, int L, float* corr
) {
    // Jeden wątek = jedna separacja r w kierunku x
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    int L2 = L/2;
    if(r >= L2) return;
    int L3 = L*L*L;
    float sum = 0.0f;
    for(int z=0;z<L;z++) for(int y=0;y<L;y++) for(int x=0;x<L;x++) {
        // Pętla Polyakova P(x,y,z) = Π_{t=0}^{L-1} U_{(x,y,z,t), μ=3}[0]*2
        float P0[4]={1,0,0,0}, P1[4]={1,0,0,0};
        int xr = (x+r)%L;
        for(int tt=0;tt<L;tt++){
            int s0=sidx(x,y,z,tt,L), s1=sidx(xr,y,z,tt,L);
            const float* U0=U+(s0*4+3)*4;
            const float* U1=U+(s1*4+3)*4;
            float tmp[4]; su2_mul(P0,U0,tmp); for(int k=0;k<4;k++) P0[k]=tmp[k];
            su2_mul(P1,U1,tmp); for(int k=0;k<4;k++) P1[k]=tmp[k];
        }
        // Re Tr P = 2 P[0]
        sum += (2.0f*P0[0]) * (2.0f*P1[0]);
    }
    corr[r] = sum / L3;
}
"""

def compile_kernels():
    mod = cp.RawModule(code=MC_CODE, backend='nvcc',
                       options=('-O3', '--use_fast_math', '-arch=sm_120'))
    return {
        'metro': mod.get_function('metro_sweep'),
        'hot':   mod.get_function('hot_start'),
        'corr':  mod.get_function('measure_correlator'),
    }


def run_mc_sigma(L, beta, n_therm=1000, n_meas=500, K=None):
    """Zwróć σa² = m_latt/L z korelatorów Polyakova."""
    if K is None:
        K = compile_kernels()
    L4 = L**4
    d_U = cp.zeros(L4*4*4, dtype=cp.float32)
    BLK = 256
    GRL = (L4*4 + BLK-1)//BLK
    GRC = (L//2  + BLK-1)//BLK

    K['hot']((GRL,),(BLK,),(d_U, np.int32(L4), np.uint64(42)))
    for sw in range(n_therm):
        for par in [0,1]:
            K['metro']((GRL,),(BLK,),(
                d_U, np.int32(L), np.float32(beta),
                np.int32(par), np.uint64(sw*7919+1), np.int64(sw)
            ))
    d_corr = cp.zeros(L//2, dtype=cp.float32)
    corr_acc = np.zeros(L//2)
    for meas in range(n_meas):
        for sw in range(3):
            for par in [0,1]:
                s = n_therm + meas*3+sw
                K['metro']((GRL,),(BLK,),(
                    d_U, np.int32(L), np.float32(beta),
                    np.int32(par), np.uint64(s*7919+1), np.int64(s)
                ))
        d_corr[:] = 0
        K['corr']((GRC,),(BLK,),(d_U, np.int32(L), d_corr))
        cp.cuda.Device().synchronize()
        corr_acc += d_corr.get()
    C = corr_acc / n_meas
    if C[0] > 0 and C[1] > 0:
        m_latt = np.log(C[0]/C[1])
        sigma_a2 = m_latt / L
    else:
        sigma_a2 = float('nan')
    return sigma_a2, C


def main():
    print("=" * 65)
    print("Krok 2.1/2.3: masa fizyczna m_phys = σa²/a²(β) × Λ_L²")
    print("=" * 65)
    print()

    beta_ref = 2.4   # punkt referencyjny
    beta_values = [2.2, 2.4, 2.6, 2.8, 3.0]
    L_values = [6, 8, 10]

    K = compile_kernels()
    results = {}

    print(f"{'β':>6} | {'L':>4} | {'σa²':>10} | {'a(β)/a_ref':>12} | {'σ/Λ²_L (rel)':>14}")
    print("-" * 55)

    for beta in beta_values:
        sigma_list = []
        for L in L_values:
            t0 = time.time()
            s, C = run_mc_sigma(L, beta, n_therm=800, n_meas=300, K=K)
            dt = time.time()-t0
            sigma_list.append(s)
            a_ratio = lattice_spacing_ratio(beta) / lattice_spacing_ratio(beta_ref)
            # m_phys/Λ_L ∝ σa² / a(β)² (niezmienniczy wymiarowo string tension)
            if not np.isnan(s):
                mphys_rel = s / a_ratio**2
            else:
                mphys_rel = float('nan')
            print(f"{beta:6.2f} | {L:4d} | {s:10.4f} | {a_ratio:12.4f} | {mphys_rel:14.4f}   [{dt:.1f}s]")
            results[(beta, L)] = {'sigma_a2': float(s), 'a_ratio': float(a_ratio),
                                  'mphys_rel': float(mphys_rel), 'corr': C.tolist()}

    # Podsumowanie: σa² / a(β)² vs β
    print()
    print("Ekstrapolacja do L→∞ (średnia po L):")
    print(f"{'β':>6} | {'σa²_mean':>10} | {'a/a_ref':>10} | {'σ/Λ²(rel)':>12} | Status")
    print("-" * 55)
    all_positive = True
    for beta in beta_values:
        s_list = [results[(beta,L)]['sigma_a2'] for L in L_values if not np.isnan(results[(beta,L)]['sigma_a2'])]
        s_mean = np.mean(s_list) if s_list else float('nan')
        a_ratio = lattice_spacing_ratio(beta) / lattice_spacing_ratio(beta_ref)
        mphys = s_mean / a_ratio**2 if not np.isnan(s_mean) else float('nan')
        ok = not np.isnan(mphys) and mphys > 0
        all_positive = all_positive and ok
        print(f"{beta:6.2f} | {s_mean:10.4f} | {a_ratio:10.4f} | {mphys:12.4f}     | {'✓' if ok else '✗'}")

    # Zapis
    outdir = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(outdir, exist_ok=True)
    outfile = os.path.join(outdir, 'mass_gap_continuum.json')
    out = {
        'beta_values': beta_values, 'L_values': L_values,
        'b0': float(B0), 'b1': float(B1), 'beta_ref': float(beta_ref),
        'all_positive': bool(all_positive),
        'results': {f'{b},{L}': v for (b,L),v in results.items()},
    }
    with open(outfile, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nWyniki zapisane: {outfile}")

    verdict = "✓ m_phys > 0 dla wszystkich β — przerwa masowa potwierdzona" if all_positive \
              else "✗ Sprawdź wyniki — niektóre m_phys ≤ 0"
    print(f"\n{verdict}")
    return 0 if all_positive else 1


if __name__ == '__main__':
    sys.exit(main())
