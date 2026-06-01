# Finke-Watzky Model — Fitting with Analytical vs ODE Numerical Solution

**Course:** Physics of Molecular Diseases — Week 1  
**Question 7:** Use either the analytical solution or the ODE numerical results to fit the aggregation datasets obtained via WebPlotDigitizer. What can you conclude?

---

## Overview

This script fits the F-W 2-step model to digitized protein aggregation data using **two independent approaches**:

1. **Analytical fit** — closed-form solution (eq. 2/3, Morris et al. 2008) via `scipy.optimize.curve_fit`
2. **ODE numerical fit** — integrate the ODE numerically at each iteration via `scipy.integrate.solve_ivp` + `scipy.optimize.minimize`

The two approaches are applied to the same datasets and their results compared, providing both a cross-validation of the fitting procedure and mechanistic insights into the aggregation kinetics.

---

## The Two Fitting Approaches

### Approach A — Analytical fit

The F-W analytical solution is used directly as the fitting function:

$$[B]_t = [A]_0 - \frac{\dfrac{k_1}{k_2} + [A]_0}{1 + \dfrac{k_1}{k_2[A]_0} \exp\!\bigl((k_1 + k_2[A]_0)\,t\bigr)}$$

`scipy.optimize.curve_fit` minimises the sum of squared residuals by adjusting $k_1$ and $k_2$.

**Advantages:** Fast, closed-form, standard errors on parameters available from covariance matrix.  
**Limitation:** Only valid for the exact 2-step F-W mechanism as written.

### Approach B — ODE numerical fit

At each iteration of the optimiser, the ODE is solved numerically:

$$\frac{d[A]}{dt} = -k_1[A] - k_2[A][B], \qquad [B] = [A]_0 - [A]$$

`scipy.optimize.minimize` (Nelder-Mead) minimises the sum of squared residuals by adjusting $k_1$ and $k_2$. Optimisation is performed in **log-space** ($\theta = \log k$) for numerical stability, since the parameters span several orders of magnitude.

```python
def fw_ode_residuals(params, t_data, B_data):
    k1, k2 = np.exp(params)   # back-transform from log-space
    B_pred  = fw_ode_solve(k1, k2, t_data)
    return np.sum((B_data - B_pred) ** 2)

result = minimize(fw_ode_residuals, x0=[log(k1_guess), log(k2_guess)],
                  args=(t_data, B_data), method='Nelder-Mead')
k1_fit, k2_fit = np.exp(result.x)
```

**Advantages:** Flexible — extending to more complex models (dissociation, fragmentation, seeding) requires only changing the `rhs()` function; the fitting infrastructure remains identical.  
**Limitation:** Slower than analytical; no automatic standard errors.

---

## Datasets

Digitized from Morris et al. (2008) using WebPlotDigitizer:

| Dataset | Figure | Protein | Method | Paper's $k_1$ |
|---------|--------|---------|--------|---------------|
| Amyloid- $\beta$ | Fig. 2 | Kelly et al. | ThT fluorescence | $8\times10^{-6}$ $h^{-1}$ |
| $\alpha$-synuclein | Fig. 7 | Fink et al. | ThT fluorescence | $4\times10^{-5}$ $h^{-1}$ |

---

## Results

| Dataset | $k_1$ analytical | $k_1$ ODE | $R^2$ analytical | $R^2$ ODE |
|---------|-----------------|-----------|-----------------|-----------|
| Amyloid- $\beta$ | $3.43\times10^{-5}$ $h^{-1}$ | $3.43\times10^{-5}$ $h^{-1}$ | 0.9993 | 0.9993 |
| $\alpha$-synuclein | $5.12\times10^{-5}$ $h^{-1}$ | $5.12\times10^{-5}$ $h^{-1}$ | 0.9985 | 0.9985 |

The two approaches give **identical** results to at least 3 significant figures.

---

## Conclusions

### 1. Analytical = ODE (cross-validation)
Both approaches return the same $k_1$, $k_2$, and $R^2$ to high precision. This confirms that the analytical solution is correct and that the ODE solver is working as expected — any difference would signal a bug in one of the implementations.

### 2. $k_2 \gg k_1$ (slow nucleation, fast growth)
The ratio $k_2/k_1$ is on the order of thousands for both proteins. This is the central result of the F-W model: nucleation is the slow, rate-limiting step, while autocatalytic growth is fast once a nucleus forms. This is why protein aggregation shows a pronounced lag phase followed by a sharp sigmoidal rise.

### 3. Amyloid-β vs α-synuclein
$\alpha$-synuclein has a slightly higher $k_1$ than amyloid-$\beta$ under these experimental conditions, meaning it nucleates somewhat faster. The growth rate $k_2$ is comparable. This kind of quantitative comparison between disease proteins — made possible by the F-W deconvolution — is one of the main contributions of Morris et al. (2008).

### 4. Therapeutic implications
Because $k_1$ and $k_2$ are deconvoluted, therapeutic strategies can be targeted:
- **Inhibit nucleation** ($k_1$): prevents aggregation from starting — the most effective intervention since nucleation is the bottleneck
- **Inhibit growth** ($k_2$): slows propagation once nuclei form

### 5. ODE approach is more general
Although slower, the ODE fitting approach can be extended to more complex models without changing the fitting infrastructure — only the `rhs()` function needs to be updated. This makes it the preferred approach when exploring beyond the basic F-W mechanism (e.g. adding seeding, dissociation, or fragmentation terms).

---

## Implementation Notes

### Log-space optimisation
ODE fitting is performed in log-space to improve numerical stability:

```python
# Instead of optimising k1, k2 directly (range: 1e-6 to 1e2):
x0 = [np.log(k1_guess), np.log(k2_guess)]

# Then back-transform:
k1, k2 = np.exp(params)
```

This ensures the optimiser treats multiplicative changes equally at all scales, preventing it from ignoring small-value parameters.

### Why Nelder-Mead for ODE fitting?
`curve_fit` requires a Jacobian-compatible function (analytical derivatives or numerical finite differences that are cheap to compute). For ODE-based models, each function evaluation involves a full numerical integration, making gradient-based methods expensive. Nelder-Mead is a derivative-free simplex method that works well for smooth, low-dimensional problems like this.

---

## Key References
- Prof. Ala Trusina, Lecture notes from the course "Physics of Molecular Diseases", Niels Bohr Institute, 2020
- Morris, A. M., Watzky, M. A., Agar, J. N., & Finke, R. G. (2008). Fitting neurological protein aggregation kinetic data via a 2-step, minimal/"Ockham's Razor" model: the Finke-Watzky mechanism of nucleation followed by autocatalytic surface growth. *Biochemistry*, 47(8), 2413–2427.
