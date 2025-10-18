# Analysis of Simulation Results

## Reproduction of Original Paper

I tried to reproduce Benjamini & Hochberg (1995) Figure 1, but got some different results.

**GitHub Repository:** https://github.com/Soul22238/STATS607UnitProject2

**Main differences:**

1. **Lower power at small m**: The paper showed power close to 1.0 even for m=4 with mode D. I got much lower power (around 0.1-0.5) with weak signals. Maybe I misunderstood their signal specification, or they implemented the "linearly decreasing" strategy differently than I did.

2. **Weird trends in m**: The paper showed power generally decreasing as m increases. Mine sometimes goes up then down. With m=4 and mode D, there's only 1 non-null hypothesis, and if it gets assigned to the weakest signal (L/4), power is terrible. At m=8, you get better signal distribution, so power can actually increase before the multiple testing penalty kicks in. Could also be that their decreasing allocation works differently from my 4:3:2:1 weights.

**What I changed:**

I made separate figures for each L value instead of combining them. Different L's show really different patterns - with L=5, all methods struggle and look similar. With L=15, FDR dominates and mode matters a lot. Seemed worth visualizing separately.

## Simulation Neutrality

**Is it fair?** 
All methods see the same data and use α=0.05, so that's fair. But FDR is designed to allow some false discoveries, so it naturally wins in power comparisons - not really a "neutral" test.

**Design choices that help certain methods:**
- Using four discrete signal levels (L/4, L/2, 3L/4, L) creates "groups" of similar p-values. Step-up procedures like Hochberg and FDR can exploit this better than Bonferroni's fixed threshold.
- No correlation between tests is pretty idealized. Real data has correlations, which might hurt some methods more than others.

**Realism:**
This is pretty idealized - known variance, perfect normality, independence. Real data has estimated variance, outliers, correlations, etc. So these results are best-case scenarios.

## What I'd Change

1. Add correlation between tests (like AR(1) structure) to be more realistic
2. Try sparse signals (1-2 strong signals) and clustered signals instead of just D/E/I modes
3. Use t-tests with estimated variance instead of z-tests
4. Maybe use fewer simulations for large m to save time

## Visualizations

The separate L plots showed some things I didn't expect:

- Mode matters WAY more with strong signals. At L=15, mode I hits power=1.0 immediately, mode D barely reaches 0.9 even at m=64.
- FDR's advantage isn't constant - it's small for L=5, huge for L=15 with mode I.
- Null ratio (the four rows) makes less difference than I thought. Mode and L dominate.

## Surprising Stuff

1. Power doesn't always decrease with m - sometimes it goes up first because of allocation randomness at small m.
2. The gap between methods grows with m - at m=4 they're close, at m=64 FDR can be 20-30% better.
