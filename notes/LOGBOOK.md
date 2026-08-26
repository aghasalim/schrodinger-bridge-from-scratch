# Logbook

## 2026-08-26 — the backward drift was reversed twice
**Tried:** first DSB run, gaussian to 8 gaussians, 4 IPF iterations.
**Measured:** forward loss went 15.8, 1903, 1.1e8, 1.5e11. Reversing a Brownian process onto its own start scored W2 4.94 against a do nothing baseline of 0.56.
**Concluded:** not the known DSB instability, my bug. The DSB target b_hat = b_k + (x_k - x_k+1)/dt is arranged so that stepping with POSITIVE dt moves backward in the index. The reversal is already in the target. I integrated with negative dt as well, reversing twice. With the sign fixed the same reversal test gives 0.072 and the IPF losses fall monotonically. Backward integration now means descending time labels with a positive position update, which is not the same as swapping t0 and t1, and there is a test asserting exactly that.

## 2026-08-26 — the forward network learned the right field on the wrong clock
**Tried:** after the sign fix, ran 8 IPF iterations expecting convergence.
**Measured:** losses fell 18.1 to 2.5 monotonically. W2 to target went 4.01, 4.14, 4.20, 4.17, 4.19, against an untransported baseline of 2.38. Getting worse, smoothly.
**Concluded:** a forward trajectory has t_k = k dt so the next state is at t_k + dt; a reverse trajectory has t_k = 1 - k dt so the next state is at t_k - dt. I used plus for both, so the forward network was trained against mirrored time labels. The regression loss cannot see this because the field it is fitting is perfectly learnable, just indexed wrong. Fixed by passing the sign of the time step into the fitting function.

## 2026-08-26 — the metric was wrong and it made me doubt two correct fixes
**Tried:** bridge matching, which should be the easy method, scored 2.95 against a 2.63 untransported baseline. Started looking for a third transport bug.
**Measured:** the produced distribution had mean [0.041, -0.045] and std 2.834 against the target's [0.053, -0.046] and std 2.848. Matching to two decimals while the metric said 3.04.
**Concluded:** the metric, not the method. Sliced Wasserstein sorts both clouds along random directions and compares them elementwise, which is only valid at equal sample sizes. With 4000 against 8000 it compared all of the first against the lowest half of the second. Both are now interpolated onto a shared quantile grid and there is a test at 500 against 8000. Real lesson: the metric deserved a test before any method did, and it was the one thing I had not tested.

## 2026-08-26 — bridge matching beats my DSB by 2x to 12x at a quarter the cost
**Tried:** full comparison, 3 pairs x 3 seeds, Sinkhorn at three eps values, DSB for 10 IPF iterations, bridge matching with ODE and SDE sampling. 448 s total on an M4 CPU.
**Measured:** bridge matching SDE gets 0.048, 0.104, 0.060 across the three pairs in 8.0 s. My best DSB gets 0.328, 0.367, 0.727 in 33 s. Sinkhorn at eps 0.03 gets 1.057, 0.127, 0.211 in 2 s but is capped at 2000 points by the cost matrix. More IPF iterations mostly do not help: one pair improves slowly, one degrades clearly after iteration 5, and two of ninety runs went non finite.
**Concluded:** matches what the literature did, which was move to simulation free bridge matching. Worth being careful about the claim though: this is my DSB after three bug fixes, not DSB, and published results are better. The defensible version is that DSB cost me days and bridge matching worked first time, and that the internal losses fall smoothly while the transport degrades, which makes the failure mode genuinely hard to catch. Also worth keeping: Sinkhorn does badly on circle to moons, 1.057 against 1.292 for doing nothing, because barycentric projection averages a spread out plan and the average of two separated crescents is empty space. That is a limitation of turning a coupling into a map, not a tuning failure.
