# Methods and detail

Long form detail moved out of the README.


## DSB does not improve with more iterations


IPF is supposed to converge. Mine does not, and this is the most interesting
negative result in the repo.

![DSB across IPF iterations](results/dsb-ipf.png)

Median sliced W2 by IPF iteration:

| pair | ipf 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| circle to moons | 0.607 | 0.466 | 0.424 | 0.476 | 0.371 | 0.376 | 0.392 | 0.400 | 0.416 | 0.356 |
| gaussian to 8 gaussians | 0.825 | 0.512 | 0.669 | 0.681 | 0.650 | 0.615 | 0.854 | 0.970 | 0.974 | 0.969 |
| moons to spiral | 0.831 | 0.861 | 0.868 | 0.806 | 0.793 | 0.806 | 0.786 | 0.793 | 0.868 | 0.834 |

One pair gets slowly better, one gets clearly worse after iteration 5, and one
never really moves. Two of ninety DSB runs went non finite entirely, both on
moons to spiral at iterations 8 and 9.

The mechanism is that each network trains on trajectories produced by the other,
so any error in one becomes training data for the next. Nothing corrects it, and
on the pair where it compounds fastest the whole loop walks off. The internal
regression losses fall smoothly the whole time, from 18 down to about 2.5, which
is what makes this hard to notice: the thing being minimised is going down while
the thing you care about is going up.

**What I am not claiming.** This is my implementation of DSB, not DSB. I found
three real bugs in it (below) and there may be a fourth. Published DSB results
are better than this. What I can say is that getting DSB to work took me
substantially more effort than bridge matching, which took one function and
worked first time, and that matches what the literature moved to.


## The static bridge


Sinkhorn is worth looking at directly, because the plan is the object everything
else approximates. As eps shrinks the coupling concentrates from a diffuse blur
onto a near deterministic map:

![entropic OT plans at four regularisation levels](results/sinkhorn-plan.png)

It is solved in log space. The direct version multiplies exp(-C/eps) and
underflows to exactly zero for any eps below about 0.05 on these toys, which
shows up as a plan full of NaN after the first iteration.

Note the one row where Sinkhorn does badly: circle to moons, 1.057 against a
do nothing baseline of 1.292. Entropic plans are spread out by construction, and
the barycentric projection averages that spread, so every source point maps to
something near the middle of its options. When the target is two separated
crescents the middle is empty space. That is a real limitation of turning a
coupling into a map, not a tuning failure.


## What I got wrong


Three bugs, and the order matters because each one hid the next.

**One. The backward drift was reversed twice.** A DSB backward drift is fit
against a target arranged so that stepping with positive dt moves backward in the
trajectory index. The reversal is already inside the target. I also integrated it
with a negative dt, which reverses again. Reversing a Brownian process onto its
own starting distribution scored 4.94 that way against a do nothing baseline of
0.56, and the IPF loop diverged to a forward loss of 1.5e11. With the sign fixed
the same test gives 0.072.

**Two. The forward network was trained against mirrored time labels.** After
fixing the sign the losses fell smoothly and the transport was still worse than
doing nothing. A forward trajectory has t_k = k dt so the next state sits at
t_k + dt, but a reverse trajectory has t_k = 1 - k dt so the next state sits at
t_k - dt. I used plus in both cases. The network learned the right field indexed
by the wrong clock, which the loss cannot see.

**Three. My distance metric was wrong, and it was the reason I doubted the other
two fixes.** Sliced Wasserstein sorts both point clouds along random directions
and compares them. Comparing element i to element i is only correct when the two
sets are the same size. With 4000 samples against 8000 it compares all of the
first against the lowest half of the second, and reports a large distance between
two samples of the same distribution. A bridge matching run whose output matched
the target mean to 0.01 and its standard deviation to 0.014 was scored at 3.04. I
went looking for a transport bug that did not exist. Both are now interpolated
onto a shared quantile grid, and there is a test that runs 500 against 8000.

The lesson I actually take from this is that the metric deserved a test before
any method did. Three of the four things I tested first were fine.
