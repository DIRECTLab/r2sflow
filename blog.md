# From Noise to Point Clouds: Flow Matching for LiDAR

<!-- DRAFT v0. Items in [TODO: ...] are placeholders for results/decisions we have not made yet. Nothing in a TODO should be read as a finding. -->

Isaac Peterson and Jacob Morrey

---

## 1. Why a state estimator cares about generating LiDAR

A Kalman filter needs a measurement model: given the state, what does the sensor report? For a spinning LiDAR in a real building that model is hopeless to write down. Returns depend on materials, incidence angle, clutter, and the robot's own body. What we *can* do is collect scans and learn the distribution of what a scan looks like.

That is a generative model: a way to draw samples from $p(\text{scan})$. Once you have one, you can (a) fill in simulated data that looks real, (b) check whether a real scan is in-distribution, and (c) use it as a prior in a Bayesian update, which is the structure of everything we did in Parts 8 and 9 of the course.

This post is about one family of generative models, **diffusion** and **flow matching**, and one application of it, **LiDAR range images**. We cover the math, then the specific model our lab uses (R2Flow), then what happened when we pointed it at a Unitree Go2-W robot, simulated and real.


## 2. Diffusion in five minutes

Pick a data sample $x_1$ (here, a LiDAR scan; we flip to the flow-matching naming convention where $x_1$ is data and $x_0$ is noise). A diffusion model defines a *forward* process that gradually destroys $x_1$ with Gaussian noise, and trains a network to undo it.

In the standard DDPM view, the forward process is a Markov chain

$$q(x_{k}\mid x_{k-1}) = \mathcal N\!\big(\sqrt{1-\beta_k}\,x_{k-1},\ \beta_k I\big),$$

which has the convenient closed form $x_k = \sqrt{\bar\alpha_k}\,x_{\text{data}} + \sqrt{1-\bar\alpha_k}\,\varepsilon$ with $\varepsilon\sim\mathcal N(0,I)$. The network is trained to predict $\varepsilon$ (or equivalently the score $\nabla_x \log p_k(x)$) from the noisy $x_k$. Sampling runs the chain backwards, one small step at a time, typically hundreds to a thousand steps.

Two facts matter for what follows:

1. The sampling cost is the number of network evaluations (NFE). Each one is a full forward pass of a large network.
2. In the continuous-time limit the reverse process has a deterministic twin, the *probability flow ODE*, whose trajectories carry the same marginals. Sampling becomes "solve an ODE from noise to data." [TODO: cite Song et al. 2021, *Score-Based Generative Modeling through SDEs*.]

That second fact is the bridge to flow matching.

## 3. Flow matching and rectified flow

Flow matching skips the SDE machinery and trains the ODE's velocity field directly.

**Setup.** Let $x_0\sim\mathcal N(0,I)$ be noise and $x_1\sim p_{\text{data}}$ a scan. We want a vector field $v_\theta(t,x)$ such that integrating

$$\frac{dx}{dt} = v_\theta(t,x),\qquad x(0)=x_0$$

to $t=1$ lands on a sample from $p_{\text{data}}$.

**The training trick.** Pair a noise draw with a data point and connect them with a straight line:

$$x_t = (1-t)\,x_0 + t\,x_1,\qquad \frac{dx_t}{dt} = x_1 - x_0.$$

The regression target is the constant velocity $u = x_1-x_0$, and the loss is

$$\mathcal L(\theta) = \mathbb E_{t,\,x_0,\,x_1}\,\big\| v_\theta(t, x_t) - (x_1 - x_0) \big\|^2 .$$

That is the entire objective. No noise schedule, no score, no $\beta_k$. In our codebase this is `torchcfm`'s `ConditionalFlowMatcher` with `sigma=0`, which the repo's comment calls "equivalent to 1-rectified flow if sigma is 0" (`train.py`). With `sigma>0` the path gets a small Gaussian perturbation; the repo exposes this as `flow.sigma` and also offers an optimal-transport pairing (`otcfm`) in place of random pairing.

**Why does regressing on one straight line give a field that works?** Many different $(x_0,x_1)$ pairs pass through the same point $(t,x)$ with different velocities. The minimizer of the squared loss is their average, $v^\star(t,x)=\mathbb E[x_1-x_0\mid x_t=x]$, and the flow defined by that conditional mean transports $\mathcal N(0,I)$ to $p_{\text{data}}$ exactly. [TODO: decide whether to include the short proof or cite Lipman et al. 2023 / Liu et al. 2023.] This is the same "best predictor is a conditional expectation" argument as the best linear predictor from Part 2, just without the restriction to linear.

**Connection to Part 5 (sampling).** In the course we sampled a Gaussian by pushing standard normal draws through a map (Cholesky) and used the change-of-variables theorem to get the new density. A flow is that idea made continuous: the ODE defines an invertible map $x_0\mapsto x_1$, and the density transforms by the instantaneous change-of-variables formula $\tfrac{d}{dt}\log p_t(x_t) = -\nabla\!\cdot v_\theta(t,x_t)$. Our model never evaluates that divergence, but it is why the "noise in, data out" picture is a legitimate change of variables and not just a trick.

### Straightness is a numerical-analysis statement

If the learned trajectories were perfectly straight, one Euler step would be exact: $x_1 = x_0 + v_\theta(0,x_0)$. They are not straight, because the conditional-mean field averages over crossing paths, so the true trajectories curve. Euler's global error scales with step size times curvature, so more curvature means more steps for the same accuracy. This is the same trade the reference posts on numerical stability are about (Gundersen on conditioning, Cook on two-pass vs. one-pass variance): the same quantity computed two ways, and the cheap way costs you digits in a regime you can predict.

**Reflow** is the fix. Train a first model (1-RF). Use it to generate $(x_0, x_1)$ pairs by running the ODE, where $x_1$ is now *generated* from the specific $x_0$. Retrain on those coupled pairs (2-RF). Because each $x_1$ is a deterministic function of its $x_0$, the straight-line targets no longer cross, so the new field is straighter and fewer steps suffice. Optionally distill further to 1-4 steps ($k$-TD).

## 4. LiDAR as an image

Diffusion and flow models are most developed for images, so the common trick is to turn a scan into one. A spinning LiDAR sweeps azimuth and elevation, so each return at range $r$, azimuth $\phi$, elevation $\theta$ is placed in pixel $(\text{row}(\theta),\text{col}(\phi))$ with value $r$. That gives a **range image** of shape $H\times W$, where $H$ is the number of beams.

Details that matter, taken from the repo's `LiDARUtility` and data docs:

- **Channels.** Depth, optionally reflectance. Missing returns ("raydrop") are encoded as an out-of-range value; after log-scale encoding an empty pixel is exactly $-1$.
- **Depth normalization.** The pretrained KITTI-360 model uses a log scale, $\log(r+1)/\log(r_{\max}+1)$, over a $[1.45, 80]$ m window, so near range gets more resolution than far range.
- **Round trip.** Range image to point cloud is just the inverse spherical projection using the known ray angles, so nothing is lost beyond quantization to the grid.

The network is an HDiT (hierarchical diffusion transformer with neighborhood attention, about 80.9M parameters per the repo's table), with a learnable positional embedding tied to the token grid.

## 5. R2Flow

R2Flow (Nakashima et al., ICRA 2025, *Fast LiDAR Data Generation with Rectified Flows*) combines the pieces above: rectified flow on range images, then reflow, then timestep distillation, to generate LiDAR in few steps. [TODO: pull one headline number from the paper (steps vs. quality vs. R2DM) after we confirm it directly from the arXiv PDF 2412.02241 rather than from memory.]

The repo's three-stage recipe:

| Stage | What is trained | Training data |
|---|---|---|
| 1-RF | velocity field from scratch | real scans, fresh Gaussian noise |
| 2-RF | start from 1-RF, reflow | 1M $(x_0,x_1)$ pairs from 1-RF using `dopri5` |
| $k$-TD | start from 2-RF, $k\in\{1,2,4\}$ | 100k pairs from 2-RF |

## 6. Our setting: Go2-W, simulated and real

Our lab's question is not "can we generate KITTI scans" but "can we use a generative model to bridge simulated and real LiDAR on a Unitree Go2-W robot." Everything below is from our fork's docs and code; [TODO: confirm names and framing with the team].

**Data.** [TODO: this is where the `lidargen` repo would be described. It was not accessible when this draft was written. See open questions.] The fork's converters turn ROS bags into range-image datasets:

- Simulated Livox: 40 rings, $-28.2^\circ$ to $+29.5^\circ$, grid $40\times512$.
- Simulated L1 (front-mounted, closest to the real robot): $64\times512$ over the lower hemisphere.
- Simulated navigation episodes: $64\times1024$ over $-24^\circ$ to $+12^\circ$, chosen so the pretrained KITTI-360 weights load unchanged.
- Real Go2 `utlidar` L1 bags, accumulated over several messages because one message is a $\sim$65 ms slice of a non-repetitive spiral.

**Fine-tuning.** Start from the released `r2flow-kitti360-1rf` weights, keep only the depth channel of the two layers that depend on channel count, and train on the simulated scans at a low learning rate ($2\times10^{-5}$). Start from 1-RF rather than 2-RF because training on real data with fresh noise undoes reflow's straightening anyway.

**A domain-gap probe.** The model is unconditional: it has no encoder, so "running a real scan through it" is transport, not reconstruction. The fork's `visualize_flow.py` substitutes a real scan for the Gaussian $x_0$ and Euler-integrates, keeping every state:

$$x_{k+1} = x_k + \tfrac1N\,v_\theta(t_k, x_k),\qquad t_k = k/N.$$

How far an input moves is a read on whether it is in-distribution. The docs report that sim scans keep their walls and doorways through all 8 steps, while the real Go2 scan loses its structure by step 6 and becomes a smooth sim-like gradient. The docs are careful to say this is a *probe*, not translation, and list why the real bag is far out of distribution (different elevation band, median range 0.42 m, 93% of returns within 1.45 m, mostly the robot's body).

[TODO: figure. Reproduce the 9-panel range-image strip for one sim scan and one real scan. Needs a checkpoint and GPU; ask team which checkpoint to use.]

## 7. Measuring it honestly

The fork has an evaluation pipeline for "did this change help." Three design choices are worth a section because they are the statistical-estimation content of the project:

1. **Null floors.** A reference-vs-reference score is not zero when the data is one continuous trajectory. The docs measure three floors (random, interleaved, contiguous splits) and get an increasing ordering consistent with scene drift. Reading a model's score against zero credits it with a gap that is just drift.
2. **A control for density.** Sim scans are about 80% filled and real about 39%, so a sim-vs-real gap could be explained by point count alone. An extra arm A′ randomly matches fill to real to separate "domains differ" from "one has more points."
3. **Effective sample size.** Extra noise draws per scan are correlated, so $n \neq \text{scans}\times\text{repeats}$. The design effect $1+(K-1)\,\mathrm{ICC}$ gives an $n_{\text{eff}}$, and confidence intervals use block *subsampling* because a bootstrap's resampled duplicates put maximal-kernel self-pairs into a U-statistic and bias KID upward.

One result from the docs that ties back to Section 3. Integrating a 1-RF model *near* $t=1$ makes the data worse, not better:

| $t_0$ | Euler steps | FRID |
|---|---|---|
| 0.995 | 1 | 0.098 |
| 0.99 | 3 | 0.157 |
| 0.95 | 13 | 5.27 |
| 0.9 | 26 | 3.52 |
| 1.0 | 0 (identity) | 0.107 |

The docs flag this as a first run on 158 scans with $N_{\text{eff}}\approx 40$: it shapes how to read the numbers but decides nothing. They attribute it to high trajectory curvature near the endpoints of 1-RF, which is the curvature argument from Section 3, and note that reflow is the paper's fix. [TODO: decide whether to rerun this ourselves with a 2-RF checkpoint, which would make a clean "straightening helps" figure.]

## 8. Our experiment

[TODO: This is the section that makes the post ours. Pick one, scoped to what we can finish before the deadline.]

- **Option A, straightness.** For a trained 1-RF and 2-RF, plot sample quality against Euler steps $\{1,2,4,8,16,32,\text{dopri5}\}$. Prediction: 2-RF reaches the plateau in far fewer steps.
- **Option B, curvature measured directly.** Compute $\int_0^1\|v_\theta(t,x_t)-(x_1-x_0)\|^2dt$ along sampled trajectories as a straightness score, and relate it to Euler error.
- **Option C, transport as a probe.** Quantify how far sim vs. real scans move under the flow as a function of $t_0$, with the null floor and fill-matched control from Section 7.

[TODO: results, plots, compute used.]

## 9. Takeaways

- Diffusion and flow matching differ in what the network regresses, not in the goal. Flow matching's target is a constant velocity along a straight line between noise and data.
- Sampling cost is ODE-integration cost. Straight trajectories mean few steps, and reflow buys straightness.
- LiDAR becomes tractable as an image problem via spherical projection, but sensor geometry, fill fraction, and depth normalization decide whether a pretrained model transfers.
- Evaluating generative models on small, temporally correlated robot data needs null floors, density controls, and effective sample sizes, or you will read drift as signal.

## References

[TODO: finalize format; verify each before submission.]

- Nakashima et al., *Fast LiDAR Data Generation with Rectified Flows*, ICRA 2025, arXiv:2412.02241.
- Lipman et al., *Flow Matching for Generative Modeling*, ICLR 2023.
- Liu et al., *Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow*, ICLR 2023.
- Ho et al., *Denoising Diffusion Probabilistic Models*, NeurIPS 2020.
- Song et al., *Score-Based Generative Modeling through Stochastic Differential Equations*, ICLR 2021.
- Lee et al., *Improving the Training of Rectified Flows*, arXiv:2405.20320 (the U-shaped timestep sampler in the repo).
- Tong et al., `torchcfm`, https://github.com/atong01/conditional-flow-matching.
- Reference posts for style: Gundersen (matrix inversion), Epperly (Gaussians in trace estimation), Cook (three ways to compute standard deviation).
- [TODO: lidargen / LiDARGen (Zyrianov et al., ECCV 2022) if relevant.]