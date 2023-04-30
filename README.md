Download Link: https://assignmentchef.com/product/solved-cs-234-assignment-3
<br>



These questions require thought, but do not require long answers. Please be as concise as possible.

<h1>1        Policy Gradient Methods (50 pts coding + 15 pts writeup)</h1>

The goal of this problem is to experiment with policy gradient and its variants, including variance reduction methods. Your goals will be to set up policy gradient for both continuous and discrete environments, and implement a neural network baseline for variance reduction. The framework for the policy gradient algorithm is setup in main.py, and everything that you need to implement is in the files networkutils.py, policy.py, policygradient.py and baselinenetwork.py. The file has detailed instructions for each implementation task, but an overview of key steps in the algorithm is provided here.

<h2>1.1       REINFORCE</h2>

Recall the policy gradient theorem,

∇<em><sub>θ</sub>J</em>(<em>θ</em>) = E<em>π<sub>θ </sub></em>[∇<em><sub>θ </sub></em>log<em>π<sub>θ</sub></em>(<em>a</em>|<em>s</em>)<em>Q<sup>π</sup></em><em><sup>θ</sup></em>(<em>s,a</em>)]

REINFORCE is a Monte Carlo policy gradient algorithm, so we will be using the sampled returns <em>G<sub>t </sub></em>as unbiased estimates of <em>Q<sup>π</sup></em><em><sup>θ</sup></em>(<em>s,a</em>). The REINFORCE estimator can be expressed as the gradient of the following objective function:

where <em>D </em>is the set of all trajectories collected by policy <em>π<sub>θ</sub></em>, and ) is trajectory <em>i</em>.

<h2>1.2       Baseline</h2>

One difficulty of training with the REINFORCE algorithm is that the Monte Carlo sampled return(s) <em>G<sub>t </sub></em>can have high variance. To reduce variance, we subtract a baseline <em>b<sub>φ</sub></em>(<em>s</em>) from the estimated returns when computing the policy gradient. A good baseline is the state value function, <em>V <sup>π</sup></em><em><sup>θ</sup></em>(<em>s</em>), which requires a training

1

update to <em>φ </em>to minimize the following mean-squared error loss:

<em>L</em>MSE

<h2>1.3       Advantage Normalization</h2>

After subtracting the baseline, we get the following new objective function:

where

A second variance reduction technique is to normalize the computed advantages, <em>A</em><sup>ˆ<em>i</em></sup><em><sub>t</sub></em>, so that they have mean 0 and standard deviation 1. From a theoretical perspective, we can consider centering the advantages to be simply adjusting the advantages by a constant baseline, which does not change the policy gradient. Likewise, rescaling the advantages effectively changes the learning rate by a factor of 1<em>/σ</em>, where <em>σ </em>is the standard deviation of the empirical advantages.

<h2>1.4       Coding Questions (50 pts)</h2>

The functions that you need to implement in networkutils.py, policy.py, policygradient.py, and baselinenetwork.py are enumerated here. Detailed instructions for each function can be found in the comments in each of these files.

Note: The ”batch size” for all the arguments is <sup>P</sup><em>T<sub>i </sub></em>since we already flattened out all the episode observations, actions, and rewards for you. In networkutils.py,

<ul>

 <li>buildmlp</li>

</ul>

In policy.py,

<ul>

 <li>act</li>

 <li>actiondistribution</li>

 <li>init</li>

 <li>std</li>

 <li>actiondistribution</li>

</ul>

In policygradient.py,

<ul>

 <li>initpolicy</li>

 <li>getreturns</li>

 <li>normalizeadvantage</li>

 <li>updatepolicy</li>

</ul>

In baselinenetwork.py,

<ul>

 <li>init</li>

 <li>forward</li>

 <li>calculateadvantage</li>

 <li>updatebaseline</li>

</ul>

<h2>1.5       Testing</h2>

We have provided some basic tests to sanity check your implementation. <strong>Please note that the tests are not comprehensive, and passing them does not guarantee a correct implementation</strong>. Use the following command to run the tests:

You can also add additional tests of your own design in tests/testbasic.py.

<h2>1.6       Writeup Questions (15 pts)</h2>

<ul>

 <li>(3 pts) To compute the REINFORCE estimator, you will need to calculate the values (we drop the trajectory index <em>i </em>for simplicity), where</li>

</ul>

Naively, computing all these values takes <em>O</em>(<em>T</em><sup>2</sup>) time. Describe how to compute them in <em>O</em>(<em>T</em>) time.

<ul>

 <li>(12 pts) The general form for running your policy gradient implementation is as follows:</li>

</ul>

if not using a baseline, or

if using a baseline. Here ENV should be cartpole, pendulum, or cheetah, and SEED should be a positive integer.

For each of the 3 environments, choose 3 random seeds and run the algorithm both without baseline and with baseline. Then plot the results using

where SEEDS should be a comma-separated list of seeds which you want to plot (e.g. –seeds 1,2,3). <strong>Please include the plots (one for each environment) in your writeup, and comment on whether or not you observe improved performance when using a baseline.</strong>

We have the following expectations about performance to receive full credit:

<ul>

 <li>cartpole: Should reach the max reward of 200 (although it may not stay there)</li>

 <li>pendulum: Should reach the max reward of 1000 (although it may not stay there)</li>

 <li>cheetah: Should reach at least 200 (Could be as large as 950)</li>

</ul>

<h1>2        Reducing Variance in Policy Gradient Methods (35 pts)</h1>

In class, we explored REINFORCE as a policy gradient method with no bias but high variance. In this problem, we will explore methods to dramatically reduce variance in policy gradient methods, potentially at the cost of increased bias.

Let us consider an infinite horizon MDP M = hS<em>,</em>A<em>,</em>R<em>,</em>T <em>,γ</em>i. Let us define

<table width="624">

 <tbody>

  <tr>

   <td width="603"><em>A<sup>π</sup></em>(<em>s<sub>t</sub>,a<sub>t</sub></em>) = <em>Q<sup>π</sup></em>(<em>s<sub>t</sub>,a<sub>t</sub></em>) − <em>V <sup>π</sup></em>(<em>s<sub>t</sub></em>)An approximation to the policy gradient is defined as</td>

   <td width="21">(1)</td>

  </tr>

  <tr>

   <td width="603">∞<em>g </em>= E<em>s</em><sub>0:∞</sub>[<sup>X</sup><em>A<sup>π</sup></em>(<em>s<sub>t</sub>,a<sub>t</sub></em>)∇<em><sub>θ </sub></em>log <em>π<sub>θ</sub></em>(<em>a<sub>t</sub>,s<sub>t</sub></em>)]<em>a</em>0:∞ <em>t</em>=0where the colon notation <em>a </em>: <em>b </em>represents the range [<em>a,a </em>+ 1<em>,a </em>+ 2<em>,…b</em>] inclusive of both ends.</td>

   <td width="21">(2)</td>

  </tr>

 </tbody>

</table>

<ul>

 <li>(5 pts) Let us define the partial sum. Show that it is not necessarily true that Var(<em>R<sub>t</sub></em><sub>+1</sub>) ≥ Var(<em>R<sub>t</sub></em>). [Hint: Construct a counterexample MDP where this statement does not hold.]</li>

 <li>(10 pts) Prove that Var(<em>R<sub>t</sub></em><sub>+1</sub>) ≥ Var(<em>R<sub>t</sub></em>) is true if we assume that <em>r<sub>t</sub></em><sub>+1 </sub>is, on average, correlated with the previous rewards, i.e. Cov(<em>r<sub>i</sub>,r<sub>t</sub></em><sub>+1</sub>) <em>&gt; </em></li>

 <li>(5 pts) In practice, we do not have access to the true function <em>A<sup>π</sup></em>(<em>s<sub>t</sub>,a<sub>t</sub></em>), so we would like to obtain an estimate instead. We will consider the general form of an estimator <em>A</em><sup>ˆ</sup><em><sub>t</sub></em>(<em>s</em><sub>0:∞</sub><em>,a</em><sub>0:∞</sub>) that can be a function of the entire trajectory.</li>

</ul>

Let <em>A</em><sup>ˆ</sup><em><sub>t</sub></em>(<em>s</em><sub>0:∞</sub><em>,a</em><sub>0:∞</sub>) = <em>Q</em><sup>ˆ</sup><em><sub>t</sub></em>(<em>s<sub>t</sub></em><sub>:∞</sub><em>,a<sub>t</sub></em><sub>:∞</sub>)−<em>b<sub>t</sub></em>(<em>s</em><sub>0:<em>t</em></sub><em>,a</em><sub>0:<em>t</em>−1</sub>), where for all <em>s<sub>t</sub>,a<sub>t</sub></em>, we have that <em>Q</em><sup>ˆ</sup><em><sub>t </sub></em>is an unbiased estimator of the true <em>Q<sup>π</sup></em>. Namely, we have that ). Note that <em>b<sub>t </sub></em>is an arbitrary function of the actions and states sampled before <em>a<sub>t</sub></em>. Prove that by using this estimate of <em>A</em>ˆ<em><sub>t</sub></em>, we obtain an unbiased estimate of the policy gradient <em>g</em>. In other words, prove that

.

<ul>

 <li>(5 pts) We will now look at a few different variants of <em>A</em><sup>ˆ</sup><em><sub>t</sub></em>. Recall the TD error <em>δ<sub>t</sub><sup>V</sup></em><sup>ˆ </sup>(<em>s<sub>t</sub>,a<sub>t</sub></em>) = <em>r<sub>t</sub></em>+<em>γV</em><sup>ˆ</sup>(<em>s<sub>t</sub></em><sub>+1</sub>)−<em>V</em><sup>ˆ</sup>(<em>s<sub>t</sub></em>). If <em>V</em><sup>ˆ </sup>= <em>V <sup>π</sup></em>, prove that is an unbiased estimate of <em>A<sup>π</sup></em>.</li>

 <li>(5 pts) Let us define. Show that. In general,</li>

</ul>

how does bias and variance change as <em>k </em>increases? (a few sentences of justification would suffice, no formal proof is necessary)

<ul>

 <li>(5 pts) Show that</li>

</ul>