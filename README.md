Download Link: https://assignmentchef.com/product/solved-cs285-hw4-model-based-rl
<br>
The goal of this assignment is to get experience with model-based reinforcement learning. In general, model-based reinforcement learning consists of two main parts: learning a dynamics function to model observed state transitions, and then using predictions from that model in some way to decide what to do (e.g., use model predictions to learn a policy, or use model predictions directly in an optimization setup to maximize predicted rewards).

In this assignment, you will do the latter. You will implement both the process of learning a dynamics model, as well as the process of creating a controller to perform action selection through the use of these model predictions. For references to this type of approach, see this <a href="https://arxiv.org/pdf/1708.02596.pdf">paper</a> and this <a href="https://arxiv.org/pdf/1909.11652.pdf">paper</a><a href="https://arxiv.org/pdf/1909.11652.pdf">.</a>

<h1>1           Model-Based Reinforcement Learning</h1>

We will now provide a brief overview of model-based reinforcement learning (MBRL), and the specific type of MBRL you will be implementing in this homework. Please see <a href="http://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-11.pdf">Lecture 11: Model-Based Reinforcement Learning</a> (with specific emphasis on the slides near page 9) for additional details.

MBRL consists primarily of two aspects: (1) learning a dynamics model and (2) using the learned dynamics models to plan and execute actions that minimize a cost function (or maximize a reward function).

<h2>1.1         Dynamics Model</h2>

In this assignment, you will learn a neural network dynamics model <em>f<sub>θ </sub></em>of the form

∆ˆ <em>t</em>+1 = <em>f</em><em>θ</em>(<strong>s</strong><em>t,</em><strong>a</strong><em>t</em>)                                                         (1)

which predicts the change in state given the current state and action. So given the prediction ∆<sup>ˆ </sup><em><sub>t</sub></em><sub>+1</sub>, you can generate the next prediction with

ˆ<strong>s</strong><em>t</em>+1 = <strong>s</strong><em>t </em>+ ∆ˆ <em>t</em>+1<em>.                                                        </em>(2)

See the previously <a href="https://arxiv.org/pdf/1708.02596.pdf">referenced paper</a> for intuition on why we might want our network to predict state differences, instead of directly predicting next state.

You will train <em>f<sub>θ </sub></em>in a standard supervised learning setup, by performing gradient descent on the following objective:

<table width="362">

 <tbody>

  <tr>

   <td width="345">L(<em>θ</em>) = X k(<strong>s</strong><em>t</em>+1 −<strong>s</strong><em>t</em>) − <em>f</em><em>θ</em>(<strong>s</strong><em>t,</em><strong>a</strong><em>t</em>)k22(<strong>s</strong><em>t</em><em>,</em><strong>a</strong><em>t</em><em>,</em><strong>s</strong><em>t</em>+1)∈D</td>

   <td width="17">(3)</td>

  </tr>

  <tr>

   <td width="345">= X</td>

   <td width="17">(4)</td>

  </tr>

 </tbody>

</table>

(<strong>s</strong><em>t</em><em>,</em><strong>a</strong><em>t</em><em>,</em><strong>s</strong><em>t</em>+1)∈D

In practice, it’s helpful to normalize the target of a neural network. So in the code, we’ll train the network to predict a <em>normalized </em>version of the change in state, as in

L(<em>θ</em>) = <sup>X </sup>kNormalize(<strong>s</strong><em>.                            </em>(5)

(<strong>s</strong><em>t</em><em>,</em><strong>a</strong><em>t</em><em>,</em><strong>s</strong><em>t</em>+1)∈D

Since <em>f<sub>θ </sub></em>is trained to predict the normalized state difference, you generate the next prediction with

ˆ<strong>s</strong><em><sub>t</sub></em><sub>+1 </sub>= <strong>s</strong><em><sub>t </sub></em>+ Unnormalize(<em>f<sub>θ</sub></em>(<strong>s</strong><em><sub>t</sub>,</em><strong>a</strong><em><sub>t</sub></em>))<em>.                                        </em>(6)

<h2>1.2         Action Selection</h2>

Given the learned dynamics model, we now want to select and execute actions that minimize a known cost function (or maximize a known reward function). Ideally, you would calculate these actions by solving the following optimization:

∞

<strong>a</strong><sup>∗</sup><em><sub>t </sub></em>= argmin<sup>X</sup><em>c</em>(ˆ<strong>s</strong><em><sub>t</sub></em>0<em>,</em><strong>a</strong><em><sub>t</sub></em>0) where ˆ<strong>s</strong><em><sub>t</sub></em>0<sub>+1 </sub>= ˆ<strong>s</strong><em><sub>t</sub></em>0 + <em>f<sub>θ</sub></em>(ˆ<strong>s</strong><em><sub>t</sub></em>0<em>,</em><strong>a</strong><em><sub>t</sub></em>0)<em>.                       </em>(7)

<strong>a</strong><em>t</em>:∞ <em>t</em><sup>0</sup>=<em>t</em>

However, solving Eqn. 7 is impractical for two reasons: (1) planning over an infinite sequence of actions is impossible and (2) the learned dynamics model is imperfect, so using it to plan in such an open-loop manner will lead to accumulating errors over time and planning far into the future will become very inaccurate.

Instead, we will solve the following gradient-free optimization problem:

<em>t</em>+<em>H</em>−1

<strong>A</strong><sup>∗ </sup>= arg                           min <sup>X </sup><em>c</em>(ˆ<strong>s</strong><em><sub>t</sub></em>0<em>,</em><strong>a</strong><em><sub>t</sub></em>0) s.t. ˆ<strong>s</strong><em><sub>t</sub></em>0<sub>+1 </sub>= ˆ<strong>s</strong><em><sub>t</sub></em>0 + <em>f<sub>θ</sub></em>(ˆ<strong>s</strong><em><sub>t</sub></em>0<em>,</em><strong>a</strong><em><sub>t</sub></em>0)<em>,                     </em>(8)

{<strong>A</strong>(0)<em>,…,</em><strong>A</strong>(<em>K</em>−1)} <em>t</em>0=<em>t</em>

in which <strong>A</strong>) are each a random action sequence of length

<ol>

 <li><em>H</em>. What Eqn. 8 says is to consider <em>K </em>random action sequences of length <em>H</em>, predict the result (i.e., future states) of taking each of these action sequences using the learned dynamics model <em>f<sub>θ</sub></em>, evaluate the cost/reward associated with each candidate action sequence, and select the best action sequence. Note that this approach only plans <em>H </em>steps into the future, which is desirable because it prevent accumulating model error, but is also limited because it may not be sufficient for solving long-horizon tasks.</li>

</ol>

Additionally, since our model is imperfect and things will never go perfectly according to plan, we adopt a model predictive control (MPC) approach, in which we solve Eqn. 8 at every time step to select the best <em>H</em>-step action sequence, but then we execute only the first action from that sequence before replanning again at the next time step using updated state information.

Finally, note that the random-shooting optimization approach mentioned above can be greatly improved (see this <a href="https://arxiv.org/pdf/1909.11652.pdf">paper</a><a href="https://arxiv.org/pdf/1909.11652.pdf">)</a>.

<h2>1.3         On-Policy Data Collection</h2>

Although MBRL is in theory off-policy—meaning it can learn from any data—in practice it will perform poorly if you don’t have on-policy data. In other words, if a model is trained on only randomly-collected data, it will (in most cases) be insufficient to describe the parts of the state space that we may actually care about. We can therefore use on-policy data collection in an iterative algorithm to improve overall task performance. This is summarized as follows:

<h2>1.4         Ensembles</h2>

A simple and effective way to improve predictions is to use an ensemble of models. The idea is simple: rather than training one network <em>f<sub>θ </sub></em>to make predictions, we’ll train <em>N </em>independently initialized networks, and average their predictions to get your final predictions

<em>.                                            </em>(9)

In this assignment, you’ll train an ensemble of networks and compare how different values of <em>N </em>effect the model’s performance.

<h1>2           Code</h1>

You will implement the MBRL algorithm described in the previous section.

<h2>2.1         Overview</h2>

Obtain the code from <a href="https://github.com/berkeleydeeprlcourse/homework_fall2020/tree/master/hw4">https://github.com/berkeleydeeprlcourse/ </a><a href="https://github.com/berkeleydeeprlcourse/homework_fall2020/tree/master/hw4">homework_fall2020/tree/master/hw4</a><a href="https://github.com/berkeleydeeprlcourse/homework_fall2020/tree/master/hw4">.</a>

You will add code to the following three files: agents/mb_agent.py, models/ff_model.py, and policies/MPC_policy.py. You will also need to edit these files by copying code from past homeworks or Piazza: infrastructure/rl_trainer.py and infrastructure/utils.py.

<h1>Problem 1</h1>

What you will implement:

Collect a large dataset by executing random actions. Train a neural network dynamics model on this fixed dataset and visualize the resulting predictions. The implementation that you will do here will be for training the dynamics model, and comparing its predictions against ground truth. You will be reusing the utilities you wrote for HW1 (or Piazza) for the data collection part (look for “get this from Piazza” markers).

<u>What code files to fill in:</u>

<ol>

 <li>cs285/agents/mb_agent.py</li>

 <li>cs285/models/ff_model.py</li>

 <li>cs285/infrastructure/utils.py</li>

 <li>cs285/policies/MPC_policy.py (just one line labeled TODO(Q1) for now)</li>

</ol>

<u>What commands to run:</u>

<table width="458">

 <tbody>

  <tr>

   <td width="458">python cs285/scripts/run_hw4_mb.py –exp_name q1_cheetah_n500_arch1x32–env_name cheetah-cs285-v0 –add_sl_noise –n_iter 1 -batch_size_initial 20000 –num_agent_train_steps_per_iter 500 -n_layers 1 –size 32 –scalar_log_freq -1 –video_log_freq -1python cs285/scripts/run_hw4_mb.py –exp_name q1_cheetah_n5_arch2x250–env_name cheetah-cs285-v0 –add_sl_noise –n_iter 1 -batch_size_initial 20000 –num_agent_train_steps_per_iter 5 -n_layers 2 –size 250 –scalar_log_freq -1 –video_log_freq -1python cs285/scripts/run_hw4_mb.py –exp_name q1_cheetah_n500_arch2x250–env_name cheetah-cs285-v0 –add_sl_noise –n_iter 1 -batch_size_initial 20000 –num_agent_train_steps_per_iter 500 -n_layers 2 –size 250 –scalar_log_freq -1 –video_log_freq -1</td>

  </tr>

 </tbody>

</table>

Your code will produce plots inside your logdir that illustrate your model prediction error (MPE). The code will also produce a plot of the losses over time. For the first command, the loss should go below 0.2 by the iteration 500. These plots illustrate, for a fixed action sequence, the difference between your model’s predictions (red) and the ground-truth states (green). Each plot corresponds to a different state element, and the title reports the mean mean-squared-error across all state elements. As illustrated in the commands above, try different neural network architectures as well different amounts of training. Compare the results by looking at the loss values (i.e., itr 0 losses.png), the qualitative model predictions (i.e., itr 0 predictions.png), as well as the quantitative MPE values (i.e., in the title of itr 0 predictions.png).

<u>What to submit: </u>For this question, submit the qualitative model predictions (itr 0 predictions.png) for each of the three runs above. Comment on which model performs the best and why you think this might be the case.

Note that for these qualitative model prediction plots, we intend for you to just copy the png images produced by the code.




<h1>Problem 2</h1>

What will you implement:

Action selection using your learned dynamics model and a given reward function.

<u>What code files to fill in:</u>

<ol>

 <li>cs285/policies/MPC_policy.py <u>What commands to run:</u></li>

</ol>

python cs285/scripts/run_hw4_mb.py –exp_name q2_obstacles_singleiteration –env_name obstacles-cs285-v0 -add_sl_noise –num_agent_train_steps_per_iter 20 –n_iter 1 -batch_size_initial 5000 –batch_size 1000 –mpc_horizon 10

Recall the overall flow of our rl trainer.py. We first collect data with our policy (which starts as random), we then train our model on that collected data, and we then evaluate the resulting MPC policy (which now uses the trained model). To verify that your MPC is indeed doing reasonable action selection, run the command above and compare Train AverageReturn (which was the execution of random actions) to Eval AverageReturn (which was the execution of MPC using a model that was trained on the randomly collected training data). You can expect Train AverageReturn to be around -160 and Eval AverageReturn to be around -70 to -50.

<u>What to submit:</u>

Submit this run as part of your run logs, and include a plot of Train AverageReturn and Eval AverageReturn in your pdf. Note that these will just be single dots on the plot, since we ran this for just 1 iteration.

<h1>Problem 3</h1>

What will you implement:

MBRL algorithm with on-policy data collection and iterative model training.

<u>What code files to fill in:</u>

None. You should already have done everything that you need, because rl trainer.py already aggregates your collected data into a replay buffer. Thus, iterative training means to just train on our growing replay buffer while collecting new data at each iteration using the most newly trained model.

<u>What commands to run:</u>

python cs285/scripts/run_hw4_mb.py –exp_name q3_obstacles –env_name obstacles-cs285-v0 –add_sl_noise –num_agent_train_steps_per_iter 20 –batch_size_initial 5000 –batch_size 1000 –mpc_horizon 10 -n_iter 12

python cs285/scripts/run_hw4_mb.py –exp_name q3_reacher –env_name reacher-cs285-v0 –add_sl_noise –mpc_horizon 10 -num_agent_train_steps_per_iter 1000 –batch_size_initial 5000 -batch_size 5000 –n_iter 15

python cs285/scripts/run_hw4_mb.py –exp_name q3_cheetah –env_name cheetah-cs285-v0 –mpc_horizon 15 –add_sl_noise -num_agent_train_steps_per_iter 1500 –batch_size_initial 5000 -batch_size 5000 –n_iter 20

You should expect rewards of around -25 to -20 for the obstacles env (takes 40 minutes), rewards of around -250 to -300 for the reacher env (takes 2-3 hours), and rewards of around 250-350 for the cheetah env takes 3-4 hours. All numbers assume no GPU.

<u>What to submit:</u>

Submit these runs as part of your run logs, and include the performance plots in your pdf.

<h1>Problem 4</h1>

What will you implement:

You will compare the performance of your MBRL algorithm as a function of three hyperparameters: the number of models in your ensemble, the number of random action sequences considered during each action selection, and the MPC planning horizon.

<u>What code files to fill in: </u>None.

<u>What commands to run:</u>

<table width="458">

 <tbody>

  <tr>

   <td width="458">python cs285/scripts/run_hw4_mb.py –exp_name q4_reacher_horizon5 -env_name reacher-cs285-v0 –add_sl_noise –mpc_horizon 5 -num_agent_train_steps_per_iter 1000 –batch_size 800 –n_iter 15python cs285/scripts/run_hw4_mb.py –exp_name q4_reacher_horizon15 -env_name reacher-cs285-v0 –add_sl_noise –mpc_horizon 15 -num_agent_train_steps_per_iter 1000 –batch_size 800 –n_iter 15python cs285/scripts/run_hw4_mb.py –exp_name q4_reacher_horizon30 -env_name reacher-cs285-v0 –add_sl_noise –mpc_horizon 30 -num_agent_train_steps_per_iter 1000 –batch_size 800 –n_iter 15python cs285/scripts/run_hw4_mb.py –exp_name q4_reacher_numseq100 -env_name reacher-cs285-v0 –add_sl_noise –mpc_horizon 10 -num_agent_train_steps_per_iter 1000 –batch_size 800 –n_iter 15 -mpc_num_action_sequences 100python cs285/scripts/run_hw4_mb.py –exp_name q4_reacher_numseq1000 -env_name reacher-cs285-v0 –add_sl_noise –mpc_horizon 10 -num_agent_train_steps_per_iter 1000 –batch_size 800 –n_iter 15 -mpc_num_action_sequences 1000python cs285/scripts/run_hw4_mb.py –exp_name q4_reacher_ensemble1 -env_name reacher-cs285-v0 –ensemble_size 1 –add_sl_noise -mpc_horizon 10 –num_agent_train_steps_per_iter 1000 –batch_size800 –n_iter 15 python cs285/scripts/run_hw4_mb.py –exp_name q4_reacher_ensemble3 -env_name reacher-cs285-v0 –ensemble_size 3 –add_sl_noise -mpc_horizon 10 –num_agent_train_steps_per_iter 1000 –batch_size800 –n_iter 15 python cs285/scripts/run_hw4_mb.py –exp_name q4_reacher_ensemble5 -env_name reacher-cs285-v0 –ensemble_size 5 –add_sl_noise -mpc_horizon 10 –num_agent_train_steps_per_iter 1000 –batch_size800 –n_iter 15</td>

  </tr>

 </tbody>

</table>