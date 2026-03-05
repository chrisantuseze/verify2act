## Reflective Planning: Vision-Language Models for

## Multi-Stage Long-Horizon Robotic Manipulation

```
Yunhai Feng^1 Jiaming Han^2 Zhuoran Yang^3 Xiangyu Yue^2 Sergey Levine^4 Jianlan Luo^4 †
```
#### Abstract

```
Solving complex long-horizon robotic manipula-
tion problems requires sophisticated high-level
planning capabilities, the ability to reason about
the physical world, and reactively choose ap-
propriate motor skills. Vision-language models
(VLMs) pretrained on Internet data could in prin-
ciple offer a framework for tackling such prob-
lems. However, in their current form, VLMs
lack both the nuanced understanding of intricate
physics required for robotic manipulation and
the ability to reason over long horizons to ad-
dress error compounding issues. In this paper, we
introduce a novel test-time computation frame-
work that enhances VLMs’ physical reasoning
capabilities for multi-stage manipulation tasks.
At its core, our approach iteratively improves
a pretrained VLM with a “reflection” mecha-
nism - it uses a generative model to imagine
future world states, leverages these predictions
to guide action selection, and critically reflects
on potential suboptimalities to refine its reason-
ing. Experimental results demonstrate that our
method significantly outperforms several state-
of-the-art commercial VLMs as well as other
post-training approaches such as Monte Carlo
Tree Search (MCTS). Videos are available at
https://reflect-vlm.github.io.
```
#### 1. Introduction

Complex multi-stage manipulation tasks remain a funda-
mental challenge in robotics (Luo et al., 2024a; Kroemer
et al., 2020; Cui & Trinkle, 2021), particularly when they
require reasoning about sophisticated physical interactions
and their consequences over long time horizons. These tasks
often involve intricate sequences of actions where each step

†Project Advisor (^1) Cornell University (^2) The Chinese University
of Hong Kong^3 Yale University^4 University of California, Berke-
ley. Correspondence to: Yunhai Feng<yunhaif@cs.cornell.edu>,
Xiangyu Yue<xyyue@ie.cuhk.edu.hk>, Jianlan Luo<jianlan-
luo@eecs.berkeley.edu>.
Copyright 2025 by the author(s).
Current image Goal image
Pick up purple.
What action should be taken next?
Pick up purple Insert purple
**...
Given the imagined future, reflect and revise original action if necessary.
Pick up yellow.**
Diffusion Model
Figure 1.Reflective planning.Our method uses a VLM to propose
actions and a diffusion dynamics model to imagine the future state
of executing the plan. The imagined future helps the VLM reflect
the initial plan and propose better action.
must account for physical constraints and potential conse-
quences, making them particularly challenging for planning
systems. Success requires not only understanding the imme-
diate effects of actions but also their long-term implications,
the ability to adapt plans based on execution outcomes, and
generalizing to novel scenarios.
While classical planning approaches, such as task and mo-
tion planning (TAMP) (Kaelbling & Lozano-Perez, 2011; ́
Garrett et al., 2020a), can in principle address such prob-
lems, their reliance on predefined symbolic representations
and explicit state estimation makes them difficult to apply
in settings without known models that require visual per-
ception (Driess et al., 2020; Wang et al., 2021). This limita-
tion has motivated the search for more flexible approaches
to robotic planning. Recent advances in vision-language
models (VLMs) have shown remarkable capabilities in pro-
cessing visual scenes and natural language instructions by
leveraging internet-scale knowledge (Chen et al., 2023; Bai
et al., 2023; OpenAI, 2024a; Google, 2024; Liu et al., 2023).
These models can effectively parse complex visual envi-
ronments and comprehend high-level task descriptions ex-


pressed in natural language, making them promising can-
didates for robotic planning problems (Driess et al., 2023;
Brohan et al., 2023b;a; Shi et al., 2024; Liu et al., 2024a),.
However, state-of-the-art VLMs still struggle with complex
physical reasoning tasks, and this limitation becomes par-
ticularly pronounced when precise physics concepts and
long-horizon planning are involved (Gao et al., 2024; Chen
et al., 2024).

In this paper, we study how to effectively leverage VLMs’
Internet-scale knowledge while addressing their limitations
in physical reasoning and long-horizon planning. We focus
on a challenging class of robotic manipulation problems
that involve sequentially manipulating interlocking objects
to achieve desired configurations, as illustrated in Fig. 5.
These tasks are particularly difficult as they require precise
understanding of physical constraints, careful reasoning
about action sequences, and the ability to plan over extended
horizons while maintaining physical feasibility at each step.

To address these challenges, we present a novel test-time
computation framework that significantly enhances VLMs’
capabilities for multi-stage robotic manipulation tasks. The
key insight of our method, ReflectVLM, is that by com-
bining VLMs with a reflection mechanism and targeted
post-training, we can create a system that better understands
physical constraints and their implications for action plan-
ning. We use the term “reflection” to refer to a process where
a VLM iteratively refines its decisions by critically exam-
ining the predicted outcomes of its proposed actions, akin
to self-critique methods in large language models (Huang
et al., 2024; Wang et al., 2023; Madaan et al., 2024). Our
approach introduces two key components: (1) a look-ahead
mechanism that uses a diffusion-based dynamics model to
generate visual predictions of future states resulting from
planned actions, and (2)a reflection process that allows the
VLM to critique and refine its planned actions by analyzing
these predicted outcomes. This combination of visual pre-
diction and iterative refinement allows the VLM to develop
a more sophisticated understanding of physical constraints
and improve its decision-making capabilities without requir-
ing extensive retraining.

Experimental results demonstrate that our approach signifi-
cantly outperforms both the latest commercial state-of-the-
art VLM models and traditional planning approaches like
Monte Carlo Tree Search (MCTS) on this class of prob-
lems. Notably, our method achieves superior performance
compared to post-training techniques such as supervised
fine-tuning (SFT) while using the same amount of labeled
data and maintaining computational efficiency. The suc-
cess of our approach suggests that enhancing VLMs with
structured reasoning mechanisms at test time can be a pow-
erful strategy for improving their performance on physically-
grounded tasks.

Our primary contribution is the mentioned test-time compu-

```
tation framework that enhances VLMs’ physical reasoning
capabilities for multi-stage manipulation tasks. Through
extensive experiments, we demonstrate that our approach
not only outperforms existing methods but also maintains
computational efficiency. Importantly, while we demon-
strate our framework’s effectiveness on manipulation tasks,
it is designed to be general and can be readily extended to
other domains requiring visual understanding and sequential
decision-making. This generality suggests broader applica-
tions in robotics and autonomous systems where physical
reasoning and long-horizon planning are essential.
```
#### 2. Related Work

```
Our framework incorporates a VLM with the reflection
mechanism to solve long-horizon robotic planning problems.
We therefore survey reflection techniques in the broader
context in large models, VLM for robotic planning, as well
as existing techniques for solving robot task and motion
planning.
```
```
2.1. Reflection
Recent work has shown that large language models can
benefit from reflection mechanisms - processes where mod-
els iteratively refine their outputs through self-critique and
revision (Renze & Guven, 2024; Shinn et al., 2024; Pan
et al., 2023; Madaan et al., 2024; Asai et al., 2023; Wang
et al., 2023; Huang et al., 2024). For example, Madaan et al.
(2024) introduced an iterative refinement approach where
models critique and improve their own outputs through self-
feedback. Chain-of-thought prompting and its variants (Wei
et al., 2022; Wang et al., 2022; Yao et al., 2024) demon-
strated that guiding models to show their reasoning process
leads to better performance. Similarly, Cheng et al. (2024);
Yu et al. (2025) extended such reflection mechanisms to
vision-language models.
However, these approaches focus primarily on language-
only or visual comprehension tasks, without addressing
physical reasoning or robotics applications. Our work ex-
tends reflection to long-horizon robotic planning by incor-
porating a diffusion model that generates imagined future
visual states. This allows the VLM to reflect on and revise
its plans based on concrete visual predictions rather than
relying solely on symbolic reasoning.
```
```
2.2. VLM for Robotic Planning
In robotics, several recent works have explored using VLMs
for planning (Driess et al., 2023; Brohan et al., 2023b;a;
Hu et al., 2023; Huang et al., 2023; Belkhale et al., 2024;
Nasiriany et al., 2024; Liu et al., 2024a; Shi et al., 2024;
Wake et al., 2024). However, these approaches either rely on
symbolic state representations or make decisions in a single-
step manner based only on current observations, without
explicitly reasoning about future consequences or utilizing
```

```
Ig It
```
```
at at +
```
```
It +
```
```
...
```
```
It + H
```
```
... ...
```
```
It +
```
```
Q1 : Goal configuration is. Current configuration is.
History actions are at -h , ..., at - 1.
What action should be taken next?
```
```
I 0
```
```
Ig
```
```
A1 : at *
```
```
Q2 : Goal configuration is. Current configuration is.
History actions are at -h , ..., at - 1.
Future configuration after at +1 , ..., at + H - 1 is.
What action should be taken next?
```
```
A2 : at *
```
```
It Ig It
It+H
```
```
“pick up yellow”
```
```
(“pick up purple”) (“pick up purple”)
```
Figure 2.Training data generation.Training data for the reflection mechanism is collected by relabeling the rollouts. For each timestep,
two training examples are generated: (Q1, A1) for action proposal and (Q2, A2) for reflection.His the imagination horizon, andhis the
history length.a∗tis the action label given by the expert policy.

reflection mechanisms.

While ReplanVLM (Mei et al., 2024b) and GameVLM (Mei
et al., 2024a) use VLMs to replan robot actions based on ex-
ecution feedback, they still rely on symbolic state represen-
tations rather than visual imagination of future states. Black
et al. (2023) utilized a diffusion model to generate fu-
ture visual states and executed them with a low-level goal-
conditioned policy, but did not leverage these predictions for
plan reflection or revision. Du et al. (2023) combines a VLM
with video prediction for beam search, but suffers from pre-
diction error accumulation and struggles with physics-based
reasoning tasks.

Our framework addresses these limitations by enabling
VLMs to imagine and evaluate potential future states
through a diffusion-based dynamics model. This allows
for sophisticated multi-step planning while maintaining the
benefits of VLMs’ pre-trained visual-language understand-
ing. The reflection mechanism further enables the VLM to
critique and refine its plans based on these imagined futures,
leading to more robust long-horizon manipulation.

2.3. Robotic Task and Motion Planning

Robotic Task and Motion Planning (TAMP) has been ex-
tensively studied (Kaelbling & Lozano-Perez, 2011; Gar- ́
rett et al., 2020a;b). Traditional approaches often combine
symbolic planning with motion planning but struggle with
real-world physical interactions and visual inputs. Learning-
based methods (Wang et al., 2021; Driess et al., 2020) show
promise in handling uncertainty and complex dynamics but
typically require significant task-specific engineering.

Our approach bridges this gap by leveraging VLMs’ broad
knowledge while adding structured physical reasoning
through visual imagination and reflection. This enables
robust long-horizon planning without requiring extensive
task-specific engineering or large amounts of training data.

#### 3. Preliminaries and Problem Statement

```
We formulate the multi-stage robotic manipulation planning
problem as a partially observable Markov decision process
(POMDP), defined by the tuple(S,A,T,O,Z). Here,Sis
the state space containing the full physical state of the envi-
ronment, including object poses and physical properties;A
is the action space consisting of high-level manipulation
primitives {pick up,insert,reorient,put down} ×
{objects}, assuming a failure rateεfor each primitive;
T(st+1|st,at)represents the transition dynamics capturing
physical interactions;Ois the observation space of RGB
images; andZ(ot|st)is the observation model mapping
states to images.
Given a goal statesg, the objective is to find a policyπthat
generates a sequence of actions to reachsg. Due to partial
observability, the policy only has access to image observa-
tions, taking the formπ(at|It,Ig)whereItis the current
observation andIgis the goal image. The policy is instan-
tiated as a VLM agentπVLM, which takes a multi-modal
input of images and text, and generates action primitives in
the form of text.
Our framework includes a pre-training phase and a post-
training phase. The post-training phase builds on the frame-
work of interactive imitation learning (Ross et al., 2011;
Kelly et al., 2018), which learns a policy by interacting with
environment and receiving expert supervision in real-time.
Thus under the standard assumption, we assume access to
an interactive expert policyπEthat generates near-optimal
actionsa∗=πE(s)for any statesat training time. In this
paper, we instantiated such an expert policy with access to
the full state of the environment to generate optimal actions,
though it could be obtained via other formats as well, e.g.,
human demonstrations. However, the VLM policy will only
have access to image observations.
```

#### 4. Reflective Planning with Vision Language

#### Models

```
To address the challenges of physical interaction and long-
horizon reasoning, we present a framework that incorporates
VLMs with reflective planning. Our approach combines
two key components: (1) a diffusion-based dynamics model
that enables the VLM to imagine and evaluate future states,
and (2) an interactive learning mechanism that allows the
VLM to reflect on and revise its decisions based on these
imagined outcomes. As shown in Fig. 1, these components
work together to enable more robust manipulation planning
while preserving the benefits of pre-trained VLMs.
```
4.1. Interactive VLM Policy Post-Training
While VLMs can generate actions based on visual inputs,
they may hallucinate physically implausible solutions with-
out actual interaction experience. To overcome this lim-
itation and enable long-horizon reasoning, we introduce
an interactive learning algorithm that teaches the VLM to
reflect on and improve its decisions through direct interac-
tion with the physical environment. This process further
enhances a base VLM policy, which is initially trained on a
fixed set of expert demonstrations. Similar to DAgger (Ross
et al., 2011), we iteratively collect new data by rolling out
the VLM policy in the environment and finetune the VLM
policy with the aggregated data. As formulated in Algo-
rithm 1,Ntrajectories are collected in each iteration. At
each timestep, we generate a learner actiona†tby prompting
the VLM with the images of the goal and current states, as
well as an expert actiona∗tfrom the oracle policy. The pairs
((Ig,It),a∗t)are then added to the dataset for finetuning. To

```
facilitate convergence, we execute the learner actiona†twith
a probability ofpand the expert actiona∗twith a probability
of 1 −p, instead of always following the actions from the
learner.
To generate training data for reflection, we can simply rela-
bel a trajectory after it is terminated, as also illustrated in
Fig. 2. Specifically, the imageIt+H, which is a future obser-
vation following the action sequenceat:t+H− 1 , is added to
the context for reflection at timestept, and the VLM is still
supervised to output the same expert actiona∗t. Intuitively,
this image provides additional information about the effect
of executing the action sequence as a feedback, which can
be leveraged by the VLM to decide whether the initially
proposed action sequence leads to a promising future state.
In essence, we are generating two forms of question an-
swering examples from interaction with the environment.
The first is to predict an optimal action given images of
the goal and current state, and the second is to reflect and
revise an initial action sequence proposal by looking into
an additional future image. Since a VLM can flexibly take
any text and images as input, these two tasks can be handled
by a single VLM with two different prompt templates, as
```
```
Algorithm 1Interactive VLM Post-Training
Require: initial state distributionρ 0 , goal state distribution
ρg, numbef of iterationsK, number of trajectories per
iterationN, episode lengthT, imagination horizonH,
expert policyπE, expert demonstrationsD∗
1:train base policyπVLMonD∗
2:D ←D∗
3:fori← 1 toKdo
4: Di←∅
5: // rollout out policyπVLMto collect dataDi
6: forn← 1 toNdo
7: s 0 ∼ρ 0 ;I 0 ←Z(s 0 )
8: sg∼ρg;Ig←Z(sg)
9: fort← 0 toT− 1 do
10: a†t∼πVLM(Ig,It);a∗t∼πE(sg,st)
11: at←a†tifrandom()< pelsea∗t
12: st+1←T(st,at);It+1←Z(st+1)
13: end for
14: Di←Di∪{((Ig,It),a∗t)} 0 ≤t<T
15: Di←Di∪{((Ig,It,It+H,at:t+H− 1 ),a∗t)} 0 ≤t<T
16: end for
17: D ←D∪Di
18: finetuneπVLMonD
19:end for
```
```
summarized in Fig. 2. See App. E for full prompts, and
App. D.1 for detailed VLM architecture.
The VLM is trained to generate actions aligned with expert
actions in the dataset with a cross entropy loss:
```
```
min
πVLM
```
###### ED

```
h
LCE(πproposeVLM (at|Ig,It),a∗t)
```
```
+LCE(πreflectVLM(at|Ig,It,It+H,at:t+H− 1 ),a∗t)
```
```
i
.
```
###### (1)

```
4.2. Diffusion Dynamics Model
A key component in reflective planning is predicting future
states accurately when evaluating potential action sequences.
While our interactive learning mechanism enables the VLM
to learn from physical interactions, we need an additional
capability during inference - the ability to imagine and eval-
uate hypothetical futures without actually executing actions
in the environment. To address this, we develop a diffusion-
based dynamics model (DDM) that efficiently generates
predicted visual observations by conditioning on the current
observation and a proposed action sequence. This allows
the VLM to simulate the consequences of its actions before
committing to them.
Building on advances in diffusion-based generative mod-
els (Rombach et al., 2021; Ho et al., 2020; Song et al., 2021),
we formulate the forward dynamics prediction as an image-
to-image translation task. Our diffusion dynamics model
takes the current observationItand actionatas input to
```

```
“pick up purple”
```
```
frozen trainable
```
```
Latent
Encoder Text
Encoder
```
```
Diffusion
UNet
```
```
concat.
```
```
It It +
```
#### N zt zt+ 1

```
at
```
### +

```
Latent
Decoder
```
Figure 3.Architecture of Diffusion Dynamics Model,which con-
sists of a latent encoder, text encoder, Diffusion UNet and latent
decoder. The latent encoder and text encoder are frozen during
training, while Diffusion UNet and latent decoder are finetuned on
our task data.N: random noise.

predict the next observationIt+1. Rather than training a dif-
fusion model from scratch, which would require substantial
computational resources and training data, we leverage the
pretrained Instructpix2pix model (Brooks et al., 2022) that
has been trained on large-scale image editing datasets as our
base model.

Data.We curate a dataset for training the diffusion model.
To encourage broader coverage of visisted states, the data
collection policy is a noised version of the oracle policy.
Due to the difficulty of this task, we also include a few test
data points to improve the fidelity and accuracy of the DDM.
Details can be found in App. D.2.

Architecture.The model architecture is shown in Fig. 3.
For the input(It,at), we first encode them into latent repre-
sentationztandzatwith pretrained latent encoder and text
encoder. Then we feedzt, a sampled noiseNand the action
conditionzatinto the diffusion UNet for de-noising. Finally,
we decode the predictedzt+1into a future observationIt+
with a latent decoder.

Training.The training of DDM consists of two separate
phases: UNet training and decoder training. The UNet train-
ing phase is to learn transformations fromzttozt+1condi-
tioned onzat, while the latent decoder training is to adapt
the pretrained VAE models into our task domain because
our task requires precise reconstruction of small pieces on
the table. Since we keep the latent encoder frozen, we can
train the two phases in parallel.

4.3. Reflective Planning

With the VLM policy trained via interactive learning and
the diffusion model serving as a dynamics proxy to imagine
future outcomes, we now introduce our reflective planning

```
mechanism for decision making at inference time. Alg. 2
shows the detailed process. We useI ̃anda ̃to denote the
generated image and action, which are not actually observed
or executed in the environment. To get the future image after
Hsteps, whereHis the planning horizon, we performH
iterations of action proposal and diffusion generation. At
each iteration, the VLM policy is prompted by the goal
imageIgand the generated imageI ̃t+kat the previous
iteration to propose an action ̃at+k. The diffusion model
T ̃then generates the future imageI ̃t+k+1conditioned on
the previous imageI ̃t+kand the actiona ̃t+k. For the first
iteration, the input imageI ̃tis just the current observation
It. After this process of imagination, the generated future
imageI ̃t+Hand the plan ̃at:t+H− 1 are concatenated with
the goal and current observation, and fed into the VLM
policy for reflection. The VLM policy will then output the
final actionatto be executed. Again, action proposal and
reflection are performed by the same VLM policy with two
different prompt templates, as indicated by the superscripts
“propose” and “reflect”.
```
```
Algorithm 2Reflective Planning (Inference)
Require: current imageIt, goal imageIg, imagination hori-
zonH
1:I ̃t←It
2:fork← 0 toH− 1 do
3: ̃at+k←πproposeVLM (Ig,I ̃t+k)
4: I ̃t+k+1←T ̃(I ̃t+k, ̃at+k)
5:end for
6:at←πVLMreflect(Ig,It,I ̃t+H, ̃at:t+H− 1 )
7:Output:at
```
#### 5.Multi-Stage Robotic Manipulation Planning

#### Tasks

```
Inspired by Luo et al. (2024b), we procedurally generated
a suite of multi-stage long-horizon manipulation tasks that
require understanding of physical interactions and reasoning
about the effects of long-term action sequences. The task is
initialized with a board and a set of small pieces randomly
placed on a table. The goal is to fully assemble the board
by inserting the pieces into the board one by one. Examples
of the initial and goal configurations are shown in Fig. 5.
Detailed task generation process is included in App. A. No-
tably, most tasks include inter-locking pieces so that they
can be inserted into the board only in a specific order. This
requires strategically choosing the object to be manipulated
at each step and inferring possible interaction between this
object and the other objects already in the board. As an ex-
ample, Fig. 5(b) shows the dependencies between the pieces
in one of the tasks. The interlocking feature further necessi-
tates the agent’s ability to replan, enabling it to recover from
failures caused by previous mistakes or bad initialization.
```

```
Goal Initial pick up blue put down blue pick up yellow put down yellow pick up orange insert orange pick up yellow insert yellow pick up purple
```
```
insert yellow pick up blue insert blue pick up brown insert brown pick up pink reorient pink insert pink pick up red reorient red insert red
```
```
reorient purple put down purple pick up yellow put down yellow pick up yellow insert yellow pick up yellow put down yellow pick up purple insert purple pick up yellow
```
```
0 1 2 3 4 5 6 7 8 9
```
(^1011121314151617181920)
**15**
pick up purple insert purple put down purple pick up yellow put down yellow
**21 22 23 24 25 26 27 28 29 30 31**
insert yellow
insert purple **pick up purple** insert yellow
Diffusion generated images
Figure 4.Filmstrip of our method solving a complicated assembly task.Frames are indexed by timestep. The goal image is in the
top-left corner (with a green border). Each frame is the observation after executing the action (in black) above it. The other action in
gray is the original action proposed by the VLM if it is revised after reflection. We highlight the reflection process at timestep 15, where
the VLM first proposes an action to pick up the purple brick, but after reflection, it chooses to pick up the yellow brick instead as the
generated future state (red-bordered image) shows little progress towards the goal.
Initial
Goal
(a) (b)
Dependency graph
Figure 5.Task examples.(a) Generated multi-stage manipulation
tasks with interlocking pieces. Top: initial configurations. Bottom:
goal configurations. See App. B for more examples. (b) The graph
shows the dependencies between the objects in the blue assembly
board on the left. Each node represents an object, and each directed
edge indicates the predecessor object should be assembled before
the successor object.
We focus on the high-level planning of this long-
horizon manipulation task. We define a set of ac-
tions in the form of “[act] [obj]”, where[act] ∈
{pick up,insert,reorient,put down} is an action
primitive, and[obj]denotes the object to be manipulated.
Specifically, “pick up” grasps a piece that is not in hand
and picks it up. It can then be inserted into the board using
the “insert” action, or put back on the table using “put
down”. By invoking “reorient”, the object in hand can
be reoriented with the black fixture if necessary, so that it
is in a suitable pose for insertion. Each action primitive is
implemented as a rule-based script controller; however, in-
tegrating other low-level controllers, such as learning-based
policies like behavior cloning, is also possible. We also de-
signed an expert policy for the mentioned motor primitives,
see App. C for implementation details.

#### 6. Experiments

```
Our experiments evaluate the effectiveness of our method
and analyze its key components. We aim to answer three key
research questions. First, how well does our method perform
in long-term planning, particularly when handling complex
physical interactions? Second, how effectively does our
method generalize across different object configurations
and types, while maintaining the ability to reason and plan
reactively in dynamic environments? Third, what is the im-
pact of the reflection mechanism on the overall performance
of our method? To address these questions, we conduct com-
prehensive experiments comparing ReflectVLM against: (1)
state-of-the-art VLM models tested in zero-shot fashions,
(2) model-based planning approaches like MCTS, and (3)
```

ablation studies examining the reflection mechanism. In this
section, we first describe our experimental setup, followed
by quantitative results and qualitative analysis.

6.1. Experiment Setup and Policy Training

To evaluate the generalization capabilities of different mod-
els, we generate two distinct task sets: a training set using
the procedure described in Sec. 5, and a separate evalua-
tion set containing previously unseen configurations. The
evaluation tasks are specifically designed to test general-
ization across varying object configurations, colors, and
spatial arrangements. We particularly emphasize challeng-
ing scenarios that require sophisticated physical reasoning
and multi-step planning. For instance, some tasks begin
with objects in physically obstructing positions that prevent
direct task completion - requiring the policy to first remove
the obstructing pieces and then develop a new plan for the
original objective. Specifically, the training set contains
1000 different tasks, each generated task was randomized
to five different initial spatial arrangements, these tasks are
used to pre-train the VLM policy. At each iteration of post-
training, we randomly sample 200 out of these 1000 tasks
to further train the VLM policy with the reflection mecha-
nism. The evaluation set contains 100 different tasks that
are unseen in the training set.

As mentioned in Sec. 3, our method utilizes an oracle policy
operating in the environment’s symbolic state space to gener-
ate expert demonstrations for training. This oracle achieves
a 97% success rate across tasks, but importantly, it operates
with access to ground-truth state information. In contrast,
our VLM policy must rely solely on visual observations.
While alternative data sources like human demonstrations
could be used for training, we chose this oracle-based ap-
proach to systematically study our method’s capabilities
under controlled conditions.

During the policy pre-training phase, we utilize the oracle
policy to provide action labels, then finetune an LLaVa-1.5-
13B model (Liu et al., 2023; 2024b) with standard super-
vised learning loss. This pre-training used 5,000 expert
demonstrations (1,000 unique tasks× 5 initial configura-
tions per task). In the post-training phase, we use the same
oracle policy to further train the VLM policy from the pre-
vious stage using the procedure described in Alg. 1. For
each iteration of post-training, we collect 1k trajectories by
rolling out the VLM policy in the environment to generate
examples for fine-tuning. See App. D for training details.

6.2. Experiment Results

In this subsection, we report the results of different methods,
and discuss their implications. Unless otherwise noted,
numbers are reported across five runs, for some commercial
VLMs such as GPT-o1, we only report one run due to cost
consideration.

```
Table 1.Post-training performanceSuccess rates (%) of post-
training variants over the number of iterations.
Method Iter. 1 Iter. 2 Iter. 3
w/o reflect 58.2 74.4 77.
w/o reflect@test 64.4 76.0 82.
reflect w/ diffusion 66.2 75.8 82.
reflect w/ sim 66.8 75.4 85.
```
```
Table 2.Inference computation cost.Inference wall clock time
per step. MCTS result is averaged over 100 tasks and 1 seed; the
others are averaged over 100 tasks and 5 seeds. All experiments
are done on a single A100 GPU.
Method Inference time (s)
Ours w/o reflect@test 0.
Ours w/ diffusion 11.
Ours w/ sim 6.
MCTS 391.
```
```
VLM zero-shot To evaluate the capabilities of state-
of-the-art vision-language models, we tested several
leading VLMs including LLaVAOneVision (Li et al.,
2024), Gemini-2.0-flash (Google, 2024), Gemini-2.0-flash-
thinking (Google, 2024), GPT-4o (OpenAI, 2024a), and
GPT-o1 (OpenAI, 2024b). We specifically included Gemini-
2.0-flash-thinking and GPT-o1 as they have demonstrated
superior reasoning capabilities across various VLM bench-
marks. As shown in Fig. 6, all models achieved notably
low success rates on our tasks. Even GPT-o1, currently
the most advanced proprietary model, succeeded in only
15 out of 100 tasks, primarily on simpler cases that did not
require sophisticated physical reasoning about interlocking
mechanisms. While Gemini-2.0-flash-thinking and GPT-o
showed marginally better performance compared to other
models, indicating some improved reasoning capabilities,
their performance remains insufficient for solving our com-
plex manipulation tasks. This significant performance gap
confirms the necessity of our proposed method for handling
physically-grounded reasoning tasks. Detailed evaluation
procedures and results can be found in App. F.
```
```
MCTS To compare with model-based planning ap-
proaches, we implemented a VLM-based MCTS policy.
This implementation uses our pretrained VLM policy as a
base policy for generating candidate actions when expand-
ing tree nodes, with value estimation provided by the oracle
policy from the simulator. See App. F for implementation
details. As shown in Fig. 6, MCTS achieves a 24.0% suc-
cess rate—higher than zero-shot VLMs but lower than our
method. Notably, while the pretrained VLM policy alone
achieves a 47.8% success rate, adding MCTS actually de-
grades performance. Our analysis revealed that although
```

```
LLaVA
OneVision
```
```
Gemini-2.
flash
```
```
Gemini-2.
flash-thinking
```
```
GPT-4o GPT-o1 MCTS BC Ours w/o
reflect
```
```
Ours w/o
reflect@test
```
```
Ours w/
diffusion
```
```
Ours w/
sim
```
```
0
```
```
20
```
```
40
```
```
60
```
```
80
```
```
100
```
```
Success Rate (%)
0.0 6.
```
```
8.0 6.
15.
```
```
24.
```
```
47.
```
```
77.8 82.2 82.
85.
```
Figure 6.Performance of our method and baselines.Success rate (%) on 100 tasks. For the zero-shot test of state-of-the-art VLMs and
MCTS, the experiments were conducted once; for other methods, the results are the average of five seeds.

MCTS helped with some challenging tasks, it would some-
times incorrectly override valid plans from the base VLM
policy. We found MCTS to be particularly challenging to
tune effectively for our domain for several reasons: (1) it is
highly sensitive to value function quality, (2) our tasks re-
quire nuanced physical reasoning that is difficult to capture
in a value function, and (3) the possibility of succeeding
from any state (by clearing the board and starting over) cre-
ates minimal value differences between states. These lim-
itations highlight the advantages of our proposed method,
which offers a lightweight, flexible approach that requires
minimal tuning and can be readily integrated with any VLM
policy.

ReflectVLM Our full method outlined in Alg. 1 and 2
incorporates reflection mechanisms in both training and in-
ference phases. To systematically evaluate the impact of
reflection, we conducted ablation experiments across sev-
eral variants of our method. As reported in Fig. 6, the
variant without reflection in both training and inference
achieved the lowest performance among our method’s vari-
ants, though it still significantly outperformed the pretrained
VLM baseline. The full method using a simulator during
inference achieves the highest success rate, serving as an
upper bound for our method’s performance. When using
a diffusion model instead of a simulator during inference,
performance degrades slightly. This is unsurprising, as our
tasks require nuanced understanding of physics and tempo-
ral dynamics—areas where current generative models still
face challenges (Kang et al., 2024; Motamed et al., 2025).
We expect our method’s performance to improve as gen-
erative models advance. We also report the post-training
dynamics in Table 1. It’s observed that the performance of
all variants increases as more training is performed and the
full method did achieve the highest performance as men-
tioned above. While the absolute performance gap between
variants may appear modest, the additional tasks solved by
including reflection are qualitatively significant. These are
typically complex scenarios requiring multiple replanning
attempts, such as removing previously placed objects to
explore alternative solutions—tasks the pretrained VLM

```
consistently failed to solve. Notably, even without reflection
during inference, our method achieves higher success rates
than the pretrained baseline. This suggests that the natural
language reflection prompts during training help the VLM
policy develop better implicit reasoning capabilities. Fig. 4
illustrates a representative example. In this complex task,
the reflection mechanism iteratively revised suboptimal ac-
tions initially proposed by the VLM policy by identifying
potentially unfavorable future states. This reflection capa-
bility proved crucial for success, as the long-horizon nature
of the task required reactive planning and continuous ad-
justment of the solution strategy. Another point to consider
is computation efficiency. Table 2 shows the wall-clock
time required per inference step. Compared to MCTS, our
method requires only a fraction of the computation time
while achieving substantially higher performance, making it
particularly appealing as a lightweight and flexible solution
for real-world applications.
```
#### 7. Discussion

```
In this work, we presented a novel post-training strategy
with reflection to improve VLM policies for long-horizon
manipulation tasks, demonstrating superior planning capa-
bilities with significantly less compute than traditional ap-
proaches like MCTS. Our current implementation opens up
exciting future directions: while we currently use final out-
comes for reflection due to VLM context constraints, future
architectures with expanded context windows could enable
richer intermediate feedback for more precise action refine-
ment; the diffusion model’s generation capabilities could be
augmented with physical constraints and improved architec-
tures to enhance prediction stability over longer horizons;
and our single-round reflection approach could be extended
to multiple rounds for iterative refinement while maintain-
ing computational efficiency. We believe our method would
benefit from continued advances in VLMs and generative
models, and we hope it could establish a new foundation
with broad applicability to sequential decision-making do-
mains requiring visual understanding, physical reasoning,
and long-horizon planning.
```

#### References

Asai, A., Wu, Z., Wang, Y., Sil, A., and Hajishirzi, H. Self-
rag: Learning to retrieve, generate, and critique through
self-reflection.arXiv preprint arXiv:2310.11511, 2023.

Bai, J., Bai, S., Du, S., Han, S., Liu, P., et al. Qwen-vl: A
versatile vision-language model for understanding, gen-
eration, and retrieval.arXiv preprint arXiv:2308.12966,
2023.

Belkhale, S., Ding, T., Xiao, T., Sermanet, P., Vuong, Q.,
Tompson, J., Chebotar, Y., Dwibedi, D., and Sadigh, D.
Rt-h: Action hierarchies using language, 2024. URL
https://arxiv.org/abs/2403.01823.

Black, K., Nakamoto, M., Atreya, P., Walke, H., Finn, C.,
Kumar, A., and Levine, S. Zero-shot robotic manipulation
with pretrained image-editing diffusion models, 2023.
URLhttps://arxiv.org/abs/2310.10639.

Brohan, A., Brown, N., Carbajal, J., Chebotar, Y., Chen,
X., Choromanski, K., Ding, T., Driess, D., Dubey, A.,
Finn, C., Florence, P., Fu, C., Arenas, M. G., Gopalakr-
ishnan, K., Han, K., Hausman, K., Herzog, A., Hsu, J.,
Ichter, B., Irpan, A., Joshi, N., Julian, R., Kalashnikov,
D., Kuang, Y., Leal, I., Lee, L., Lee, T.-W. E., Levine,
S., Lu, Y., Michalewski, H., Mordatch, I., Pertsch, K.,
Rao, K., Reymann, K., Ryoo, M., Salazar, G., Sanketi,
P., Sermanet, P., Singh, J., Singh, A., Soricut, R., Tran,
H., Vanhoucke, V., Vuong, Q., Wahid, A., Welker, S.,
Wohlhart, P., Wu, J., Xia, F., Xiao, T., Xu, P., Xu, S.,
Yu, T., and Zitkovich, B. Rt-2: Vision-language-action
models transfer web knowledge to robotic control, 2023a.
URLhttps://arxiv.org/abs/2307.15818.

Brohan, A., Brown, N., Carbajal, J., Chebotar, Y., Dabis,
J., Finn, C., Gopalakrishnan, K., Hausman, K., Herzog,
A., Hsu, J., Ibarz, J., Ichter, B., Irpan, A., Jackson, T.,
Jesmonth, S., Joshi, N. J., Julian, R., Kalashnikov, D.,
Kuang, Y., Leal, I., Lee, K.-H., Levine, S., Lu, Y., Malla,
U., Manjunath, D., Mordatch, I., Nachum, O., Parada, C.,
Peralta, J., Perez, E., Pertsch, K., Quiambao, J., Rao, K.,
Ryoo, M., Salazar, G., Sanketi, P., Sayed, K., Singh, J.,
Sontakke, S., Stone, A., Tan, C., Tran, H., Vanhoucke,
V., Vega, S., Vuong, Q., Xia, F., Xiao, T., Xu, P., Xu,
S., Yu, T., and Zitkovich, B. Rt-1: Robotics transformer
for real-world control at scale, 2023b. URLhttps://
arxiv.org/abs/2212.06817.

Brooks, T., Holynski, A., and Efros, A. A. Instructpix2pix:
Learning to follow image editing instructions. arXiv
preprint arXiv:2211.09800, 2022.

Chen, B., Xu, Z., Kirmani, S., Ichter, B., Driess, D., Flo-
rence, P., Sadigh, D., Guibas, L., and Xia, F. Spatialvlm:

```
Endowing vision-language models with spatial reason-
ing capabilities, 2024. URLhttps://arxiv.org/abs/
2401.12168.
```
```
Chen, X., Dai, J., Li, X., Peng, B., Singh, M., Tao, S., Wang,
X., Wang, Y., Xia, Y., et al. Pali-x: On scaling up a
multilingual vision and language model.arXiv preprint
arXiv:2305.18565, 2023.
```
```
Cheng, K., Li, Y., Xu, F., Zhang, J., Zhou, H., and Liu,
Y. Vision-language models can self-improve reasoning
via reflection, 2024. URLhttps://arxiv.org/abs/
2411.00855.
```
```
Cui, J. and Trinkle, J. Toward next-generation learned robot
manipulation.Science Robotics, 6, 2021.
```
```
Driess, D., Ha, J.-S., and Toussaint, M. Deep visual rea-
soning: Learning to predict action sequences for task and
motion planning from an initial scene image, 2020. URL
https://arxiv.org/abs/2006.05398.
```
```
Driess, D., Black, A., Kataoka, H., Tsurumine, Y., Koyama,
Y., Mansard, N., Fox, D., Choromanski, K., Ichter, B.,
Hausman, K., et al. Palm-e: An embodied multimodal
language model.arXiv preprint arXiv:2303.03378, 2023.
```
```
Du, Y., Yang, M., Florence, P., Xia, F., Wahid, A., Ichter,
B., Sermanet, P., Yu, T., Abbeel, P., Tenenbaum, J. B.,
Kaelbling, L., Zeng, A., and Tompson, J. Video language
planning, 2023. URLhttps://arxiv.org/abs/2310.
10625.
```
```
Gao, J., Sarkar, B., Xia, F., Xiao, T., Wu, J., Ichter, B., Ma-
jumdar, A., and Sadigh, D. Physically grounded vision-
language models for robotic manipulation, 2024. URL
https://arxiv.org/abs/2309.02561.
```
```
Garrett, C. R., Chitnis, R., Holladay, R., Kim, B., Silver,
T., Kaelbling, L. P., and Lozano-P ́erez, T. Integrated
task and motion planning, 2020a. URLhttps://arxiv.
org/abs/2010.01083.
```
```
Garrett, C. R., Lozano-P ́erez, T., and Kaelbling, L. P. Pddl-
stream: Integrating symbolic planners and blackbox sam-
plers via optimistic adaptive planning, 2020b. URL
https://arxiv.org/abs/1802.08705.
```
```
Google. Introducing gemini: Our largest and
most capable ai model. https://blog.
google/technology/google-deepmind/
google-gemini-ai-update-december-2024/
#ceo-message, 2024. Accessed: 2024-02-14.
```
```
Ho, J., Jain, A., and Abbeel, P. Denoising diffusion proba-
bilistic models, 2020. URLhttps://arxiv.org/abs/
2006.11239.
```

Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang,
S., Wang, L., and Chen, W. LoRA: Low-rank adapta-
tion of large language models. InInternational Confer-
ence on Learning Representations, 2022. URLhttps:
//openreview.net/forum?id=nZeVKeeFYf9.

Hu, Y., Lin, F., Zhang, T., Yi, L., and Gao, Y. Look before
you leap: Unveiling the power of gpt-4v in robotic vision-
language planning, 2023. URLhttps://arxiv.org/
abs/2311.17842.

Huang, J., Chen, X., Mishra, S., Zheng, H. S., Yu, A. W.,
Song, X., and Zhou, D. Large language models cannot
self-correct reasoning yet, 2024. URLhttps://arxiv.
org/abs/2310.01798.

Huang, W., Wang, C., Zhang, R., Li, Y., Wu, J., and Fei-Fei,
L. Voxposer: Composable 3d value maps for robotic
manipulation with language models, 2023. URLhttps:
//arxiv.org/abs/2307.05973.

Kaelbling, L. P. and Lozano-P ́erez, T. Hierarchical task and
motion planning in the now. In2011 IEEE International
Conference on Robotics and Automation, pp. 1470–1477,

2011. doi: 10.1109/ICRA.2011.5980391.

Kang, B., Yue, Y., Lu, R., Lin, Z., Zhao, Y., Wang, K.,
Huang, G., and Feng, J. How far is video generation from
world model: A physical law perspective, 2024. URL
https://arxiv.org/abs/2411.02385.

Kelly, M., Sidrane, C., Driggs-Campbell, K., and Kochen-
derfer, M. J. Hg-dagger: Interactive imitation learn-
ing with human experts. 2019 International Confer-
ence on Robotics and Automation (ICRA), pp. 8077–
8083, 2018. URLhttps://api.semanticscholar.
org/CorpusID:52939433.

Kroemer, O., Niekum, S., and Konidaris, G. A review of
robot learning for manipulation: Challenges, represen-
tations, and algorithms, 2020. URLhttps://arxiv.
org/abs/1907.03146.

Li, B., Zhang, Y., Guo, D., Zhang, R., Li, F., Zhang, H.,
Zhang, K., Zhang, P., Li, Y., Liu, Z., and Li, C. Llava-
onevision: Easy visual task transfer, 2024. URLhttps:
//arxiv.org/abs/2408.03326.

Liu, F., Fang, K., Abbeel, P., and Levine, S. Moka: Open-
world robotic manipulation through mark-based visual
prompting, 2024a. URLhttps://arxiv.org/abs/
2403.03174.

Liu, H., Li, C., Wu, Q., and Lee, Y. J. Visual instruction
tuning, 2023. URLhttps://arxiv.org/abs/2304.
08485.

```
Liu, H., Li, C., Li, Y., and Lee, Y. J. Improved baselines
with visual instruction tuning, 2024b. URLhttps://
arxiv.org/abs/2310.03744.
```
```
Luo, J., Xu, C., Geng, X., Feng, G., Fang, K., Tan, L.,
Schaal, S., and Levine, S. Multistage cable routing
through hierarchical imitation learning.IEEE Transac-
tions on Robotics, 40:1476–1491, 2024a. doi: 10.1109/
TRO.2024.3353075.
```
```
Luo, J., Xu, C., Liu, F., Tan, L., Lin, Z., Wu, J., Abbeel, P.,
and Levine, S. Fmb: A functional manipulation bench-
mark for generalizable robotic learning.The International
Journal of Robotics Research, 2024b.
```
```
Madaan, A., Tandon, N., Gupta, P., Hallinan, S., Gao,
L., Wiegreffe, S., Alon, U., Dziri, N., Prabhumoye, S.,
Yang, Y., et al. Self-refine: Iterative refinement with self-
feedback. Advances in Neural Information Processing
Systems, 36, 2024.
```
```
Mei, A., Wang, J., Zhu, G.-N., and Gan, Z. Gamevlm:
A decision-making framework for robotic task planning
based on visual language models and zero-sum games.
arXiv preprint arXiv:2405.13751, 2024a.
```
```
Mei, A., Zhu, G.-N., Zhang, H., and Gan, Z. Replanvlm:
Replanning robotic tasks with visual language models.
IEEE Robotics and Automation Letters, 2024b.
```
```
Motamed, S., Culp, L., Swersky, K., Jaini, P., and Geirhos,
R. Do generative video models learn physical principles
from watching videos?, 2025. URLhttps://arxiv.
org/abs/2501.09038.
```
```
Nasiriany, S., Xia, F., Yu, W., Xiao, T., Liang, J., Dasgupta,
I., Xie, A., Driess, D., Wahid, A., Xu, Z., Vuong, Q.,
Zhang, T., Lee, T.-W. E., Lee, K.-H., Xu, P., Kirmani,
S., Zhu, Y., Zeng, A., Hausman, K., Heess, N., Finn,
C., Levine, S., and Ichter, B. Pivot: Iterative visual
prompting elicits actionable knowledge for vlms, 2024.
URLhttps://arxiv.org/abs/2402.07872.
```
```
OpenAI. Gpt-4o system card, 2024a. URLhttps://
arxiv.org/abs/2410.21276.
```
```
OpenAI. Openai o1 system card, 2024b. URLhttps:
//arxiv.org/abs/2412.16720.
```
```
Pan, L., Saxon, M., Xu, W., Nathani, D., Wang, X.,
and Wang, W. Y. Automatically correcting large lan-
guage models: Surveying the landscape of diverse self-
correction strategies.arXiv preprint arXiv:2308.03188,
2023.
```
```
Renze, M. and Guven, E. Self-reflection in llm agents:
Effects on problem-solving performance.arXiv preprint
arXiv:2405.06682, 2024.
```

Rombach, R., Blattmann, A., Lorenz, D., Esser, P., and
Ommer, B. High-resolution image synthesis with latent
diffusion models, 2021.

Ross, S., Gordon, G., and Bagnell, D. A reduction of imita-
tion learning and structured prediction to no-regret online
learning. InProceedings of the fourteenth international
conference on artificial intelligence and statistics, pp.
627–635, 2011.

Shi, L. X., Hu, Z., Zhao, T. Z., Sharma, A., Pertsch, K.,
Luo, J., Levine, S., and Finn, C. Yell at your robot:
Improving on-the-fly from language corrections, 2024.
URLhttps://arxiv.org/abs/2403.12910.

Shinn, N., Cassano, F., Gopinath, A., Narasimhan, K., and
Yao, S. Reflexion: Language agents with verbal rein-
forcement learning. Advances in Neural Information
Processing Systems, 36, 2024.

Silver, D., Schrittwieser, J., Simonyan, K., Antonoglou,
I., Huang, A., Guez, A., Hubert, T., Baker, L., Lai, M.,
Bolton, A., et al. Mastering the game of go without
human knowledge.nature, 550(7676):354–359, 2017.

Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A.,
Ermon, S., and Poole, B. Score-based generative model-
ing through stochastic differential equations, 2021. URL
https://arxiv.org/abs/2011.13456.

Wake, N., Kanehira, A., Sasabuchi, K., Takamatsu, J., and
Ikeuchi, K. Gpt-4v (ision) for robotics: Multimodal task
planning from human demonstration.IEEE Robotics and
Automation Letters, 2024.

Wang, X., Wei, J., Schuurmans, D., Le, Q., Chi, E., Narang,
S., Chowdhery, A., and Zhou, D. Self-consistency im-
proves chain of thought reasoning in language models.
arXiv preprint arXiv:2203.11171, 2022.

Wang, Y., Kordi, Y., Mishra, S., Liu, A., Smith, N. A.,
Khashabi, D., and Hajishirzi, H. Self-instruct: Aligning
language models with self-generated instructions, 2023.
URLhttps://arxiv.org/abs/2212.10560.

Wang, Z., Garrett, C. R., Kaelbling, L. P., and Lozano-Perez, ́
T. Learning compositional models of robot skills for task
and motion planning, 2021. URLhttps://arxiv.org/
abs/2006.06444.

Wei, J., Wang, X., Schuurmans, D., Bosma, M., Xia, F., Chi,
E., Le, Q. V., Zhou, D., et al. Chain-of-thought prompting
elicits reasoning in large language models.Advances in
neural information processing systems, 35:24824–24837,
2022.

Yao, S., Yu, D., Zhao, J., Shafran, I., Griffiths, T., Cao, Y.,
and Narasimhan, K. Tree of thoughts: Deliberate problem

```
solving with large language models.Advances in Neural
Information Processing Systems, 36, 2024.
```
```
Yu, X., Peng, B., Vajipey, V., Cheng, H., Galley, M., Gao,
J., and Yu, Z. Exact: Teaching ai agents to explore with
reflective-mcts and exploratory learning, 2025. URL
https://arxiv.org/abs/2410.02052.
```

#### A. Task generation

We here describe the procedure to generate assembly boards in detail with an example. A board is discretized into voxels
and can be represented by a 3d array, where each value indicates the piece the voxel belongs to. Initially none of the voxels
is occupied, so they are all set to an empty value 0, as shown in Fig. 7(a). Then we iteratively add pieces to the board. We
first sample the size of the base board, which is (12, 12, 3) in this example (Fig. 7(b)). Then we set these voxels to 1 to
indicate they belong to the base board. We also maintain a variablemaxheight, which represents the highest layer that
contains non-zero voxels. To generate a brick, we sample its size and position subject to some constraints (Fig. 7(c)). The
first two constraints ensure that this brick is within the range of the base board, and the third constraint makes sure this
brick will intersect with some previously generated brick. As before, we set the value of the red voxels to 2 to indicate they
are from the new brick. Note that the voxels in the lower layer previously have a value of 1 since they belonged to the
base board, but now their value is rewritten to 2. This also creates a hole on the base board. After generating this brick, we
also updatemaxheightto 4 since we have 4 layers now. Fig. 7(d) shows the process of generating another brick. As the
new blue brick intersects with the old red brick at the four critical voxels highlighted in purple (Fig. 7(e)), we can assign
the value of these critical voxels to either that of the red one or the blue one. For example, keep these voxels to the red
brick results in an opening on the blue one (Fig. 7(f)). Stopping the generation process here gives us a board with three
interlocking pieces, as shown in Fig. 7(g).

```
(0,0,0)^12
```
```
12
```
```
3
```
```
sample size (a, b, h),
position (x, y, z)
s.t. a + x ≤ 12
b + y ≤ 12
z < 4
```
```
size =(2, 6, 2)
position = (6, 1, 3)
```
```
max_height= 3 max_height= 4
sample size (a, b, h),
position (x, y, z)
s.t. a + x ≤ 12
b + y ≤ 12
z < 3
```
```
max_height= 5
```
```
size = (8,2,2)
position = (2,2,2)
```
```
(a) (b) (c)
```
```
(d) (e) (f)
```
```
(g)
```
```
12
```
```
12
```
```
3
```
```
12
```
```
12
```
```
3
```
Figure 7.Example of task generation.(a) Voxel representation of the board. (b) Generating a base board. (c) Generating a red brick. (d)
Generating another blue brick. (e) Critical voxels (highlighted in purple) at the intersection of the two bricks. (f) Handling intersection by
assigning the critical voxels to the red brick. (g) Explosion view of the board consisting of three interlocking pieces.

#### B. Samples of generated tasks

```
Initial
```
```
Goal
```
Figure 8.Samples of generated tasks.We procedurally generate a variety of multi-stage manipulation tasks, ranging from simple peg
insertion to complex assembly tasks that contains multiple interlocking pieces. Top: initial configurations. Bottom: goal configurations.


#### C. Expert policy

The expert policy assumes access to the states of the objects in the simulator, such as the position and orientation of each
piece. It is also provided with the dependency graph of the task, as discussed in Sec. 5. We define the status of each piece to
be one of the following:

- DONE: if it is properly inserted into board;
- READY: if it is not inserted yet but ready to be manipulated;
- BADB: if it is inbadstate since it isblockingother bricks, implying it needs to be removed;
- BADD: if it is inbadstate since it isdown, implying it needs to be reoriented;
- BLOCKEDP: if it isblockedsince somepredecessorbrick(s) should be inserted before;
- BLOCKEDS: if it isblockedsince somesuccessorbrick(s) is inserted before.

Based on the status of each piece, we can also define a set of possible statuses for the whole assembly task:

- DONE: if the board is fully assembled, i.e., all pieces are inDONEstate;
- READY: if some brick is inREADYorBADDstate;
- BADB: if we need to reset some brick(s) to proceed as it is blocking other bricks.

When queried, the expert policy first checks the status of each piece according to the simulation states, and decide the status
of the whole task based on the statuses of all pieces. Then it decides the action to take following Algorithm 3.

Algorithm 3Expert Policy
Require: task statusstatusglobal, object in handobjhand,
1:ifobjhandis not Nonethen
2: ifall predecessors ofobjhandareDONEthen
3: ifobjhandis inBADDstatethen
4: return“reorientobjhand”
5: else ifobjhandis inBLOCKEDSstatethen
6: return“put downobjhand”
7: else
8: return“insertobjhand”
9: end if
10: else
11: return“put downobjhand”
12: end if
13:else
14: ifstatusglobal==READYthen
15: choose an objectobjinREADYorBADDstate
16: return“pick upobj”
17: else ifstatusglobal==BADBthen
18: choose an objectobjinBADBstate
19: return“pick upobj”
20: else
21: return“done”
22: end if
23:end if

#### D. Training details

D.1. VLM Policy

Architecture.As shown in Fig. 9, the architecture of our VLM consists of a vision encoder and a Large Language Model
(LLM). By default, we useclip-vit-large-patch14-336^1 as the vision encoder, andvicuna-13b-v1.5^2 as the LLM.
We initialize our VLM with LLaVA-v1.5 weights^3 that are pre-trained on general visual instruction tuning datasets. Since

(^1) https://huggingface.co/openai/clip-vit-large-patch14-
(^2) https://huggingface.co/lmsys/vicuna-13b-v1.
(^3) https://huggingface.co/liuhaotian/llava-v1.5-13b


# Large Language Model

##### text image text image text

##### text

```
Vision
Encoder
```
```
Vision
Encoder
```
##### shared vision encoder

## LoRA

Figure 9.Architecture of our VLM.The model consists of a vision encoder and an LLM. We also add Low-Rank Adaptation (LoRA) (Hu
et al., 2022) layers to LLM for efficient adaptation. The input sequence contains interleaved images and text, where images are encoded
into latent embeddings with a shared vision encoder. Finally, the concatenation of text and image embeddings are fed into VLM for
multimodal reasoning.

our task prompts consist of interleaved images and text (refer to Sec. E), we use a shared vision encoder to extract latent
embeddings and concatenate them back to an input sequence.

Training Parameters.The full training parameters are listed in Table 3. For efficient adaptation of VLM to our task, we
only finetune newly added LoRA (Hu et al., 2022) layers. The rank of LoRA layers is 128 by default.

```
Table 3.Training parameters of VLM.
```
```
Res
LoRA Training Batch
Optimizer
Warmup Learning rate Weight LR
Rank Epoch Size Epoch BC Iter. 1,2,3 Decay Schedule
336px 128 1 128 AdamW 0.03 5e-5 1e-5 0.0 Cosine
```
D.2. Diffusion Dynamics Model

Data Generation.We generate 10Kdifferent boards and use sub-optimal policies to collect transitions. The sub-optimal
policies are implemented by setting a probabilityp={ 0. 2 , 0. 5 , 0. 7 , 0. 9 , 1. 0 }to replace the expert action by a random
action. We collect 50Ktrajectories; each has a maximum length of 50 and is terminated upon success. In total, we have
about 1Mtransitions. We randomly sample 50Ktransitions for evaluation, and the rest is used for training.

Training Parameters.The full training parameters are listed in Table 4. We initialize the Diffusion Dynamics Model with
pretrained Instructpix2pix (Brooks et al., 2022)^4.

```
Table 4.Training parameters of Diffusion Dynamics Models.
```
```
Model Res
Training Batch
Optimizer
Warmup Learning Weight Beta1, Grad LR
Steps Size Steps Rate Decay Beta2 Norm Schedule
UNet 512px 20K 640 AdamW 2K 1e-4 0.01 0.9, 0.999 1.0 Cosine
Decoder 512px 4K 160 AdamW 1K 1e-7 0.01 0.9, 0.999 1.0 Cosine
```
(^4) https://huggingface.co/timbrooks/instruct-pix2pix


#### E. Prompts

E.1. Action proposal prompt

```
There is a puzzle consisting of a board and several pieces with different colors on the table. The goal is to assemble
the puzzle with the robot arm. In each step, one of the following four actions can be taken: pick up [obj], put down
[obj], reorient [obj], and insert [obj], where [obj] refers to the piece to be manipulataed. The image of the goal state
is:<image>. The image of the current state is:<image>. The most recently executed actions are:{history}.
What action should be taken next? Note that [obj] should be a color chosen from the following list:{colors}.
```
E.2. Reflection prompt

```
There is a puzzle consisting of a board and several pieces with different colors on the table. The goal is to assemble
the puzzle with the robot arm. In each step, one of the following four actions can be taken: pick up [obj], put down
[obj], reorient [obj], and insert [obj], where [obj] refers to the piece to be manipulataed. The image of the goal state
is:<image>. The image of the current state is:<image>. The most recently executed actions are:{history}.
The next five steps planned by the model is{initplan}, from which we are going to only execute the first action.
Note that if the full plan was executed sequentially, the future state would be:<image>. What action should be
taken for the immediate next step? Note that [obj] should be a color chosen from the following list:{colors}. You
can modify the initial plan if it leads to an undesired future state.
```
#### F. Baseline details

F.1. Zero-shot VLMs

We prompt state-of-the-art close-sourced and open-sourced VLMs for zero-shot evaluation, including LLaVA-Onevision,
Gemini-2.0 (gemini-2.0-flash-exp), Gemini-2.0-thinking (gemini-2.0-flash-thinking-exp-1219), GPT-4o and
GPT-o1. We resize all input images to 336×336 pixels for fair comparisons with our model. We set the generation
temperature and max planing step to 0 and 50. The evaluation prompt is:

```
You are an intelligent robot equipped with cameras and robotic arms, your primary task is to observe and interact
with the objects on the desktop.
```
```
{Action proposal prompt (Sec. E.1)}
```
```
You can only output the action, e.g., pick up red. Do not output anything else.
```
Since the instruction following capability of LLaVA-Onevision is quite limited, we cannot extract valid actions from its
response. For other close-sourced VLMs, we list the detailed evaluation results in Table 5. We also visualize some success
cases in Figures 10 and 11, and failure cases in Figures 12 to 15.

```
Table 5.Detailed evaluation results of zero-shot VLMs.
Model Success Trajectory ID / Planing Steps Max Steps Min Steps Avg Steps
Gemini-2.0 5/6, 12/4, 16/18, 47/11, 60/4, 86/6 18 4 8.
Gemini-2.0-Thinking 5/6, 12/4, 40/20, 47/16, 50/8, 60/8, 86/10, 90/11 20 4 10.
GPT-4o 12/15, 16/5, 19/4, 47/10, 60/4, 90/6 15 4 7.
```
```
GPT-o
```
###### 12/9, 16/6, 17/15, 47/8, 50/16, 58/18, 60/14, 62/33,

###### 33 4 13.

###### 66/6, 67/12, 72/32, 77/9, 85/9, 86/6, 90/


###### F.2. MCTS

We implemented MCTS similar to AlphaGo Zero (Silver et al., 2017) but with a VLM policy for action proposal and a
heuristic value estimator. States and actions are represented by nodes and edges, respectively. The algorithm iteratively
expands the search tree and estimates the value for different actions. We store the visit countN(s,a), total action value
W(s,a), and action valueQ(s,a) =W(s,a)/N(s,a)on edges. Each iteration consists of three phases: (1) select, (2)
expand, and (3) backup.

In select phase, it traverses the tree by selecting the edge that has the largest action valueQ(s,a)plus an upper confidence
boundU(s,a) =cexplore

pP
a′N(s,a′)/(1 +N(s,a)), wherecexploreis the factor to balance exploring less visited edges
and exploiting edges with high value. We usecexplore= 0. 5 in our experiments. If there is no actions associated to a node
yet, it samples 5 top-likelihood actions with the VLM, with duplicates removed, and adds them to the node.

In expand phase, it expands the selected edge by simulating the action in the simulator, getting the next state, and adding the
new state to the tree as a new node. It then estimates the value of the new state by rolling out the expert policy from that
state. The estimated value isV= exp(−λS), whereSis the number of steps the expert policy takes to reach the goal from
the new state, andλ= 0. 1 is a scaling factor.

In backup phase, it updates the statistics of the edges on the path from the root to the expanded node:N(s,a)←N(s,a)+1,
W(s,a)←W(s,a) +V, andQ(s,a)←W(s,a)/N(s,a).

The search completes after 50 iterations. Among all actions connected to the root node, the action with the highestQvalue
is chosen to execute. We replan with MCTS at each timestep.


Figure 10.Success cases of zero-shot VLMs.Top: Gemini-2.0; Middle: Gemini-2.0-Thinking; Bottom: GPT-4o.


Figure 11.Success cases of zero-shot VLMs (GPT-o1).


Figure 12.Failure case of Gemini-2.0.


Figure 13.Failure case of Gemini-2.0-Thinking.


Figure 14.Failure case of GPT-4o.


Figure 15.Failure case of GPT-o1.


##### 𝐼%

```
Put down brown Insert red
```
```
Pick up orange
```
```
Pick up yellow Insert orange
```
```
Insert yellow
```
```
Pick up red Insert orange Put down yellow
```
```
Put down pink Insert brown Pick up pink
```
##### 𝑎%

##### 𝐼%&'

##### (a) Success actions

##### (b) Infeasible actions

##### (c) Failure actions

##### 𝐼%

##### 𝑎%

##### 𝐼%&'

##### 𝐼%

##### 𝑎%

##### 𝐼%&'

```
Figure 16.Examples of Diffusion Dynamic Models.
```

