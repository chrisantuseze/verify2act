#### Points2Plans: From Point Clouds to Long-Horizon Plans

#### with Composable Relational Dynamics

###### Yixuan Huang^1 ,^2 , Christopher Agia^1 , Jimmy Wu^3 , Tucker Hermans^2 ,^4 , and Jeannette Bohg^1

```
Constrained Packing Constrained Retrieval
```
```
Multi-object Retrieval Occluded Object Retrieval
```
###### Data Generation Planning Real WorldExecution

###### 𝐷={(𝑠,𝑎,𝑠!)}"%#$

###### 𝑠

###### 𝑠!

###### 𝑎

```
Point
Clouds
```
```
Relational
Dynamics
```
```
Skills
```
```
Instructions
```
```
Task
Planner
```
```
Fig. 1: Training, planning, and execution phases of Points2Plans. Left:In simulation, we sample an environment state and execute a manipulation
primitive at random to generate a dataset of single-step environment transitions and train a relational dynamics model.Middle:At planning time, Points2Plans
receives a language instruction and a partial-view, segmented point cloud of the scene and then performs long-horizon planning in a hierarchical fashion with
a task planner (e.g., a language model) and the learned relational dynamics model. If planning is successful, Points2Plans returns a sequence of manipulation
primitives (what to execute) and their associated continuous parameters (how to execute them) for the given task.Right:Points2Plans executes its plan to
solve a variety of unseen long-horizon tasks in the real world.
Abstract— We present Points2Plans, a framework for com-
posable planning with a relational dynamics model that enables
robots to solve long-horizon manipulation tasks from partial-
view point clouds. Given a language instruction and a point
cloud of the scene, our framework initiates a hierarchical plan-
ning procedure, whereby a language model generates a high-
level plan and a sampling-based planner produces constraint-
satisfying continuous parameters for manipulation primitives
sequenced according to the high-level plan. Key to our approach
is the use of a relational dynamics model as a unifying
interface between the continuous and symbolic representations
of states and actions, thus facilitating language-driven planning
from high-dimensional perceptual input such as point clouds.
Whereas previous relational dynamics models require training
on datasets of multi-step manipulation scenarios that align
with the intended test scenarios, Points2Plans uses only single-
step simulated training data while generalizing zero-shot to a
variable number of steps during real-world evaluations. We
evaluate our approach on tasks involving geometric reasoning,
multi-object interactions, and occluded object reasoning in both
simulated and real-world settings. Results demonstrate that
Points2Plans offers strong generalization to unseen long-horizon
tasks in the real world, where it solves over 85% of evaluated
tasks while the next best baseline solves only 50%.
```
```
I. INTRODUCTION
Before robots can make their entry as general-purpose
helpers in e.g., household environments, they must learn to
solvesequential manipulationtasks in the presence of partial
```
(^1) Stanford University. (^2) University of Utah. (^3) Princeton University. 4
NVIDIA Research.
occlusions while receiving high-dimensional sensor data as
input. Consider the “constrained packing” task shown in
Fig. 1, where the robot must place all cups into the shelf
without collision. To succeed, the robot has to reason about
the long-horizon effects of its actions (e.g. what happens if
the first cup is placed at the front of the shelf?) without
perfect knowledge of object geometries or poses.
The most common paradigm for solving sequential ma-
nipulation tasks decomposes a task into a sequence of skills
for the robot to execute [1,2]. The open problem remains:
how to sequence skills without beingmyopic; returning to
our example, placing the first cup at the front of the shelf
prevents future placements. Traditionally, this problem is
addressed by task and motion planning (TAMP) systems,
which perform a search for feasible solutions at the sym-
bolic and geometric level [3,4]. However, TAMP typically
assumes access to explicit 3D object models and symbolic
operators with predefined effects [5]; assumptions that may
not hold in increasingly unstructured and partially observable
environments. Other approaches leverage policy hierarchies
to learn long-horizon strategies with reinforcement learning
(RL) [6–8]. However, these approaches aim to learn skill
sequencing strategies for each new long-horizon task, while
we seek to compose skills through planning to solve a
large set of downstream tasks [9–11]. We thereby ask:How
can we enable composable planning in high-dimensional
observation spaces without predefined symbolic operators?
In this paper, we argue that transformer-based relational

## arXiv:2408.14769v2 [cs.RO] 4 Mar 2025


dynamics (RD) [12,13] is key to enabling composable, long-
horizon planning directly from partial-view point clouds. RD
models implicitly capture the symbolic and geometric effects
of robot actions in a shared (object-centric) latent space,
which facilitates goal-directed planning. We first propose
an RD model architecture that requires only randomized
single-step environment transitions(s,a,s′)for training, but
can be iteratively applied to predict long-horizon trajectories
(s 1 ,a 1 ,...,sH)at plan time. Each single-step transition
(s,a,s′)corresponds to the states before and after executing
a manipulation primitive (e.g., picking an apple from a
basket and placing it on the table), and thus represents
an abstraction over low-level trajectories executed on the
robot [14]. Second, we introduce a sampling-based planning
algorithm that selects robot actions that maximize the like-
lihood of symbolic goals predicted by the RD model. This
algorithm uses a new rollout strategy that interweavesdelta-
stateprediction of objects in the latent space with object pose
updates in geometric space, resulting in greater accuracy over
long-horizons compared to predicting absolute object states
as in prior work [13]. Finally, we leverage large language
models (LLMs) [15] to accelerate our planner by predicting
candidate plan skeletons; in effect, significantly reducing
the number of discrete skill sequences our planner must
search through for any given task. The combination of these
components forms the Points2Plans planning framework.
Our contributions are three-fold: 1) A relational dy-
namics model that excels at long-horizon prediction of
point cloud states without the need to train on multi-
step data; 2) A latent-geometric space dynamics roll-
out strategythat significantly increases the horizons over
which predicted point cloud states are reliable for planning;
3) A planning framework, Points2Plans, that integrates
our RD model, rollout strategy, sampling-based planner,
and task planner to solve complex long-horizon tasks. In
extensive experiments, we demonstrate that Points2Plans
generalizes to sequential manipulation tasks involving par-
tial occlusions, long-horizon geometric dependencies, and
multi-object interactions in both simulated and real-world
settings. For qualitative demonstrations of our approach
operating on a mobile manipulator platform and supplemen-
tary materials, please refer to our project page available at
sites.google.com/stanford.edu/points2plans.

II. RELATEDWORK
A standard approach to solvinglong-horizon manipula-
tion taskssequences manipulationskills[16–21] according
to a high-level plan produced by symbolic planners [2,22–
26], language models [1,11,27–30], or combinations [31–
35]. Reusability of the underlying skills should, in theory,
support generalization to a variety of tasks. However, existing
methods are often limited by myopic planning strategies.
For example, works employing visuomotor skills seldom
consider the feasibility of skill sequences and hence evaluate
tasks that do not require geometric reasoning [1,2]. Con-
versely, works that optimize skill sequences for geometrically
complex tasks often rely on hand-crafted state representa-

```
tions [9–11]. Our work jointly addresses these limitations
by enabling long-horizon lookahead planning from high-
dimensional observations (i.e., 3D point clouds).
Alternative approaches leverage policy hierarchiesto
solve long-horizon manipulation tasks through options [36–
38], parameterized action Markov decision processes [7,39–
42], model-based RL [6,43,44], and meta-learning [45,46].
Skill chaining has also been used to coordinate dependencies
among skills in a sequence [47,48]. These approaches attain
strong performance within the distribution of tasks they
are trained on, but may struggle to generalize to unseen
tasks [6,9]. Instead of training on long-horizon demonstration
data, our approach relies on random single-step environment
transitions to train a dynamics model, which is then used to
compose skills for entirely new long-horizon tasks.
A number of workslearn dynamics modelsfor planning
in high-dimensional observation spaces. Latent space dynam-
ics are used for model-based control [49–54], but predict
state changes at small timescales. Graph neural networks can
predict deformable [55,56] and multi-object [57–62] dynam-
ics. Other works generate demonstration data via differen-
tiable simulation to learn skill abstractions for deformable
object planning from high-dimensional input [63,64]. Several
works learn RD models [12,13,65] that operate on 3D point
clouds, but they lackcomposability(i.e., to support multi-
step planning, they must be trained on multi-step trajectories)
and are only demonstrated on tasks requiring up to three
consecutive skills. Our method extends the task horizon over
which RD predictions are reliable and enables the efficient
sequencing of unseen skill sequences at test time.
```
```
III. PROBLEMSETUP
We aim to solve sequential manipulation tasks given
segmented, partial-view 3D point clouds of the sceneo 1
and a natural language instruction l describing the task.
Satisfying the instructionl entails achieving a goal con-
figuration of objects (i.e., a goal state) G that can be
expressed with predicates from a closed setR. Predicates are
boolean-valued functions that describe object properties e.g.,
Movable(a),Openable(a)∈ Rand relationships among
objects e.g., Above(a, b),Inside(a, b) ∈ R, including
those that dictate action feasibility e.g.,Blocking(a, b)∈
R. A ground predicate is a predicate expressed over spe-
cific object instances (e.g.,Above(cup, table)), and afact
is an assertion of truth over a ground predicate (e.g.,
Above(cup, table) = true). Thus, we define the goal state
as a conjunction of desired factsG=g 1 ∧...∧gM, where
each factgjspecifies a desired spatial relationship among
objects. In this work, we assume that the closed set of
predicatesRis pre-specified, while noting that methods exist
to learn predicates from data [66,67].
Manipulation Primitives.To solve long-horizon tasks,
we assume access to a library of manipulation primitives
Lφ={φ^1 ,...,φK}. Each primitiveφktakes as input con-
tinuous parametersak∈Akand executes a trajectory on the
robot [18]. For example, to pick up an object, the parameters
apickcould correspond to a target grasp pose in the object
```

```
argmax 𝑃𝑮= 1 𝑧!$"#)
{𝑎!, 𝑎",𝑎#}$&%!
```
```
On(blue, rack)
On(yellow, rack)
On(orange, rack)
```
```
On(blue, shelf)
On(yellow, shelf)
On(orange, shelf)
Goal predicates 𝑮
```
```
Segmented point clouds 𝑜!
𝑧!
```
###### ...

```
X
```
###### ...

```
Task plan ( 𝑯 = 3)
```
```
{PickPlace(blue)} {PickPlace(yellow)} {PickPlace(orange)}
𝜙! 𝜙" 𝜙#
```
```
Task Planning and Goal
Prediction Module
```
```
Encoder Decoder
```
```
Decoder
```
```
𝑧$!%!
𝑧$"%!
```
```
𝑧$&%!
```
###### ...

```
𝑟$!%!
𝑟$"%!
```
```
𝑟$&%!
```
```
{𝑎!, 𝑎",𝑎#}'&(!
```
```
Dynamics Rollout
(See Figure 3) √
```
```
√
Feasibility check
Continuous Parameters Sampler
```
```
Human: Please
placeall the cups
upon the shelf
```
Fig. 2: Overview of Points2Plans.A partial-view segmented point cloudo 1 is first encoded into the (object-centric) latent statez 1.
The latent statez 1 is then decoded into predicates that serve as environment context for the task planning and goal prediction module
(e.g., an LLM), from which a task planφ1:Hand a symbolic goalGare sampled. Points2Plans then invokes a sampling-based planning
procedure to compute continuous parametersa1:Hfor the manipulation primitives in the task planφ1:H. Infeasible plans (e.g., collisions)
are rejected, and the plan that maximizes the goal likelihood in the final statezH+1is returned.

relative frame; instantiating the primitiveφpick

```

apick
```

would
move the robot’s end-effector to the target pose and close its
gripper. Anactionψkis defined as a pair of a primitive and
a parameter⟨φk,ak⟩.
Perception.We assume access to two perception modules:
a) a segmentation method that can return segmented point
cloudso; b) an object detector that returns the semantic class
of each object. In this work, we use open-source models for
segmentation [68] and detection [69].
The Planning Objective. Given an instruction l and
segmented partial-view point cloudso 1 , our objective is to
compute a planτ= [ψ 1 ,...,ψH](we use range subscripts to
denote sequences e.g.,ψ1:H) that when executed maximizes
the probability of the goal implied by instructionl:

```
arg max
G,ψ1:H
```
```
p(l|G,o 1 )p(G |ψ1:H,o 1 ). (1)
```
The first term defines the probability of observing an in-
structionlgiven observationo 1 and the user’s hidden logical
goalG. The second term defines the probability of achieving
the logical goalGgiven the initial observationo 1 and robot
actionsψ1:H.

IV. PROPOSEDAPPROACH: POINTS2PLANS
The planning objective in Eq. 1 can be optimized via a
hierarchical approach [70] that first generates atask planin
the form of a sequence of primitivesφ1:Hand then evaluates
its feasibility when planning the parametersa1:H of the
primitive sequence. We formulate our hierarchical planner
via two distributions in Eq. 2

```
arg max
G,ψ1:H
```
```
q 1 (φ1:H,G |l,o 1 )q 2 (a1:H|φ1:H,G,o 1 ). (2)
```
The first distribution q 1 (φ1:H,G|l,o 1 )represents the task
planner, which serves two roles: a) proposing candidate task
plansφ1:Hthat are symbolically correctw.r.t.the instructionl

```
and initial observationo 1 , and b) converting the instructionl
into its corresponding goal stateGused to ensure completion
of the task. In this work, we use LLMs [15] to predict
candidate task plansφ1:H and goalsG from instructions
l and textual scene descriptions, while noting that other
symbolic [13] and data-driven [71] alternatives are possible.
Given a candidate task plan, we must determine whether it
can be feasibly executed in the environment and achieve the
desired goal. Therefore, upon samplingφ ̃1:HandG ̃ from
the task planner q 1 (φ1:H,G|l,o 1 ), we sample parameters
̃a1:Hfrom the second distributionq 2 (a1:H|φ ̃1:H,G ̃,o 1 )to
approximately solve the optimization problem in Eq. 2. This
second distribution represents the probability that parameters
̃a1:Hsatisfy the goalG ̃given observationo 1 and task plan
φ ̃1:H. To obtain parameters ̃a1:H, we propose a long-horizon
planning procedure with a transformer-based RD model. The
full planning procedure is visualized in Fig. 2.
In the following sections, we outline our RD model archi-
tecture (Sec. IV-A), a hybrid rollout strategy for predicting
point cloud states (Sec. IV-B), and finally, we present our
full planning approach (Sec. IV-C).
A. Composable Relational Dynamics
Modeling the effects of actions on the environment (i.e.,
thedynamics) is essential for long-horizon planning. Yet,
obtaining dynamics models that are both accurate and appli-
cable to a wide range of downstream tasks is challenging for
several reasons: a) they are difficult to learn with e.g., imper-
fect state knowledge or multi-object interactions; b) models
trained on one distribution of long-horizon sequences may
not generalize well to others. To address these challenges,
we propose several design considerations for transformer-
based RD models [13] that yield significant improvements
in prediction accuracy and allow the model to be chained
to predict entirely new long-horizon sequences. Our RD
model is comprised of three components: an encoderEnc,
```

```
𝑧! 𝛿𝑧!
```
```
Predicted
pos changes
𝛿𝑝!
```
```
Transformed
point clouds
𝑜!"#
```
```
𝑧!"#
```
```
Encoder
```
```
Repeat 𝑯 times
```
```
Dynamics Decoder
```
```
Transformation
```
```
𝜓!
Fig. 3: Points2Plans hybrid rollout strategy.
```
a transformer-based dynamics modelT, and a decoderDec,
all of which are jointly trained on single-step environment
transitions. We describe the details of each component below.
Encoder. The encoder Enc takes as input segmented
point cloudsot=o^1 t,...,oMt at timesteptand produces a
factored, object-centric latent statezt=zt^1 ,...,zMt , where
Mis the number of objects in the scene (which may vary
across tasks). It embeds each per object segment using
PointConv [72] and appends a learned positional embedding
in PyTorch [73] to the resultant per object latent, giving
zt=Enc(ot).
Dynamics.We propose adelta-dynamicsmodelT that
takes as input the current latent stateztand actionψt=
⟨φt,at⟩, and predicts the delta state in the latent space
asδzt = T(zt,ψt). We use a transformer as the delta-
dynamics modelT since its inductive bias can represent
interactions among the multiple objects inztas a result of
actionψt. Our hypothesis is that it is easier to learn the
relative effectδztof an actionψtthan it is to directly predict
the resulting absolute statezt+1 (i.e., hereafter referred
toabsolute dynamics[12,13]), since the relative effects of
actions might be similar across many stateszt. We show in
Sec. V that the choice of delta dynamics translates to notable
improvements in pose and predicate prediction accuracy.
Decoder.The decoderDecconsists of two heads: a rela-
tion decoderDecrand a pose decoderDecp. The relation de-
coderDecrpredicts the probability of each ground predicate
beingtrue. More formally, ifUrepresents the set of ground
predicates, thenrt=Decr(zt) ={p(u= true|zt)|∀u∈
U}. The probability p(u= true|zt)for one ground predi-
cateu∈Uis denoted byDecur(zt). This decoder head can
operate on any latent statezt; for instance, it can be used to
detect facts at the initial state asr 1 =Decr(Enc(o 1 )) =
Decr(z 1 ). The pose decoderDecptakes as input a delta
state in the latent spaceδztand predicts the relative pose
change of all objects in the scene asδpt=δp^1 t,...,δpMt =
Decp(δzt). Hence, this decoder can only be applied to delta
states predicted byT.

B. Hybrid Rollout Strategy

We propose a hybrid latent-geometric space dynamics
rollout strategy that uses the RD encoderEnc, dynamics
T, and decoderDec(described in Sec. IV-A) to predict the
future states of a given planτ=ψ1:H, i.e., based on the task
planφ1:Hand its continuous parametersa1:H. The rollout
strategy is visualized in Fig. 3.
Let us consider the first timestep: the hybrid rollout
strategy first encodes segmented point cloudo 1 into the
latent statez 1 =Enc(o 1 ). Conditioned on the first action

```
ψ 1 , the delta state is then predicted asδz 1 = T(z 1 ,ψ 1 ).
The decoderDecppredicts the delta change in poseδp 1.
Finally,δp 1 is used to transform the point cloudso 1 to obtain
o 2 =ω(δp 1 )o 1. This process is repeated for all timesteps
Hin the plan resulting in the final point cloudoH+1and
latentzH+1state. By interweaving latent and geometric state
representations, our rollout strategy mitigates compounding
prediction errors in the latent space.
```
```
C. Planning Action Sequences with Relational Dynamics
We outline our full approach to planning an action se-
quenceψ1:Hfrom a language instructionland the segmented
point cloud of the sceneo 1 (visualized in Fig. 2). Our
approach is hierarchical: we solve the optimization problem
in Eq. 2 by first generating a candidate task planφ ̃1:Hwith
the LLM and then attempting to sample a set of continuous
parametersa ̃1:Hthat the robot can feasibly execute.
Given an instructionlwith an initial observationo 1 , we
sampleφ ̃1:HandG ̃fromq 1 (φ1:H,G|l,o 1 ). In practice, we
use a shooting-based method [11], which queries an LLM
few-shot to predictNtask plans{φ ̃i1:Hi}Ni=1and their corre-
sponding symbolic goals{G ̃i}Ni=1. For each task planφ ̃1:H
and goalG ̃predicted by the LLM, we seek to generate prim-
itive parameters ̃a1:Hfrom distributionq 2 (a1:H|φ ̃1:H,G ̃,o 1 )
that will approximately maximize the objective in Eq. 1. We
formulate the planning process for parameters ̃a1:Has the
following constrained optimization problem:
```
```
a ̃∗1:H = arg max
a ̃1:H∼q 2
```
```
Y
```
```
g∈G ̃
```
```
Decgr(zH+1) (3)
```
```
subject toDeccr(zt)< ε,∀c∈C,∀t∈ 1 ,...,H+ 1
(4)
wherezt=Enc(ot),∀t∈ 1 ,...,H+ 1 (5)
δzt=T
```
```

zt,⟨φ ̃t, ̃at⟩
```
```

,∀t∈ 1 ,...,H (6)
δpt=Decp(δzt),∀t∈ 1 ,...,H (7)
ot+1=ω(δpt)ot,∀t∈ 1 ,...,H (8)
```
```
We optimize Eq. 3 to maximize the probability of achieving
goal predicatesG ̃using sampling-based optimization tech-
niques [74]. The relation decoderDecr is used to com-
pute the probability of a ground goal predicategholding
true in the final latent statezH+1, which we denote with
Decgr(zH+1).Cin Eq. 4 represents the set of all feasibility-
related ground predicates, such asBlocking(bowl, cup).
During optimization, we reject parameter sequences ̃a1:H
that violate feasibility constraints, i.e., ground predicates
c∈ Cwhose probability (predicted by the relation decoder
Deccr(zt)) exceeds a calibrated thresholdεc. For example,
we would reject a plan that attempts to grasp acup if
Blocking(bowl, cup)holds true. The remaining equa-
tions (Eq. 5-Eq. 8) correspond to the steps of our hybrid
rollout strategy (Sec. IV-B).
For each candidate task planφ ̃1:Hand goalG ̃predicted
by the LLM, we compute their corresponding parameters
̃aQ∗1:H via optimization (Eq. 3). If the success probability
g∈G ̃Dec
g
r(zH+1)resulting from the optimal planψ
∗
1:H=
```

⟨φ ̃1:H, ̃a∗1:H⟩exceeds a success thresholdεs(e.g., 90 %), we
execute the plan on the robot. However, if no task plan
predicted by the LLM is successful or constraint-satisfying,
we fall back to a graph search strategy that enumerates all
possible primitive sequences up to a specified search depth
(as in [13]). This ensures that more task plans will be tested
should the LLM fail to produce a correct plan.

V. EXPERIMENTS
We conduct experiments to test the following questions:
Q1: Can Points2Plans generalize to unseen long-horizon
tasks despite only being trained on single-step environment
transitions?Q2:Does our hybrid rollout strategy and delta-
dynamics model improve prediction accuracy compared to
previous RD rollout formulations?Q3:Does Points2Plans
outperform approaches that sequence skills without pre-
dicting dynamics or reasoning about feasibility?Q4:Can
LLMs improve the planning efficiency of Points2Plans? We
generate a dataset of over 36,000 random executions of
manipulation primitives in IsaacGym [75] to train our RD
model and use GPT-4 [76] as the LLM for all experiments.
Dynamics Planning Baselines. We test the perfor-
mance of Points2Plans against five planning baselines.
Points2Plans−Geo (read “minus geo”) uses the same RD
model as Points2Plans but performs rollouts exclusively in
the latent space, i.e., without the point cloud transforma-
tion in our hybrid rollout strategy (Sec. IV-B). Conversely,
Points2Plans−Delta uses our hybrid rollout strategy but em-
ploys an absolute-dynamics model to predict the absolute
state zt of objects instead of the delta stateδzt. eRD-
Transformer [13] represents the current state-of-the-art RD
planning approach, which uses a latent space rollout strategy
and an absolute-dynamics model. We train eRDTransformer
using single-step transitions instead of multi-step transi-
tions for fair comparison. Pairwise-RD [49] is equivalent
to eRDTransformer except it only captures pairwise object
interactions with a multi-layer perceptron (MLP) instead of
multi-object interactions with transformers. Greedy selects
actionsa1:Hto avoid collisions without dynamics prediction
and long-horizon planning.
Task Planning Baselines. We test the performance of
Points2Plans under an LLM task planner and two baselines.
Search performs an exhaustive graph search as in [13]
for task planning. Points2Plans−Feasibility uses the LLM
without access to the feasibility-related ground predicates,
e.g.,Blocking(a, b).
Experimental Tasks.We evaluate our approach and base-
lines across a suite of sequential manipulation tasks (Fig. 1).
Constrained Packingtasks the robot with shelving multiple
objects in a spatially constrained environment (e.g., a kitchen
cupboard). To succeed, the robot must carefully plan the
placement positions of the objects so as to avoid collisions.
We compare Points2Plans to the RD baselines on this task
as it requires accurate RD predictions of geometric and
symbolic states.Constrained Retrievaltasks the robot with
retrieving target objects in a constrained environment. To
succeed, the robot must identify and remove objects that oc-

```
clude the target objects before retrieving them. We compare
Points2Plans to the task planning baselines on this task as it
requires the planner to infer the logically correct task plan
based on the initial state.Multi-object Retrievaltasks the
robot with retrieving an object inside a container (e.g., a
bowl) in a constrained environment. Here, the robot must
first remove the container from the constrained environment
before grasping the object from inside the container. This
task tests our planner’s ability to reason about multi-object
interactions and nested geometric dependencies.Occluded
Object Retrieval tasks the robot with retrieving objects
in a dark environment (i.e., without perception) given the
history of states and actions up until the timesteptat which
the lights are turned off. To succeed, the robot must plan
from itsmemoryof object positions and relations encoded
in the latent statezt. We present quantitative results on
theConstrained PackingandConstrained Retrievaltasks,
and qualitative results on theMulti-object Retrieval and
Occluded Object Retrievaltasks.
Across all tasks, we use three manipulation primitives cor-
responding to pick-and-place, pick-and-toss, and open/close
actions. Further details on our experiments (e.g., primitives,
predicates, hardware, training, prompts, and implementation
details) are provided in the supplementary materials made
available at sites.google.com/stanford.edu/points2plans. The
supplementary materials also includes extended experiments
studying Points2Plans’ generalization to novel scenarios and
robustness to segmentation noise.
VI. RESULTS
Simulation Experiments. We compare Points2Plans
against all planning baselines on theConstrained Packing
task to evaluate the effect of the RD model and rollout
strategy on planning performance. We run 500 trials for each
combination of planning horizon and approach. To measure
the planning performance, we report success rate, position
prediction error, and predicate prediction F1 score.
Results shown in Fig. 4a demonstrate that Points2Plans
generalizes to unseen long-horizon tasks more effectively
than the baselines (Q1). Comparing Points2Plans and
Points2Plans−Geo in Fig. 4d, we observe that our hybrid
rollout strategy contributes greatly to predicate prediction
accuracy over long horizons (Q2). Moreover, comparing
Points2Plans with Points2Plans−Delta, eRDTransformer, and
Pairwise-RD in Fig. 4c shows the importance of our delta-
dynamics model, as the baselines (which employ an absolute-
dynamics model) exhibit a larger accumulation of position
prediction error over increasing prediction horizons (Q2). Fi-
nally, we see that the Greedy approach performs significantly
worse than Points2Plans in Fig. 4a, indicating that multi-step
planning is required for the long-horizon tasks (Q3).
Real-World Experiments. We evaluate Points2Plans
against Points2Plans−Delta (the next best-performing base-
line) in the real world. We run 10 trials of each method
per task. The results in Fig. 4b show that Points2Plans
solves over 85% of long-horizon tasks and significantly
outperforms Points2Plans−Delta, which solves only 50% of
```

```
0.0 3 steps 4 steps 5 steps
```
```
0.
```
```
0.
```
```
0.
```
```
0.
```
```
1.
```
```
Success Rate
```
```
0.0 3 steps 4 steps 5 steps
```
```
0.
```
```
0.
```
```
0.
```
```
0.
```
```
1.
```
Success Rate

```
Points2Plans Points 2 Plans−Delta eRDTransformer Points 2 Plans−Geo Pairwise-RD Greedy Search Points 2 Plans−Feasibility
```
```
0.0 3 steps 4 steps 5 steps
```
```
0.
```
```
0.
```
```
0.
```
```
0.
```
```
1.
```
```
F1 Score
```
```
0.00 3 steps 4 steps 5 steps
```
```
0.
```
```
0.
```
```
0.
```
```
0.
```
```
0.
```
```
Error (m)
```
```
3 steps 4 steps 5 steps
```
```
101
```
```
102
```
```
103
```
```
Planning Time (s)
```
```
0.0 3 objs 4 objs 5 objs
```
```
0.
```
```
0.
```
```
0.
```
```
0.
```
```
1.
```
```
Correctness
```
(^0) 3 steps 4 steps 5 steps
2
4
6
8
10
Number of Successes
**(a) Simulation Success Rate
(b) Real World Success
(c) Position Prediction Error
(d) Predicate Prediction F
(e) TaskPlanningTime
(f) Task Planning Correctness**
Fig. 4: Simulation and real-world resultsfor theConstrained Packing(a-d) andConstrained Retrieval(e-f) tasks. As task complexity
increases, Points2Plans significantly outperforms baselines in terms of planning success rate (a-b), position prediction error (c), and
predicate classification accuracy (d). Interfacing Points2Plans with an LLM task planner increases planning efficiency (e) and correctness
(f). Planning time is shown on a logarithmic scale. Errors bars denote standard deviations across 500 trials.
the tasks. Fig. 5 (top row) illustrates how the baseline fails
to plan collision-free placements for multiple objects, due
to prediction errors from its RD model. In contrast, Fig. 1
and Fig. 5 show that Points2Plans effectively generalizes to
various real-world tasks without fine-tuning (Q1).
**Initial Scene Points2Plans Baselines**
Fig. 5: Points2Plans generalizes to unseen long-horizon tasks,
whereas the baselines struggle to find collision-free plans.
Task Planning Ablation.We evaluate the performance of
Points2Plans when configured with different task planning
strategies in theConstrained Retrievaltask. We run 500
trials of each approach per task. Fig. 4e shows that the
planning time of Points2Plans increases only linearly with
the LLM task planner, whereas the Search task planner
(enumerating all possible discrete parameters) results in an
exponential increase in planning timew.r.t.to the task hori-
zon (Q4). In Fig. 4f, we see that the Points2Plans−Feasibility
baseline struggles to predict feasible task plans, highlighting
the importance of providing feasibility-related predicates to
the LLM task planner as in Points2Plans. Plan executions of
Points2Plans and Points2Plans−Feasibility are shown in the
bottom row of Fig. 5. The baseline fails to remove occluding
objects before attempting to grasp the target objects behind
them, while Points2Plans infers a feasible task plan based
on feasibility-related predicates detected by the RD model.
VII. CONCLUSION
In this work, we study the problem of solving sequen-
tial manipulation tasks from partial-view point clouds and
language instructions. We present a long-horizon planning
framework, Points2Plans, that uses transformer-based rela-
tional dynamics to sequence manipulation skills and coordi-
nate their geometric dependencies. In experiments, we show
that interleaving additive, delta-state predictions in the latent
space with rigid-body transformations in the geometric space
leads to more accurate predictions of point cloud states over
long horizons. As a result, our relational dynamics model
can accurately learn the effects of robot skills from a dataset
of random, single-step transitions, and then compose the
skill effects at planning time to solve multi-step tasks. We
deploy Points2Plans on a mobile manipulator platform and
demonstrate that it can generalize to diverse real-world tasks
such as shelving kitchenware, retrieving occluded objects,
and planning from memory. Future work includes the design
of methods to identify and recover from execution failures,
online fine-tuning of the relational dynamics model to im-
prove real-world transfer, and interfacing with closed-loop
policies to solve tasks that require finer-grained motions.
VIII. ACKNOWLEDGEMENTS
This work was partially supported by NSF Awards #
and #2149585, by DARPA under grant N66001-19-2-4035,
and by a Sloan Research Fellowship. Toyota Research Insti-
tute and Toshiba provided funds to support this work.


```
REFERENCES
```
```
[1] M. Ahn, A. Brohan, N. Brown, Y. Chebotar, O. Cortes, B. David,
C. Finn, C. Fu, K. Gopalakrishnan, K. Hausmanet al., “Do as i
can, not as i say: Grounding language in robotic affordances,”arXiv
preprint arXiv:2204.01691, 2022. 1, 2
[2] B. Wu, R. Martin-Martin, and L. Fei-Fei, “M-ember: Tackling long-
horizon mobile manipulation via factorized domain transfer,” in 2023
IEEE International Conference on Robotics and Automation (ICRA).
IEEE, 2023, pp. 11 690–11 697. 1, 2
[3] M. Toussaint, “Logic-geometric programming: An optimization-based
approach to combined task and motion planning.” inIJCAI, 2015, pp.
1930–1936. 1, A
[4] C. R. Garrett, T. Lozano-P ́erez, and L. P. Kaelbling, “Pddlstream:
Integrating symbolic planners and blackbox samplers via optimistic
adaptive planning,” inProceedings of the international conference on
automated planning and scheduling, vol. 30, 2020, pp. 440–448. 1,
A
[5] C. R. Garrett, R. Chitnis, R. Holladay, B. Kim, T. Silver, L. P. Kael-
bling, and T. Lozano-Perez, “Integrated task and motion planning,” ́
Annual review of control, robotics, and autonomous systems, vol. 4,
pp. 265–293, 2021. 1, A
[6] D. Xu, A. Mandlekar, R. Mart ́ın-Mart ́ın, Y. Zhu, S. Savarese, and
L. Fei-Fei, “Deep affordance foresight: Planning through what can be
done in the future,” in2021 IEEE international conference on robotics
and automation (ICRA). IEEE, 2021, pp. 6206–6213. 1, 2
[7] M. Dalal, D. Pathak, and R. R. Salakhutdinov, “Accelerating robotic
reinforcement learning via parameterized action primitives,”Advances
in Neural Information Processing Systems, vol. 34, pp. 21 847–21 859,
```
2021. 1, 2
[8] L. X. Shi, J. J. Lim, and Y. Lee, “Skill-based model-based rein-
forcement learning,” inProceedings of The 6th Conference on Robot
Learning, ser. Proceedings of Machine Learning Research, K. Liu,
D. Kulic, and J. Ichnowski, Eds., vol. 205. PMLR, 14–18 Dec 2023,
pp. 2262–2272. 1
[9] C. Agia, T. Migimatsu, J. Wu, and J. Bohg, “Stap: Sequencing task-
agnostic policies,” in2023 IEEE International Conference on Robotics
and Automation (ICRA). IEEE, 2023, pp. 7951–7958. 1, 2, A
[10] U. A. Mishra, S. Xue, Y. Chen, and D. Xu, “Generative skill chaining:
Long-horizon skill planning with diffusion models,” inConference on
Robot Learning. PMLR, 2023, pp. 2905–2925. 1, 2
[11] K. Lin, C. Agia, T. Migimatsu, M. Pavone, and J. Bohg, “Text2motion:
From natural language instructions to feasible plans,”Autonomous
Robots, vol. 47, no. 8, pp. 1345–1365, 2023. 1, 2, 4, A
[12] Y. Huang, A. Conkey, and T. Hermans, “Planning for Multi-Object
Manipulation with Graph Neural Network Relational Classifiers,” in
IEEE International Conference on Robotics and Automation (ICRA),
2023. [Online]. Available: https://arxiv.org/abs/2209.11943 2, 4
[13] Y. Huang, N. C. Taylor, A. Conkey, W. Liu, and T. Hermans, “Latent
Space Planning for Multi-Object Manipulation with Environment-
Aware Relational Classifiers,”IEEE Transactions on Robotics (T-RO),
2024. [Online]. Available: https://arxiv.org/pdf/2305.10857.pdf 2, 3,
4, 5, A
[14] T. Silver, A. Athalye, J. B. Tenenbaum, T. Lozano-P ́erez, and L. P.
Kaelbling, “Learning neuro-symbolic skills for bilevel planning,” in
Proceedings of The 6th Conference on Robot Learning, ser. Pro-
ceedings of Machine Learning Research, K. Liu, D. Kulic, and
J. Ichnowski, Eds., vol. 205. PMLR, 14–18 Dec 2023, pp. 701–714.
2
[15] R. Bommasani, D. A. Hudson, E. Adeli, R. Altman, S. Arora, S. von
Arx, M. S. Bernstein, J. Bohg, A. Bosselut, E. Brunskillet al., “On
the opportunities and risks of foundation models,”arXiv preprint
arXiv:2108.07258, 2021. 2, 3
[16] B. D. Argall, S. Chernova, M. Veloso, and B. Browning, “A survey
of robot learning from demonstration,”Robotics and autonomous
systems, vol. 57, no. 5, pp. 469–483, 2009. 2
[17] B. C. Da Silva, G. Konidaris, and A. G. Barto, “Learning parame-
terized skills,” inProceedings of the 29th International Coference on
International Conference on Machine Learning, 2012, pp. 1443–1450.
2
[18] J. Felip, J. Laaksonen, A. Morales, and V. Kyrki, “Manipulation
primitives: A paradigm for abstraction and execution of grasping and
manipulation tasks,”Robotics and Autonomous Systems, vol. 61, no. 3,
pp. 283–296, 2013. 2

```
[19] D. Kalashnikov, A. Irpan, P. Pastor, J. Ibarz, A. Herzog, E. Jang,
D. Quillen, E. Holly, M. Kalakrishnan, V. Vanhouckeet al., “Scalable
deep reinforcement learning for vision-based robotic manipulation,” in
Conference on robot learning. PMLR, 2018, pp. 651–673. 2
[20] M. Xu, Z. Xu, C. Chi, M. Veloso, and S. Song, “XSkill: Cross
embodiment skill discovery,” in7th Annual Conference on Robot
Learning, 2023. [Online]. Available: https://openreview.net/forum?id=
8L6pHd9aS6w 2
[21] W. Liu, Y. Du, T. Hermans, S. Chernova, and C. Paxton, “Structdiffu-
sion: Language-guided creation of physically-valid structures using
unseen objects,” inProceedings of Robotics: Science and Systems
(RSS), 2023. 2
[22] L. P. Kaelbling and T. Lozano-P ́erez, “Learning composable models
of parameterized skills,” in2017 IEEE International Conference on
Robotics and Automation (ICRA). IEEE, 2017, pp. 886–893. 2
[23] D.-A. Huang, D. Xu, Y. Zhu, A. Garg, S. Savarese, L. Fei-Fei, and
J. C. Niebles, “Continuous relaxation of symbolic planner for one-
shot imitation learning,” in2019 IEEE/RSJ International Conference
on Intelligent Robots and Systems (IROS). IEEE, 2019, pp. 2635–
```
2642. 2
[24] W. Yuan, C. Paxton, K. Desingh, and D. Fox, “Sornet: Spatial object-
centric representations for sequential manipulation,” inConference on
Robot Learning. PMLR, 2022, pp. 148–157. 2
[25] S. Cheng and D. Xu, “League: Guided skill learning and abstraction
for long-horizon manipulation,”IEEE Robotics and Automation Let-
ters, 2023. 2
[26] N. Kumar, T. Silver, W. McClinton, L. Zhao, S. Proulx, T. Lozano-
P ́erez, L. P. Kaelbling, and J. Barry, “Practice makes perfect: Planning
to learn skill parameter policies,” inRobotics: Science and Systems
(RSS), 2024. 2
[27] W. Huang, F. Xia, T. Xiao, H. Chan, J. Liang, P. Florence, A. Zeng,
J. Tompson, I. Mordatch, Y. Chebotar, P. Sermanet, T. Jackson,
N. Brown, L. Luu, S. Levine, K. Hausman, and brian ichter, “Inner
monologue: Embodied reasoning through planning with language
models,” in 6th Annual Conference on Robot Learning, 2022.
[Online]. Available: https://openreview.net/forum?id=3R3Pz5i0tye 2
[28] D. Driess, F. Xia, M. S. M. Sajjadi, C. Lynch, A. Chowdhery, B. Ichter,
A. Wahid, J. Tompson, Q. Vuong, T. Yu, W. Huang, Y. Chebotar,
P. Sermanet, D. Duckworth, S. Levine, V. Vanhoucke, K. Hausman,
M. Toussaint, K. Greff, A. Zeng, I. Mordatch, and P. Florence, “Palm-
e: an embodied multimodal language model,” inProceedings of the
40th International Conference on Machine Learning, ser. ICML’23.
JMLR.org, 2023. 2
[29] J. Liang, W. Huang, F. Xia, P. Xu, K. Hausman, B. Ichter, P. Florence,
and A. Zeng, “Code as policies: Language model programs for em-
bodied control,” in2023 IEEE International Conference on Robotics
and Automation (ICRA). IEEE, 2023, pp. 9493–9500. 2
[30] I. Singh, V. Blukis, A. Mousavian, A. Goyal, D. Xu, J. Tremblay,
D. Fox, J. Thomason, and A. Garg, “Progprompt: Generating situated
robot task plans using large language models,” in2023 IEEE Interna-
tional Conference on Robotics and Automation (ICRA). IEEE, 2023,
pp. 11 523–11 530. 2
[31] D. Xu, R. Mart ́ın-Mart ́ın, D.-A. Huang, Y. Zhu, S. Savarese, and
L. F. Fei-Fei, “Regression planning networks,”Advances in neural
information processing systems, vol. 32, 2019. 2
[32] C. Wang, D. Xu, and L. Fei-Fei, “Generalizable task planning through
representation pretraining,”IEEE Robotics and Automation Letters,
vol. 7, no. 3, pp. 8299–8306, 2022. 2
[33] T. Silver, V. Hariprasad, R. S. Shuttleworth, N. Kumar, T. Lozano-
P ́erez, and L. P. Kaelbling, “Pddl planning with pretrained large
language models,” inNeurIPS 2022 foundation models for decision
making workshop, 2022. 2
[34] B. Liu, Y. Jiang, X. Zhang, Q. Liu, S. Zhang, J. Biswas, and P. Stone,
“Llm+ p: Empowering large language models with optimal planning
proficiency,”arXiv preprint arXiv:2304.11477, 2023. 2
[35] L. Zha, Y. Cui, L.-H. Lin, M. Kwon, M. G. Arenas, A. Zeng, F. Xia,
and D. Sadigh, “Distilling and retrieving generalizable knowledge
for robot manipulation via language corrections,” in 2024 IEEE
international conference on robotics and automation (ICRA). IEEE,
2024. 2
[36] R. S. Sutton, D. Precup, and S. Singh, “Between mdps and semi-mdps:
A framework for temporal abstraction in reinforcement learning,”
Artificial intelligence, vol. 112, no. 1-2, pp. 181–211, 1999. 2
[37] P.-L. Bacon, J. Harb, and D. Precup, “The option-critic architecture,” in


Proceedings of the AAAI conference on artificial intelligence, vol. 31,
no. 1, 2017. 2
[38] O. Nachum, S. S. Gu, H. Lee, and S. Levine, “Data-efficient hi-
erarchical reinforcement learning,”Advances in neural information
processing systems, vol. 31, 2018. 2
[39] W. Masson, P. Ranchod, and G. Konidaris, “Reinforcement learning
with parameterized actions,” inProceedings of the AAAI conference
on artificial intelligence, vol. 30, no. 1, 2016. 2
[40] R. Chitnis, S. Tulsiani, S. Gupta, and A. Gupta, “Efficient bimanual
manipulation using learned task schemas,” in2020 IEEE International
Conference on Robotics and Automation (ICRA). IEEE, 2020, pp.
1149–1155. 2
[41] S. Nasiriany, H. Liu, and Y. Zhu, “Augmenting reinforcement learning
with behavior primitives for diverse manipulation tasks,” in 2022
International Conference on Robotics and Automation (ICRA). IEEE,
2022, pp. 7477–7484. 2
[42] K. Fang, P. Yin, A. Nair, H. R. Walke, G. Yan, and S. Levine,
“Generalization with lossy affordances: Leveraging broad offline data
for learning visuomotor tasks,” in6th Annual Conference on Robot
Learning, 2022. [Online]. Available: https://openreview.net/forum?id=
esOrVR8-rc 2
[43] D. Shah, A. T. Toshev, S. Levine, and brian ichter, “Value function
spaces: Skill-centric state abstractions for long-horizon reasoning,”
in International Conference on Learning Representations, 2022.
[Online]. Available: https://openreview.net/forum?id=vgqS1vkkCbE 2
[44] L. X. Shi, J. J. Lim, and Y. Lee, “Skill-based model-based
reinforcement learning,” in 6th Annual Conference on Robot
Learning, 2022. [Online]. Available: https://openreview.net/forum?id=
iVxy2eO601U 2
[45] D. Xu, S. Nair, Y. Zhu, J. Gao, A. Garg, L. Fei-Fei, and S. Savarese,
“Neural task programming: Learning to generalize across hierarchi-
cal tasks,” in2018 IEEE international conference on robotics and
automation (ICRA). IEEE, 2018, pp. 3795–3802. 2
[46] D.-A. Huang, S. Nair, D. Xu, Y. Zhu, A. Garg, L. Fei-Fei, S. Savarese,
and J. C. Niebles, “Neural task graphs: Generalizing to unseen tasks
from a single video demonstration,” inProceedings of the IEEE/CVF
conference on computer vision and pattern recognition, 2019, pp.
8565–8574. 2
[47] Y. Chen, C. Wang, L. Fei-Fei, and K. Liu, “Sequential dexterity:
Chaining dexterous policies for long-horizon manipulation,” in7th
Annual Conference on Robot Learning, 2023. [Online]. Available:
https://openreview.net/forum?id=2Qrd-Yw4YmF 2
[48] Y. Lee, J. J. Lim, A. Anandkumar, and Y. Zhu, “Adversarial skill
chaining for long-horizon robot manipulation via terminal state
regularization,” in5th Annual Conference on Robot Learning, 2021.
[Online]. Available: https://openreview.net/forum?id=K5-J-Espnaq 2
[49] C. Paxton, C. Xie, T. Hermans, and D. Fox, “Predicting stable
configurations for semantic placement of novel objects,” in 5th
Annual Conference on Robot Learning, 2021. [Online]. Available:
https://openreview.net/forum?id=5DjX89Wyhk- 2, 5
[50] F. Ebert, C. Finn, S. Dasari, A. Xie, A. Lee, and S. Levine, “Visual
foresight: Model-based deep reinforcement learning for vision-based
robotic control,”arXiv preprint arXiv:1812.00568, 2018. 2
[51] D. Hafner, T. Lillicrap, I. Fischer, R. Villegas, D. Ha, H. Lee, and
J. Davidson, “Learning latent dynamics for planning from pixels,” in
International conference on machine learning. PMLR, 2019, pp.
2555–2565. 2
[52] D. Hafner, T. Lillicrap, J. Ba, and M. Norouzi, “Dream to
control: Learning behaviors by latent imagination,” inInternational
Conference on Learning Representations, 2020. [Online]. Available:
https://openreview.net/forum?id=S1lOTC4tDS 2
[53] P. Sundaresan, J. Wu, and D. Sadigh, “Learning sequential
acquisition policies for robot-assisted feeding,” in 7th Annual
Conference on Robot Learning, 2023. [Online]. Available: https:
//openreview.net/forum?id=o2wNSCTkq0 2
[54] Y. Li, S. Li, V. Sitzmann, P. Agrawal, and A. Torralba, “3d neural
scene representations for visuomotor control,” inConference on Robot
Learning. PMLR, 2022, pp. 112–123. 2
[55] H. Shi, H. Xu, S. Clarke, Y. Li, and J. Wu, “Robocook: Long-
horizon elasto-plastic object manipulation with diverse tools,” in7th
Annual Conference on Robot Learning, 2023. [Online]. Available:
https://openreview.net/forum?id=69y5fzvaAT 2
[56] H. Shi, H. Xu, Z. Huang, Y. Li, and J. Wu, “Robocraft: Learn-
ing to see, simulate, and shape elasto-plastic objects in 3d with

```
graph networks,”The International Journal of Robotics Research, p.
02783649231219020, 2023. 2
[57] M. Chang, T. Ullman, A. Torralba, and J. Tenenbaum, “A
compositional object-based approach to learning physical dynamics,”
in International Conference on Learning Representations, 2017.
[Online]. Available: https://openreview.net/forum?id=Bkab5dqxe 2
[58] P. Battaglia, R. Pascanu, M. Lai, D. Jimenez Rezendeet al., “In-
teraction networks for learning about objects, relations and physics,”
Advances in neural information processing systems, vol. 29, 2016. 2
[59] A. Sanchez-Gonzalez, N. Heess, J. T. Springenberg, J. Merel, M. Ried-
miller, R. Hadsell, and P. Battaglia, “Graph networks as learnable
physics engines for inference and control,” inInternational conference
on machine learning. PMLR, 2018, pp. 4470–4479. 2
[60] T. Kipf, E. Fetaya, K.-C. Wang, M. Welling, and R. Zemel, “Neural re-
lational inference for interacting systems,” inInternational conference
on machine learning. PMLR, 2018, pp. 2688–2697. 2
[61] D. Driess, Z. Huang, Y. Li, R. Tedrake, and M. Toussaint, “Learning
multi-object dynamics with compositional neural radiance fields,” in
Conference on robot learning. PMLR, 2023, pp. 1755–1768. 2
[62] A. Simeonov, Y. Du, B. Kim, F. Hogan, J. Tenenbaum, P. Agrawal, and
A. Rodriguez, “A long horizon planning framework for manipulating
rigid pointcloud objects,” inConference on Robot Learning. PMLR,
2021, pp. 1582–1601. 2
[63] X. Lin, Z. Huang, Y. Li, J. B. Tenenbaum, D. Held, and C. Gan,
“Diffskill: Skill abstraction from differentiable physics for deformable
object manipulations with tools,” in International Conference on
Learning Representation (ICLR), 2022. 2
[64] X. Lin, C. Qi, Y. Zhang, Z. Huang, K. Fragkiadaki, Y. Li,
C. Gan, and D. Held, “Planning with spatial-temporal abstraction
from point clouds for deformable object manipulation,” in 6th
Annual Conference on Robot Learning, 2022. [Online]. Available:
https://openreview.net/forum?id=tyxyBj2w4vw 2
[65] Y. Huang, J. Yuan, C. Kim, P. Pradhan, B. Chen, L. Fuxin, and T. Her-
mans, “Out of Sight, Still in Mind: Reasoning and Planning about
Unobserved Objects with Video Tracking Enabled Memory Models,”
inIEEE International Conference on Robotics and Automation (ICRA),
```
2024. 2, A
[66] T. Silver, R. Chitnis, N. Kumar, W. McClinton, T. Lozano-P ́erez,
L. Kaelbling, and J. B. Tenenbaum, “Predicate invention for bilevel
planning,” inProceedings of the AAAI Conference on Artificial Intel-
ligence, vol. 37, no. 10, 2023, pp. 12 120–12 129. 2, A
[67] N. Shah, J. Nagpal, P. Verma, and S. Srivastava, “From reals to logic
and back: Inventing symbolic vocabularies, actions and models for
planning from raw data,”arXiv preprint arXiv:2402.11871, 2024. 2,
A
[68] A. Kirillov, E. Mintun, N. Ravi, H. Mao, C. Rolland, L. Gustafson,
T. Xiao, S. Whitehead, A. C. Berg, W.-Y. Loet al., “Segment
anything,” inProceedings of the IEEE/CVF International Conference
on Computer Vision, 2023, pp. 4015–4026. 3
[69] X. Gu, T.-Y. Lin, W. Kuo, and Y. Cui, “Open-vocabulary
object detection via vision and language knowledge distillation,”
in International Conference on Learning Representations, 2022.
[Online]. Available: https://openreview.net/forum?id=lL3lnMbR4WU
3
[70] L. P. Kaelbling and T. Lozano-Perez, “Hierarchical task and motion ́
planning in the now,” in2011 IEEE International Conference on
Robotics and Automation. IEEE, 2011, pp. 1470–1477. 3
[71] D. Driess, J.-S. Ha, and M. Toussaint, “Deep visual reasoning:
Learning to predict action sequences for task and motion planning
from an initial scene image,” inRobotics: Science and Systems (RSS),
2020. 3, A
[72] W. Wu, Z. Qi, and L. Fuxin, “Pointconv: Deep convolutional networks
on 3d point clouds,” inProceedings of the IEEE/CVF Conference on
computer vision and pattern recognition, 2019, pp. 9621–9630. 4,
A
[73] A. Paszke, S. Gross, F. Massa, A. Lerer, J. Bradbury, G. Chanan,
T. Killeen, Z. Lin, N. Gimelshein, L. Antiga, A. Desmaison, A. Kopf,
E. Yang, Z. DeVito, M. Raison, A. Tejani, S. Chilamkurthy, B. Steiner,
L. Fang, J. Bai, and S. Chintala, “Pytorch: An imperative style, high-
performance deep learning library,” inAdvances in Neural Information
Processing Systems 32. Curran Associates, Inc., 2019, pp. 8024–8035.
4, A
[74] R. Rubinstein, “The cross-entropy method for combinatorial and
continuous optimization,”Methodology and computing in applied
probability, vol. 1, pp. 127–190, 1999. 4


[75] V. Makoviychuk, L. Wawrzyniak, Y. Guo, M. Lu, K. Storey,
M. Macklin, D. Hoeller, N. Rudin, A. Allshire, A. Handa, and
G. State, “Isaac gym: High performance GPU based physics simulation
for robot learning,” inThirty-fifth Conference on Neural Information
Processing Systems Datasets and Benchmarks Track (Round 2), 2021.
[Online]. Available: https://openreview.net/forum?id=fgFBtYgJQX 5,
A
[76] J. Achiam, S. Adler, S. Agarwal, L. Ahmad, I. Akkaya, F. L. Aleman,
D. Almeida, J. Altenschmidt, S. Altman, S. Anadkatet al., “Gpt-
technical report,”arXiv preprint arXiv:2303.08774, 2023. 5
[77] K. Rawlik, M. Toussaint, and S. Vijayakumar, “On Stochastic Optimal
Control and Reinforcement Learning by Approximate Inference,” in
Robotics: Science and Systems, 2012. A
[78] A. Conkey and T. Hermans, “Planning under uncertainty to
goal distributions,” Arxiv Preprint, 2020. [Online]. Available:
[http://arxiv.org/abs/2011.04782](http://arxiv.org/abs/2011.04782) A
[79] C. Chi, S. Feng, Y. Du, Z. Xu, E. Cousineau, B. Burchfiel, and S. Song,
“Diffusion policy: Visuomotor policy learning via action diffusion,” in
Proceedings of Robotics: Science and Systems (RSS), 2023. A12, A
[80] A. Prasad, K. Lin, J. Wu, L. Zhou, and J. Bohg, “Consistency
policy: Accelerated visuomotor policies via consistency distillation,”
inRobotics: Science and Systems (RSS), 2024. A
[81] J. Y. Gil and R. Kimmel, “Efficient dilation, erosion, opening, and
closing algorithms,”IEEE Transactions on Pattern Analysis and Ma-
chine Intelligence, vol. 24, no. 12, pp. 1606–1617, 2002. A
[82] B. Vu, T. Migimatsu, and J. Bohg, “Coast: Constraints and streams
for task and motion planning,” inIEEE International Conference on
Robotics and Automation (ICRA), 2024. A
[83] A. Curtis, X. Fang, L. P. Kaelbling, T. Lozano-P ́erez, and C. R.
Garrett, “Long-horizon manipulation of unknown objects via task and
motion planning with estimated affordances,” in2022 International
Conference on Robotics and Automation (ICRA). IEEE, 2022, pp.
1940–1946. A
[84] D. Driess, O. Oguz, J.-S. Ha, and M. Toussaint, “Deep visual heuris-
tics: Learning feasibility of mixed-integer programs for manipulation
planning,” in2020 IEEE international conference on robotics and
automation (ICRA). IEEE, 2020, pp. 9563–9569. A
[85] D. Driess, J.-S. Ha, R. Tedrake, and M. Toussaint, “Learning geometric
reasoning and control for long-horizon tasks from visual input,”
in2021 IEEE international conference on robotics and automation
(ICRA). IEEE, 2021, pp. 14 298–14 305. A
[86] M. Dalal, A. Mandlekar, C. R. Garrett, A. Handa, R. Salakhutdinov,
and D. Fox, “Imitating task and motion planning with visuomotor
transformers,” in7th Annual Conference on Robot Learning, 2023.
[Online]. Available: https://openreview.net/forum?id=QNPuJZyhFE
A
[87] C. Aeronautiques, A. Howe, C. Knoblock, I. D. McDermott, A. Ram,
M. Veloso, D. Weld, D. W. Sri, A. Barrett, D. Christiansonet al.,
“Pddl— the planning domain definition language,”Technical Report,
Tech. Rep., 1998. A
[88] B. Ames, A. Thackston, and G. Konidaris, “Learning symbolic repre-
sentations for planning with parameterized skills,” in2018 IEEE/RSJ
International Conference on Intelligent Robots and Systems (IROS).
IEEE, 2018, pp. 526–533. A
[89] G. Konidaris, L. P. Kaelbling, and T. Lozano-Perez, “From skills
to symbols: Learning symbolic representations for abstract high-level
planning,”Journal of Artificial Intelligence Research, vol. 61, pp. 215–
289, 2018. A
[90] T. Silver, R. Chitnis, J. Tenenbaum, L. P. Kaelbling, and T. Lozano-
P ́erez, “Learning symbolic operators for task and motion planning,”
in2021 IEEE/RSJ International Conference on Intelligent Robots and
Systems (IROS). IEEE, 2021, pp. 3182–3189. A
[91] Z. Wang, C. R. Garrett, L. P. Kaelbling, and T. Lozano-P ́erez,
“Learning compositional models of robot skills for task and motion
planning,”The International Journal of Robotics Research, vol. 40,
no. 6-7, pp. 866–894, 2021. A
[92] R. Chitnis, T. Silver, J. B. Tenenbaum, T. Lozano-Perez, and L. P.
Kaelbling, “Learning neuro-symbolic relational transition models for
bilevel planning,” in2022 IEEE/RSJ International Conference on
Intelligent Robots and Systems (IROS). IEEE, 2022, pp. 4166–4173.
A
[93] W. Huang, C. Wang, R. Zhang, Y. Li, J. Wu, and L. Fei-Fei, “Voxposer:
Composable 3d value maps for robotic manipulation with language
models,” in7th Annual Conference on Robot Learning, 2023. [Online].
Available: https://openreview.net/forum?id=98LF30mOC A

```
[94] A. Ahmetoglu, B. Celik, E. Oztop, and E. Ugur, “Discovering predic-
tive relational object symbols with symbolic attentive layers,”IEEE
Robotics and Automation Letters, 2024. A
[95] S. Kambhampati, K. Valmeekam, L. Guan, K. Stechly, M. Verma,
S. Bhambri, L. Saldyt, and A. Murthy, “Llms can’t plan, but
can help planning in llm-modulo frameworks,” arXiv preprint
arXiv:2402.01817, 2024. A
```

```
APPENDIX
```
```
Overview
```
The appendix provides additional details and results. First, we present detailed derivations for our planning objec-
tive (Appx. A) and sampling distributions (Appx. B). Second, we provide the details of the planning and optimiza-
tion (Appx. C). Third, we include extra experimental details (Appx. D to Appx. F). Finally, we provide information on
implementation (Appx. I), failure cases (Appx. J), hardware (Appx. K), generalization experiments (Appx. L), robustness to
noisy segmentations (Appx. M), additional related work (Appx. N), and detailed limitations (Appx. O). Qualitative results
are available at sites.google.com/stanford.edu/points2plans.

```
A Generative Model and Problem Formulation................................ A
```
```
B Approximate Sampling Distributions.................................... A
```
```
C Planning and Optimization Details..................................... A
```
```
D Predicates Definition............................................ A
```
```
E LLM Prompt Details............................................ A
```
```
F Dataset Generation and Training Details.................................. A
```
```
G Baseline Comparison Details........................................ A
```
```
H Primitives Definition............................................. A
```
```
I Neural Network Implementation Details.................................. A
```
```
J Failure Cases Analysis........................................... A
```
```
K Hardware Setup............................................... A
```
```
L Generalization to Unseen Scenarios.................................... A
```
```
M Generalization to Noisy Segmentation Masks............................... A
```
```
N Additional Related Work.......................................... A
```
```
O Detailed Limitations............................................. A
```

A. Generative Model and Problem Formulation

# 𝑙 𝒢

# 𝜙!

# 𝑥"#!

# 𝑜!

# 𝜙$ 𝜙"

# 𝑥! 𝑥$ 𝑥%

# 𝑎! 𝑎"

# ...

# 𝑎$ ...

Fig. 6:A causal Bayes net to derive Eq. 1.Grepresents the goal predicates,lis the language instruction,o 1 is the initial observation,
φ1:Hare the task plans,a1:Hare the continuous parameters, andx1:Hrepresent world states (including predicatesr1:Hand positions
p1:H. Shaded nodes represent observed variables.

Figure 6 shows a causal Bayes net defining the relevant variables to our planning problem. Recall that
a Bayesian network defines a factorization via conditional independence over the joint probability distribu-
tion of all variables in the model. In our model, the joint distribution is thus, p(G,o 1 ,l,ψ1:H,x1:H+1) =
p(l|G,o 1 )p(G |xH+1)p(o 1 |x 1 )p(x 1 )

```
QH
k=1p(xk+1|xk,ak,φk)p(ak|φk)p(φk).
```
Hereo 1 defines the observed world observation at time step 1. We define the observed language instruction asland the
unobserved goal predicates asG. Our model assumes that the user has a goal in mind that the robot is capable of achieving.
However, instead of explicitly writing this goal into a command prompt for the robot, the user provides a natural language
command. As such the robot must infer the underlying goal predicates conditioned on the language instruction and the world
observation shared between the user and robot.

The variableψ1:Hdefines the robot’s plan (sequence of primitives and continuous parameters), whilexkdefines the world
state at timek. We represent the state dynamics and action priors in the format common to that used in the planning as
inference literature e.g. [77,78].

While one could fully separate inference of the goal from the robot planning problem, by treating them as a joint problem
we ensure that the robot only infers goals that are feasible for it to achieve. This matches our assumption that the human
operator is non-adversarial and providing instructions the robot should be able to perform. The robot’s planning task is thus
to infer both the goal predicates,Gand planψ1:Hthat achieves this goal. While we are primarily concerned with finding
the plan, identifying the goal predicates provides a fixed target for our planner. This goal can also be helpful if we want
to replan online to correct for execution errors. As an alternative, we could examine marginalizing out the goal variable by
integrating over all possible goals. This would require us to use a form of planning to goal distributions [78], which we
defer to future work.

```
Given this model associated with the causal graph in Fig. 6, we can now derive the planning objective of Eq. 1 through
```

the following steps. (Note that Eq. 1-Eq. 8 appear in the main paper.)

```
arg max
ψ1:h,G
```
```
p(G,o 1 ,l,ψ1:H,x1:H+1) (9)
```
```
= arg max
ψ1:H,G
```
```
p(l|G,o 1 )p(G |xH+1)p(o 1 |x 1 )p(x 1 )
```
```
YH
```
```
k=
```
```
p(xk+1|xk,ak,φk)p(ak|φk)p(φk) (10)
```
```
= arg max
ψ1:H,G
```
```
p(l|G,o 1 )p(G |xH+1)p(o 1 |x 1 )p(x 1 )p(xH+1|x 1 ,ψ1:H)p(ψ1:H) (11)
```
```
= arg max
ψ1:H,G
```
```
p(l|G,o 1 )p(G |xH+1)p(x 1 |o 1 )p(o 1 )p(xH+1|x 1 ,ψ1:H)p(ψ1:H) (12)
```
```
= arg max
ψ1:H,G
```
```
p(l|G,o 1 )p(G |xH+1)p(xH+1|ψ1:H,o 1 )p(ψ1:H)p(o 1 ) (13)
```
```
= arg max
ψ1:H,G
```
```
p(l|G,o 1 )p(G |xH+1)p(xH+1|ψ1:H,o 1 ) (14)
```
```
= arg max
ψ1:H,G
```
```
p(l|G,o 1 )p(G |ψ1:H,o 1 ) (15)
```
The first step is simply applying the factorization of the Bayes net. Eq. 11 makes two changes to remove the product over
time steps. First we apply the definition p(ψ1:H) =

QH
k=1p(ak|φk)p(φk), which we name the plan prior. Second we
integrate over (i.e. marginalize out) state trajectories as a function of the actions sequence (i.e. plan) to encode the distribution
over terminal statesxH+1as a function of the initial state and plan. For Eq. 12 we then apply the definition of conditional
distributions to the initial state prior and observation function. This allows us to then marginalize out the initial state variable
x 1 in Eq. 13. The next step removes the prior on the initial observation as it is constant and assumes the prior on plans is
uniform and thus also constant. The final step then marginalizes out all possible terminal states to recover our problem as
stated in Eq. 1.

B. Approximate Sampling Distributions

We now turn to the derivation of our approximate sampling distributionsq 1 (φ1:H,G,|l,o 1 )and q 2 (a1:H|φ1:H,G,o 1 ).
We start by taking the first term in Eq. 1 and use Bayes’ theorem (Eq. 17) to condition on the observed language instruction
l

```
arg max
ψ1:H,G
```
```
p(l|G,o 1 )p(G |ψ1:H,o 1 ) (16)
```
```
= arg max
ψ1:H,G
```
```
p(G |l,o 1 )p(l|o 1 )
p(G |o 1 )
```
```
p(G |ψ1:H,o 1 ) (17)
```
```
= arg max
ψ1:H,G
```
```
p(G |l,o 1 )
p(G |o 1 )
```
```
p(G |ψ1:H,o 1 ) (18)
```
where we can simplify the numerator in Eq. 17 to Eq. 18 since the value oflis known and thusp(l|o 1 )is constant. We
now turn our attention to the second term in Eq. 1 and Eq. 18. Here we again use Bayes’ theorem and the definitions of


conditional probability distributions to rearrange terms from Eq. 19 through Eq. 21:

```
arg max
ψ1:H,G
```
```
p(G |l,o 1 )
p(G |o 1 )
```
```
p(G |ψ1:H,o 1 ) (19)
```
```
= arg max
ψ1:H,G
```
```
p(G |l,o 1 )
p(G |o 1 )
```
```
p(ψ1:H|G,o 1 )p(G |o 1 )
p(ψ1:H|o 1 )
```
```
(20)
```
```
= arg max
ψ1:H,G
```
```
p(G |l,o 1 )
p(ψ1:H|G,o 1 )
p(ψ1:H|o 1 )
```
```
(21)
```
```
= arg max
ψ1:H,G
```
```
p(G |l,o 1 )
p(a1:H,φ1:H|G,o 1 )
p(a1:H,φ1:H|o 1 )
```
```
(22)
```
```
= arg max
ψ1:H,G
```
```
p(G |l,o 1 )
p(φ1:H|G,o 1 )
p(φ1:H|o 1 )
```
```
p(a1:H|φ1:H,G,o 1 )
p(a1:H|φ1:H,o 1 )
```
```
(23)
```
```
= arg max
ψ1:H,G
```
```
p(G |l,o 1 )
p(φ1:H|G,l,o 1 )
p(φ1:H|o 1 )
```
```
p(a1:H|φ1:H,G,o 1 )
p(a1:H|φ1:H,o 1 )
```
```
(24)
```
```
= arg max
ψ1:H,G
```
```
p(G |l,o 1 )
p(φ1:H,G |l,o 1 )
p(G |l,o 1 )p(φ1:H|o 1 )
```
```
p(a1:H|φ1:H,G,o 1 )
p(a1:H|φ1:H,o 1 )
```
```
(25)
```
```
= arg max
ψ1:H,G
```
```
p(φ1:H,G |l,o 1 )
p(φ1:H|o 1 )
```
```
p(a1:H|φ1:H,G,o 1 )
p(a1:H|φ1:H,o 1 )
```
```
(26)
```
where Eq. 22 comes from applying the definition of the action components and Eq. 23 comes from the fact that skills
φ1:Hcan be chosen before selecting their parametersa1:H. The equality in Eq. 24 results from the existing conditional
independence relations allowing us to introducelwithout changing the values of the probability distribution. Eq. 25 results
from the definition of conditional distributions. Finally, this allows us to cancel the term appearing in both the numerator and
denominator resulting in Eq. 26. These two terms are then the distributions we approximate with our sampling distributions
q 1 (·)and q 2 (·).

```
Hence we can now summarize the derivation above as the following relationships
```
```
ψ∗1:H,G∗= arg max
ψ1:H,G
```
```
p(l|G,o 1 )p(G |ψ1:H,o 1 ) (27)
```
```
= arg max
ψ1:H,G
```
```
p(φ1:H,G |l,o 1 )
p(φ1:H|o 1 )
```
```
p(a1:H|φ1:H,G,o 1 )
p(a1:H|φ1:H,o 1 )
(28)
```
```
≈arg max
ψ1:H,G
```
```
q 1 (φ1:H,G |l,o 1 )q 2 (a1:H|φ1:H,G,o 1 ) (29)
```
where the approximation in Eq. 29 is made by assuming a uniform distribution over actions given the initial observation (o.e.
the denominator becomes constant). We can then approximately solve the planning objective in Eq. 1 by sequentially
generating samples from the two approximate sampling distributions asφ ̃1:H,G ̃∼q 1 (φ1:H,G |l,o 1 )followed bya ̃1:H∼

```
q 2
```
```

a1:H
φ ̃1:H,G, ̃ o 1
```
```

.
```
C. Planning and Optimization Details

We use a shooting-based planner to determine the actionsψ1:Hfrom an initial observationo 1 and a language instruction
l. We provide the details of the shooting-based Points2Plans planner in Alg. 1.

Note that if no plan predicted by the LLM is successful or constraint-satisfying, we fall back to a search-based strategy
that enumerates all possible primitive sequences up to a specified search depth (as in [13]), optimizes them with Eq. 3, and
checks if the plan satisfies any goal predicted by the LLM. (Note that Eq. 1-Eq. 8 appear in the main paper.) In practice,
we find that the planner seldom falls back to the search-based strategy; however, it ensures that more primitive sequences
will be tested should the LLM fail to produce a correct plan.

Furthermore, we provide more details about the constrained optimization (Eq. 3). The optimization process includes
encoding the point clouds into the latent states (Eq. 5), the delta-state predictions by the delta-dynamics(Eq. 6),
decoding thedelta-statein latent space to relative pose changes with the pose decoderDecp(Eq. 7) , and point cloud
transformations (Eq. 8).


Algorithm 1Shooting-based Points2Plans planner

```
1:globals:LLM,Enc,Decgr,Deccr,T,ω, q 2 (a1:H|φ1:H,G,o 1 )
2:functionSHOOTING(l,o 1 ,C)
3: {φ ̃i1:Hi}Ni=1,{G ̃i}Ni=1∼LLM(l,o 1 ) ▷Generate task plans and goals
4: fori= 1...Ndo
5: C={} ▷Initiate candidate set for each task plan and goal
6: { ̃aj1:Hi}Kj=1∼q 2
```
```

̃a1:Hi
```
(^) φ ̃i
1:Hi,G ̃i,o 1

▷Sample actions
7: forj= 1...Kdo
8: z 1 =Enc(o 1 ) ▷Encode initial observation
9: zj 1 =z 1
10: oj 1 =o 1
11: fort= 1...Hido
12: δzjt=T

zjt,⟨φ ̃t, ̃ajt⟩

▷Delta-dynamicsfunction
13: δpjt=Decp

δzjt

14: ojt+1=ω(δpjt)ojt ▷Point clouds transformations
15: zjt+1=Enc

ojt+

▷Encode transformed point clouds
16: ifDeccr

zjt+

>=εthen
17: raisecollision found, break ▷Collision found, reject this sequence ̃aj1:Hi
18: end if
19: ift==Hthen ▷No collision found
20: C←C∪{j} ▷Add this sequence to candidate set
21: end if
22: end for
23: end for
24: ifC! =∅then
25: j∗= arg maxj∈C
Q
g∈G ̃iDec
gr(zj
Hi+1)
26: returnφ ̃i1:Hi ̃aj
∗
1:Hi ▷Return the task plan with the continuous parameters
27: end if
28: end for
29: raiseLLM failure, fall back to Search
30:end function
D. Predicates Definition
Our system includes unary predicates and binary predicates.
For unary predicates, our system encodes whether a segment is movable (e.g., a shelf in a cupboard is not movable while
an object on the shelf is movable), whether a segment is a drawer, and whether the drawer is opened or closed.
For binary predicates, our system includes two kinds. First, our system includes nine spatial predicates:left, right, front,
behind, above, below, contact, boundary,andinside. The definitions of these predicates are the same as [65]. Second, we
define feasibility-related predicates to indicate the feasibility of each object. We define two feasibility-related predicates:
blocking-behindandblocking-inside. We defineblocking-behind(a, b) as true if behind(b, a), below(a, high-surface), below(b,
high-surface), above(a, low-surface), and above(b, low-surface), meaning both a and b are in a constrained environment, and
b is behind a. We defineblocking-inside(a, b) as true if inside(b, a), below(a, high-surface), below(b, high-surface), above(a,
low-surface), above(b, low-surface), meaning both a and b are in a constrained environment and b is inside a.
E. LLM Prompt Details
Our prompts include prompt templates (black), LLM output (orange), and in-context examples (grey). Placeholders,
denoted by braces, are substituted with task-related objects for different scenarios.
The in-context examples are toy examples of the tasks that the LLM solves at test time. These toy examples describe the
usage semantics of the available primitives and predicates, and help constrain the LLM output.


I am the reasoning system of a mobile manipulator robot operating in a household
environment. Given 1) an instruction from a human user and 2) the current symbolic state
of the environment, I will predict a set of possible symbolic goals that the robot could
achieve to fulfill the user’s instruction.

Definitions:

- Symbolic states and symbolic goals are defined as a set of predicates expressed over
specific objects.
- The term ’predicate’ refers to an object state (e.g., Opened(cabinet)) or a relationship
among objects (e.g., On(cup, shelf)).

The robot can perceive the following information about the environment: - The objects in
the environment - The states of individual objects - The relationships among objects

The robot can detect the following states of individual objects: - Opened(a): Object a is
opened - Closed(a): Object a is closed

The robot can detect the following relationships among objects: - On(a, b): Object a is on
object b - Inside(a, b): Object a is in object b

There may be multiple symbolic goals that fulfill the user’s instruction. Therefore, I will
format my output in the following form:

Goals: List[List[str]]

Rules: - I will output a set of symbolic goals as a list of lists after ’Goals:’. Each
nested list represents one goal - I will not output all possible symbolic goals, but the
most likely goals that fulfill the user’s instruction - If there are multiple symbolic goals
that fulfill the instruction, I will output the simplest goals first"


I am the task planning system of a mobile manipulator robot operating in a household
environment. Given 1) an instruction from a human user, 2) the current symbolic state of the
environment, and 3) a set of possible symbolic goals that the robot could achieve to fulfill
the user’s instruction, I will predict a set of task plans that the robot should execute to
satisfy the symbolic goals.

Definitions:

- Symbolic states and symbolic goals are defined as a set of predicates expressed over
specific objects.
- The term ’predicate’ refers to an object state (e.g., Opened(cabinet)) or a relationship
among objects (e.g., On(cup, shelf)).
- A task plan is a sequence of actions that the robot can execute (e.g., Pick(cup, table),
Place(cup, shelf))

The robot can perceive the following information about the environment:

- The objects in the environment
- The states of individual objects
- The relationships among objects

The robot can detect the following states of individual objects:

- Opened(a): Object a is opened - Closed(a): Object a is closed

The robot can detect the following relationships among objects:

- On(a, b): Object a is on object b - Inside(a, b): Object a is in object b

The robot can execute the following actions:

- Pick(a, b): The robot picks object a from object b
- Place(a, b): The robot places object a on or in object b
- Open(a): The robot opens object a
- Close(a): The robot closes object a

Action preconditions:

- If the robot is already holding an object, it CANNOT Pick, Open, or Close another object
- The robot CAN ONLY Place an object that it is already holding

There may be multiple symbolic goals that fulfill the user’s instruction. Therefore, I will
format my output in the following form:

Plans: List[List[str]]

Rules:

- I will output a set of task plans as a list of lists after ’Plans:’. Each nested list
represents one task plan
- I will output one task plan for each symbolic goal. Hence, each goal and its corresponding
plan will be located at the same index in the ’Goals’ and ’Plans’ lists
- I will only output task plans that are feasible with respect to the defined action
preconditions.


```
Instructions: Put all the objects on the shelf
```
Objects: [’{object 1}’, ’{object 2}’, ’{object 3}’, ’ground’, ’shelf’]

Predicates: [’On({object 1}, ground’, ’On({object 2}, ground’, ’On({object 3}, ground’]

Goals: [’On({object 1}, shelf’, ’On({object 2}, shelf’, ’On({object 3}, shelf’]

Plans: [’Pick({object 1}, ground)’, ’Place({object 1}, shelf)’, ’Pick({object 2}, ground)’,
’Place({object 2}, shelf)’, ’Pick({object 3}, ground)’, ’Place({object 3}, shelf)’]

```
Instructions: Retrieve object 1.
```
Objects: [’{object 1}’, ’{object 2}’, ’{object 3}’, ’{object 4}’, ’ground’, ’shelf’]

Predicates: [’On({object 1}, shelf’, ’On({object 2}, shelf’, ’On({object 3}, shelf’,
’On({object 4}, shelf’, ’Blocking({object 3}, {object 4})’, ’Blocking({object 3},{object
1 })’, ’Blocking({object 2}, {object 1})’, ’Blocking({object 4},{object 1})’]

Goals: [ ’{On(object 2}, ground’, ’{On(object 3}, ground’, ’{On(object 4}, ground’,
’{On(object 1}, ground’]

Plans: [’Pick({object 3}, shelf)’, ’Place({object 3}, ground)’, ’Pick({object 2}, shelf)’,
’Place({object 2}, ground)’, ’Pick({object 4}, shelf)’, ’Place({object 4}, ground)’,
’Pick({object 1}, shelf)’, ’Place({object 1}, ground)’]

```
In-context Examples:
```
Instructions: Put object 1 on the sink.

Objects: [’object 1’, ’sink’, ’kitchen table’]

Predicates: [’On(object 1, kitchen table’]

Goals: [’On(object 1, sink’]

Plans: [’Pick(object 1, kitchen table)’, ’Place(object 1, sink)’]

Instructions: Get me object 1 from the drawer. I’m in the bedroom. Don’t leave the drawer
open.

Objects: [’object 1’, ’object 2’, ’drawer’, ’closet’, ’bed’]

Predicates: [’Inside(object 1, drawer)’, ’Inside(object 2, closet)’, ’Closed(drawer)’,
’Closed(closet)’]

Goals: [’On(object 1, bed)’, ’Closed(drawer)’]

Plans: [’Open(drawer)’, ’Pick(object 1, drawer)’, ’Place(object 1, bed)’, ’Close(drawer)’]


```
Instructions: Bring me object 2. I’m sitting on the reading chair by the coffee table.
```
```
Objects: [’object 1’, ’object 2’, ’bookshelf’, ’reading chair’, ’coffee table’]
```
```
Predicates: [’On(object 1, object 2’, ’On(object 2, bookshelf’]
```
```
Goals: [[’On(object 2, coffee table)’], [’On(object 2, reading chair)’]]
```
```
Plans: [ [’Pick(object 1, object 2)’, ’Place(object 1, bookshelf)’, ’Pick(object 2,
bookshelf)’, ’Place(object 2, coffee table)’], [’Pick(object 1, object 2)’, ’Place(object
1, bookshelf)’, ’Pick(object 2, bookshelf)’, ’Place(object 2, reading chair)’] ]
```
```
Instructions: Please retrieve object 1.
```
```
Objects: [’object 1’, ’object 2’, ’shelf’, ’ground’]
```
```
Predicates: [’On(object 1, shelf’, ’On(object 2, shelf’]
```
```
Goals: [[’On(object 1, shelf)’], [’On(object 2, shelf) ’]]
```
```
Plans: [ [’Pick(object 1, shelf)’, ’Place(object 1, ground)’], [’Pick(object 2, shelf)’,
’Place(object 2, ground)’, ’Pick(object 1, shelf)’, ’Place(object 1, ground)’, ] ]
```
1) Connections between RD Models and LLMs:We have several connections as the interface between RD models and
LLMs. First, given the predicatesabove(A, B) andcontact(A, B), thenon(A, B) holds true, and vice versa. Second, given
the plans as pick(A, D) and place(A, C) from LLMs, the RD models will receive this plan as pick-and-place(A,C).

F. Dataset Generation and Training Details

We generate the training datasets in the IsaacGym [75] simulator. First, we generate a variable number of randomized
objects (size and pose) and save the object pose, segmented point cloud, and predicate as(ot,ˆrt,pˆt). Then we randomly
execute a primitive in the simulator and save the primitiveψt. We teleport the objects to model the effects of each primitive.
After the primitive execution, we record the post-action scene as(ot+1,ˆrt+1,pˆt+1). The dataset contains more than 36,
primitive executions. We show several single-step simulation executions in Fig. 7.

We set up one camera in simulation to generate the segmented point clouds. Due to the generalization ability of our RD
framework to different view points, we can position the real-world camera at different view angles, as long as the object
point clouds are within a suitable range to ensure decent quality. For example, we use the Realsense D435 camera for
real-world experiments, with an ideal range of 0.3m to 3m. Note that we focus on the critical segments of environmental
point clouds (e.g., horizon surfaces for the cupboard and drawers for the table), as not all segments are visible due to the
partial-view nature of the input point clouds.

We define loss functions with four terms for each transition(ot,ˆrt,pˆt,ψt,ot+1,ˆrt+1,pˆt+1)in the training datasets. First,
we obtainzt=Enc(ot)andzt+1=Enc(ot+1). To enable our framework to detect the current predicates, we define the
cross-entropy loss between the currently detected predicates and the ground-truth predicates:Lcp=CE(Decr(zt),ˆrt) +
CE(Decr(zt+1),ˆrt+1). Second, to enable the model to accurately predict the change of pose, we define the second loss
term asLpos=a·

p
b·||δpt−(ˆpt+1−pˆt)||. We use two parameters,a,b, to balanceLposwith other loss terms likeLpd.
In practice, we use a = 5 and b = 12. Third, to regularize the latent states, we first obtain the predicted latent states asz′t+1=
zt+δzt. We define the regularization loss term asLreg=||zt+1−z′t+1||^22. Fourth, to predict the future predicates, we define
a cross-entropy loss between the predicted predicates and the ground-truth predicates asLfp=CE(Decqr

```

z′t+
```

,ˆrt+1).
We train our framework end-to-end with the sum of these four loss terms asL=Lcp+Lpos+Lreg+Lfpusing the
Adam optimizer with a learning rate of 1 e− 4. We only train our framework with single-step transitions while it can solve
long-horizon planning problems in a composable way.


```
Initial
Scene
```
```
Post-action
Scene
```
Fig. 7:We show several examples of single-step primitive executions in simulation. The first two columns show examples of drawers,
while the next two columns show examples of the constrained cupboard with different numbers of objects.

G. Baseline Comparison Details

We show the details of how baselines fail in the “constrained packing” task and the “constrained retrieval” task in Fig. 8.
Please refer to the supplemental video for the demos.

H. Primitives Definition

We use three primitives in this paper: pick-and-place, pick-and-toss, and open/close. Since we have a mobile manipulator,
we separate the movements of the mobile base and the Kinova arm. Based on the objects to manipulate, we first move the
mobile base to a reachable space for the arm to manipulate the objects. Then, we run the arm planner to manipulate the
objects.

Pick-and-place is defined as first grasping the object and then placing it on the supporting surface. For the grasp, we use
the point cloud center of each segment to generate the grasps. We grasp the center of the objects except for large objects.
For large objects, we use a heuristics offset. For example, we grasp the side of a bowl instead of the center.

For the placement, we generate a placement height based on the surface height plus a height offset. For the toss, we first
move the base to a position at a fixed distance from the target, then execute the predefined toss trajectory.

For the open/close actions, we first determine the handle center using the segmented point clouds. Then we move the
robot to the pre-open/close position. After this, the arm executes the motion with continuous parameters encoding how much
the drawer will open or close.

I. Neural Network Implementation Details

Our RD model is composed of three components: an encoderEnc, a transformer-based dynamics modelT, and a decoder
Dec. We describe the details of each component below.

Encoder:We first use the farthest point sampling method to downsample each point cloud to 128 points. Based on the
input as segmented point cloudsot=o^1 t,...,oMt at timestept, we first use a PointConv [72] to get per object features as
Pti=PointConv(oit). The PointConv model we use incorporates three set abstraction layers. Each abstraction layer receives
input points data and input points position data. The output from each layer consists of sampled points position data and


```
Points2Plans
```
```
Points2Plans
```
```
Points2Plans−Delta
```
```
Points2Plans−Feasibility
```
Fig. 8:We show two failure cases of baselines. The first two rows demonstrate that Points2Plans succeeds while Points2Plans−Delta
fails in the “constrained packing” task. The next two rows show that Points2Plans succeeds while Points2Plans−Feasibility fails in the
“constrained retrieval” task.

sampled points feature data, with the input and output points position data having 3 channels. The first set abstraction layer
has 128 points with 8 samples, utilizing a bandwidth of 0.1. It employs an MLP with 3+3 input channels, 32 output channels,
and a kernel size of 1. The second layer has 64 points with 16 samples and a bandwidth of 0.2, using an MLP with 32+
input channels, 64 output channels, and a kernel size of 1. The third layer is a groupall layer, generating 128-dimension
features per segment with a bandwidth of 0.4, and utilizes an MLP with 64+3 input channels, 128 output channels, and a
kernel size of 1.

Then we concatenate per object feature with the positional embedding of each object in PyTorch [73] aszti=Pti⊕Ii
whereIi=Embpos(i). Each positional embedding has 128-dimension features. Next, we combine the per object latent into
an object-centric latent statezt=z^1 t,...,zMt , where eachztihas 256 features.

Dynamics:Thedelta-dynamicsmodelTtakes the input asztandψt=⟨φt,at⟩.φtincludes skill idsi, manipulated
obj idmi, and placement surface idpi. For eachsi, we use a different dynamics modelTsiwith a transformer. For the
transformers, we utilize 2 sub-encoder layers, 2 heads in the multi-head attention models, and an input/output model size
of 256.

We encode eachatwith an action encoder (MLPsi) asamt =Imi⊕MLPsi(at), whereImiencodes which object
this primitive will operate on. We further use the placement id to represent which surface to place the object on, as
apt =Ipi⊕MLPsi(at). If there is no surface to place, for example, in an open drawer action, we use zero embeddings
forIpi. The action encoder is a two-layer MLP, with each layer containing 128 neurons. Then the dynamics modelTsi
takes the input as M+2 tokenszt^1 ,...,zMt ,amt,apt. We discard the action tokens at the output head and obtain the output
δzt=δz^1 t,...,δztM

```
Decoder:Based on the latent statezt, we use different MLPs for different output heads. First, we use one MLP for unary
```

predicate prediction:rut=Decur(zt). Second, we use one MLP for constrained binary predicates prediction:rct=Deccr(zt).
Third, we use one MLP for spatial binary predicate prediction:rst=Decsr(zt).

ForDecur, we use a two-layer MLP with a hidden layer of 64 neurons. The output contains 3 bits. The first bit encodes
whether the segment is a shelf or an object. The second and the third bits encode whether the segment is a drawer and
whether the drawer is open, respectively. ForDeccr, we use a three-layer MLP and each hidden layer contains 64 neurons.
The output contains 2 bits representingblocking-behindandblocking-inside. ForDecsr, we use a three-layer MLP with each
hidden layer containing 64 neurons. The output contains 9 bits representing 9 different spatial predicates.

The pose decoderDecptakes as input adelta-statein the latent spaceδztand predicts the relative pose change of all
objects in the scene asδpt=δp^1 t,...,δpMt =Decp(δzt). Hence, this decoder can only be applied ondelta-statepredicted
byT.

ForDecp, we use a two-layer MLP with one hidden layer containing 64 neurons. The output head contains 2 bits
representingδx,δy. Forz, we encode this parameter as part of our discrete parameter for the supporting surface, as shown
in the primitive definitions in Sec. H. LLMs will generate it as part of the task plan.

For the MLPs, we use Sigmoid in the output head of predicate decodersDecur,Deccr,andDecsrsince these decoders
output binary variables. For all other MLPs, we use ReLU as the activation function.

J. Failure Cases Analysis

##### Points2Plans Failure Case 1

##### Points2Plans Failure Case 2

### ...

### ...

Fig. 9:Two failure cases of Points2Plans. The first row shows a failure case due to an unstable placement in the “constrained packing”
task. The second row demonstrates that Points2Plans fails in the “constrained retrieval” task because of a failed grasp.

We show two failure cases of Points2Plans in Fig. 9. These failure cases are caused by primitive execution failures. They
highlight the limitations of open-loop execution in Points2Plans and motivate the incorporation of closed-loop policies [79,80]
in future work. Please refer to our website (sites.google.com/stanford.edu/points2plans) for detailed executions of these two
failure cases.

K. Hardware Setup

The models are trained on a standard workstation with a single GPU (NVIDIA GeForce RTX 3090 Ti, 24 GB). All real-
world experiments are conducted on a mobile platform with a custom mobile base and a Kinova arm. We use a RealSense
D435 camera for perception in the real world.


L. Generalization to Unseen Scenarios

```
Training Training Training Unseen Poses
```
```
Unseen Env Unseen Sizes Unseen View Unseen Shapes
```
```
Fig. 10:Examples of our training dataset and some test dataset with unseen poses, environments, sizes, view, and shapes.
```
```
TABLE I:Generalization to Unseen Scenarios
Method Points2Plans Points2Plans−Delta
Unseen sizes of objects 58% 33%
Unseen camera view angles 61% 42%
Unseen environments 50% 33%
Unseen shapes of objects (YCB objects) 58% 49%
Unseen poses of objects 69% 51%
```
We show the extra simulation success rates of Points2Plans and the best-performing baseline to demonstrate the model’s
generalization ability to unseen camera viewpoints, environments, and different sizes, shapes, and poses of objects. We run
100 trials per approach per generalization metric in the constrained packing task.

From the results shown in the table. I, we find Points2Plans performs well when it generalizes to unseen scenarios and
outperforms the baseline. Please refer to Fig. 10 for the visualizations of comparisons between training datasets and test
datasets with novel scenes.

M. Generalization to Noisy Segmentation Masks

```
Ground Truth Small Noise Large Noise
Fig. 11:Examples of ground truth segmentation mask and noisy segmentation masks.
```
We show our method’s and the best-performing baseline’s robustness to the noise in the segmentation mask. We use the
erosion and dilation algorithm [81] to generate the noise for the segmentation masks. We use kernel size = 5 for generating


```
TABLE II:Robustness to Noisy Segmentation Masks
Method Points2Plans Points2Plans−Delta
No Noise 81% 69%
Small Noise 78% 67%
Large Noise 73% 39%
```
small noise and kernel size = 10 for generating large noise to segmentation masks. We run 100 trials per approach per noise
metric in the constrained packing task.

From the results shown in the table. II, we find Points2Plans performs well in the robustness to the noise for the
segmentation mask while the baseline performs poorly, especially with large noise. Please refer to Fig. 11 for the details
and the visualizations of noisy segmentation masks.

N. Additional Related Work

Task and motion planningsolves long-horizon tasks through symbolic and geometric reasoning [3–5,82]. Perception
modules can be used to alleviate TAMP’s assumption on full state observability [83]. Other works learn vision-based planning
heuristics [71,84] or behavior policies [85,86] from pre-computed TAMP solutions. We highlight two distinctions of our
approach compared to TAMP: our system a) predicts symbolic effects instead of using predefined symbolic operators in
e.g., PDDL [87] and b) plans in a latent space directly encoded from 3D partial-view point clouds. Our approach is related
to learning symbolic operators [66,88–91] and object dynamics [9,92] for long-horizon planning, but differs in the use of
our RD model, which captures both symbolic and geometric effects of actions in a shared latent space.

O. Detailed Limitations

Points2Plans executes its plans in an open-loop fashion, i.e., without considering feedback from the environment. Exploring
closed-loop strategies for refining plans or correcting execution failures via replanning are possible points of extension. Our
relational dynamics model currently only predicts object positions when rolling out a plan, which is sufficient for tasks
involving simple (e.g., symmetric) object geometries. Scaling Points2Plans to more complex object geometries necessitates
the prediction of their full pose. We also assume access to a set of hand-designed manipulation primitives. Interfacing with
closed-loop policies [79,93] might allow Points2Plans to solve tasks that require more fine-grained motions. Furthermore, our
framework assumes a fixed set of predicates to learn skill effects. Drawing from predicate learning techniques [67,94] could
improve the generality of our framework, e.g., facilitating planning in an open-world setting. Finally, while we demonstrate
faster planning with LLMs, their task planning performance degrades with longer tasks and more complex instructions [95].
Including more sophisticated LLM-based task planning strategies [11] would improve the overall robustness of our planner.


