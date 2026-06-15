// ─── Document Settings ───────────────────────────────────────────────────────
#set document(
  title: "Autonomous Vehicle Lane Following" +
         " with Obstacle Avoidance",
  author: (
    "João Ferreira",
    "Henrique Teixeira",
    "Miguel Almeida",
  ),
)

#set page(
  paper: "a4",
  margin: (
    x: 1.5cm,
    y: 2cm,
  ),
  numbering: "1",
  header: context {
    if counter(page).get().first() > 1 [
      #set text(size: 8pt, fill: luma(130))
      #grid(
        columns: (1fr, 1fr),
        align(left)[
          Lane Following with Obstacle Avoidance
        ],
        align(right)[
          Group A
        ],
      )
      #line(
        length: 100%,
        stroke: 0.4pt + luma(180),
      )
    ]
  }
)

#set text(
  font: "New Computer Modern",
  size: 9pt,
  lang: "en",
)
#set par(justify: true, leading: 0.55em)
#set heading(numbering: "1.")

#show heading.where(level: 1): it => {
  set text(size: 9pt, weight: "bold")
  set align(center)
  v(0.8em)
  upper(it)
  v(0.3em)
}
#show heading.where(level: 2): it => {
  set text(size: 9pt, style: "italic")
  v(0.5em)
  it
  v(0.2em)
}

// ─── Title Block ─────────────────────────────────────────────────────────────
#align(center)[
  #v(0.5cm)
  #text(size: 20pt, weight: "bold")[
    Autonomous Vehicle Lane Following \
    with Obstacle Avoidance
  ]
  #v(0.5cm)
  #grid(
    columns: (1fr, 1fr, 1fr),
    gutter: 0.3cm,
    align(center)[
      #text(size: 10pt, weight: "bold")[
        João Ferreira
      ] \
      #text(size: 8pt, fill: luma(60))[
        Faculty of Engineering \
        University of Porto \
        up202306717
      ]
    ],
    align(center)[
      #text(size: 10pt, weight: "bold")[
        Henrique Teixeira
      ] \
      #text(size: 8pt, fill: luma(60))[
        Faculty of Engineering \
        University of Porto \
        up202306640
      ]
    ],
    align(center)[
      #text(size: 10pt, weight: "bold")[
        Miguel Almeida
      ] \
      #text(size: 8pt, fill: luma(60))[
        Faculty of Engineering \
        University of Porto \
        up202303926
      ]
    ],
  )
  #v(0.5cm)
]

// ─── Two-column body ─────────────────────────────────────────────────────────
#show: rest => columns(2, rest)

// ─── Abstract ────────────────────────────────────────────────────────────────
#align(center)[
  #text(size: 9pt, weight: "bold")[Abstract]
]

#text(size: 8.5pt)[
*_This paper presents the design and
evaluation of a Reinforcement Learning
(RL) agent for autonomous lane following
with obstacle avoidance in a simulated
environment. The agent operates an
E-puck robot in Webots, processing
RGB camera frames and LiDAR point
clouds to follow a yellow centre line
while avoiding static and dynamic
obstacles. We compare two action
spaces --- discrete (DQN) and continuous
(PPO) --- against two reward paradigms
--- dense and sparse --- and further
assess camera robustness under varying
field-of-view distances and visual
distortion filters. Our hypothesis is
that continuous control paired with a
dense reward signal will yield the best
sample efficiency and driving
performance, while adversarial visual
training will produce a more robust
policy. Results and conclusions are
presented after completing the full
experimental campaign._*
]

#v(0.4em)
*Index Terms* --- Reinforcement
Learning, Autonomous Vehicles, Lane
Following, Obstacle Avoidance, DQN,
PPO, Webots, Curriculum Learning.

// ─── I. Introduction ─────────────────────────────────────────────────────────
= Introduction

Autonomous driving has emerged as one
of the most challenging and impactful
problems in modern robotics and
artificial intelligence. While
rule-based and classical control
approaches dominated the early
literature, deep learning and,
more recently, deep Reinforcement
Learning (RL) have demonstrated
compelling results in end-to-end
driving pipelines --- learning directly
from sensor observations to motor
commands without hand-crafted
intermediate representations.

A fundamental sub-problem within
autonomous driving is *lane following*:
keeping the vehicle centred on a
designated lane while maintaining
safe forward progress. When combined
with *obstacle avoidance* --- the ability
to detect and react to static objects
such as barrels and dynamic agents such
as pedestrians or other vehicles ---
the task becomes a rich testbed for
evaluating RL algorithms, reward
function design, and sensory robustness.

This work trains and evaluates an RL
agent in the Webots robot simulator
using an E-puck platform. The agent
receives RGB camera frames and LiDAR
readings as observations and must
learn to stay centred on a yellow
centre line while avoiding collisions.
Three controlled experiments examine:
(i) the effect of action space
discretisation, (ii) the impact of
reward function density, and (iii)
camera robustness under field-of-view
variation and visual distortion.

The remainder of this paper is
organised as follows. Section II
surveys related work. Section III
describes the experimental setup and
algorithms. Section IV defines the
reward functions. Section V outlines
the evaluation methodology.
Section VI presents results.
Section VII concludes.

// ─── II. State of the Art ────────────────────────────────────────────────────
= State of the Art

== End-to-End Learning for Driving

The seminal work of Pomerleau (1989)
with the ALVINN network established
the viability of neural networks for
lane keeping from camera input.
Modern successors such as the NVIDIA
PilotNet @bojarski2016end demonstrated
that a CNN trained via imitation
learning can steer a real vehicle
using only a front-facing camera,
achieving highway lane following
without explicit perception modules.

End-to-end RL approaches remove the
need for expert demonstrations
entirely. @kendall2019learning showed
that a model-free RL agent could learn
to drive in simulation from raw pixels,
motivating the pixel-to-action paradigm
we adopt here.

== Deep Q-Networks and Variants

Mnih et al. @mnih2015humanlevel
introduced the Deep Q-Network (DQN),
combining Q-learning with a CNN
function approximator, experience
replay, and a target network to
stabilise training. DQN achieved
human-level performance across Atari
games and has since been applied to
discrete-action driving tasks.
Extensions such as Double DQN,
Dueling DQN, and Prioritised
Experience Replay further improve
sample efficiency in sparse-reward
settings.

== Proximal Policy Optimisation

For continuous action spaces,
policy-gradient methods are preferred.
Schulman et al. @schulman2017proximal
proposed PPO, a first-order algorithm
that enforces a trust-region constraint
via a clipped surrogate objective.
PPO has become the de-facto standard
for continuous robotic control owing
to its stability, ease of tuning, and
strong empirical performance. It has
been applied successfully to
autonomous driving in both simulation
@highway-env and real platforms.

== Multi-Sensor Fusion

Real autonomous systems fuse camera
and LiDAR data to overcome the
limitations of each modality.
Camera-only systems suffer under poor
lighting; LiDAR-only systems lack
semantic information. Chen et
al. @chen2017multiview showed that
fusing both streams outperforms
unimodal baselines in object detection.
Our architecture mirrors this by
concatenating CNN-extracted visual
features with a LiDAR reading vector
before the policy head.

== Reward Shaping and Curriculum Learning

Reward function design critically
affects convergence speed and final
policy quality @ng1999policy.
Dense, shaped rewards provide a
learning signal at every timestep,
while sparse rewards require the agent
to solve the credit-assignment problem
across entire episodes. Curriculum
learning @bengio2009curriculum
mitigates sparse-reward difficulty by
progressively increasing task
complexity --- a strategy we adopt for
our sparse-reward baseline.

== Sim-to-Real and Robustness

Domain randomisation @tobin2017domain
and input augmentation improve the
transferability of policies trained
in simulation to the real world by
exposing the agent to a wide
distribution of visual conditions
during training. Distortion filters
(fog, rain, low-light) and FOV
variation are practical instantiations
of this principle that we investigate
in Experiment 3.

// ─── III. Proposed Approach ──────────────────────────────────────────────────
= Proposed Approach

== Simulation Environment

All experiments run in *Webots 2023b*
using an *E-puck* robot on a custom
looped track with a yellow centre line.
The E-puck is equipped with:

- *Camera*: 64 × 64 RGB for line
  detection.
- *LiDAR*: 360° point cloud / 1-D
  radial scan for proximity sensing.
- *Actuators*: differential-drive
  motors via velocity commands.

Obstacles --- barrels, pedestrians,
and vehicles --- are spawned
procedurally at random intervals so
the agent cannot memorise a fixed
layout. Each object is described by
its centroid and type-specific
parameters (barrel radius, pedestrian
radius, vehicle bounding rectangle).
A *Gymnasium* wrapper exposes a
standard `step` / `reset` API to
the RL training loop.

== Observation Space

At each timestep the agent observes:

+ A 64 × 64 × 3 RGB camera frame.
+ A 1-D LiDAR vector of range
  readings.
+ Current forward velocity $v$.
+ Alignment angle $theta$ of the
  vehicle heading relative to the
  yellow centre line.

The camera frame is processed by a
shared CNN backbone; its feature map
is flattened and concatenated with
the numerical vector $(v, theta,
bold(l))$, where $bold(l)$ is the
LiDAR reading, before the policy or
value head.

== Algorithms

*DQN (discrete):* A four-action
policy --- Left, Right, Straight,
Brake --- implemented with experience
replay ($|cal(D)|=10^5$), a target
network updated every 1000 steps,
and $epsilon$-greedy exploration
with linear decay.

*PPO (continuous):* Steering
$phi in [-1,1]$ and throttle
$a in [-1,1]$ are output as a
diagonal Gaussian. We use a
clip parameter $epsilon=0.2$,
GAE-$lambda=0.95$, and normalised
advantages.

Both algorithms use the same CNN
backbone: three convolutional layers
(32, 64, 64 filters; kernels 8×8,
4×4, 3×3; stride 4, 2, 1) followed
by a 512-unit fully-connected layer,
matching the architecture of @mnih2015humanlevel.

// ─── IV. Reward Functions ────────────────────────────────────────────────────
= Reward Functions

== Dense Reward

$R = w_1 V cos theta
    - w_2 |d|
    - w_3 C$

where $d$ is lateral distance from
the centre line and $C$ is a large
constant penalty (e.g. $-100$)
triggered when LiDAR proximity falls
below a collision threshold.

== Time-to-Collision (TTC) Reward

$R = w_1 V cos theta
    - w_2 d^2
    - w_3 max(0, 1 - D_min/D_"safe")^2
    - w_4 C_"term"$

The quadratic safety term is zero
beyond $D_"safe"$ and grows as the
agent approaches an obstacle,
encouraging proactive evasion. A
terminal penalty $C_"term"$ ends
the episode on actual collision.

== Sparse Reward

$R = cases(
  +1 & "checkpoint / lap",
   -1 & "collision",
   0 & "otherwise"
)$

Because a raw sparse signal rarely
converges, curriculum learning is
applied in three stages:

+ *Stage 1* --- No obstacles;
  agent learns basic lap completion.
+ *Stage 2* --- Static obstacles
  (barrels) introduced.
+ *Stage 3* --- Dynamic obstacles
  (pedestrians, vehicles) added.

// ─── V. Empirical Evaluation ─────────────────────────────────────────────────
= Empirical Evaluation

Each experiment is repeated *100
times*; all metrics are averaged
across runs.

== Evaluation Metrics

#table(
  columns: (auto, 1fr),
  inset: 0.4em,
  stroke: 0.3pt + luma(180),
  [*Metric*], [*Description*],
  [Collisions],
  [Total collisions per run.],
  [Success Rate],
  [% of runs without collision.],
  [Cross-Track Error],
  [Mean lateral deviation (m).],
  [Mean Lap Time],
  [Avg. circuit time (s).],
  [Safety Score],
  [Distance / near-misses.],
)

== Experiment 1 --- Action Space

*Goal:* Compare discrete (DQN) vs.
continuous (PPO) control.

*Measured:* Sample efficiency
(reward vs. training steps),
trajectory smoothness, and
recovery behaviour.

*Hypothesis:* Continuous control
will outperform discrete control
due to finer-grained authority over
steering and throttle, whereas
discrete actions may be insufficient
for nuanced obstacle scenarios.

== Experiment 2 --- Reward Function

Both reward types are crossed with
both action spaces (2 × 2 matrix):

#align(center)[
  #table(
    columns: (auto, auto, auto),
    inset: 0.5em,
    stroke: 0.3pt + luma(180),
    [], [*Dense*], [*Sparse*],
    [*Discrete*], [✓], [✓],
    [*Continuous*], [✓], [✓],
  )
]

*Hypothesis:* Dense rewards converge
faster across both action spaces,
providing per-step credit assignment
rather than episode-level inference.

== Experiment 3 --- Camera Robustness

*Part A --- FOV:* 15 m, 40 m, and
80 m look-ahead distances. Short
FOV causes reactive braking; long
FOV introduces distant-object noise.

*Part B --- Distortions:* Fog, rain,
and low-light filters applied to
the RGB input during training and
evaluation.

*Hypothesis:* Training with varied
FOV and distortion filters yields a
more robust policy by forcing
reliance on structural, distortion-
invariant scene features rather than
pixel-perfect colour gradients.

// ─── VI. Results ─────────────────────────────────────────────────────────────
= Results

#v(0.3em)
#align(center)[
  #rect(
    width: 100%,
    height: 5cm,
    stroke: 0.6pt + luma(180),
    fill: luma(248),
    radius: 3pt,
  )[
    #align(center + horizon)[
      #text(fill: luma(160), size: 8pt)[
        _[Figure placeholder:\
        Training reward curves for\
        all four Exp. 2 conditions]_
      ]
    ]
  ]
]

#v(0.5em)

#align(center)[
  #rect(
    width: 100%,
    height: 3.5cm,
    stroke: 0.6pt + luma(180),
    fill: luma(248),
    radius: 3pt,
  )[
    #align(center + horizon)[
      #text(fill: luma(160), size: 8pt)[
        _[Table placeholder:\
        Final metrics across all\
        experiments (mean ± std)]_
      ]
    ]
  ]
]

#v(0.5em)
Results will be populated upon
completion of the full experimental
campaign (final submission:
June 19, 2026).

// ─── VII. Conclusion ─────────────────────────────────────────────────────────
= Conclusion

#v(0.3em)
#rect(
  width: 100%,
  height: 4cm,
  stroke: 0.6pt + luma(180),
  fill: luma(248),
  radius: 3pt,
)[
  #align(center + horizon)[
    #text(fill: luma(160), size: 8pt)[
      _[Conclusion to be written after\
      experiments are finalised.\
      Will summarise key findings,\
      confirm or refute hypotheses,\
      and discuss limitations and\
      future work.]_
    ]
  ]
]

// ─── References ──────────────────────────────────────────────────────────────
= References

#set text(size: 8pt)
#bibliography(
  "bibliography.bib",
  title: none,
  style: "ieee",
)
