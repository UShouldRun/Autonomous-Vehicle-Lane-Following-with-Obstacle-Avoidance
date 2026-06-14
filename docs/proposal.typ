#set document(
  title: "Autonomous Vehicle Lane Following with Obstacle Avoidance",
  author: ("João Pedro Gouveia Ferreira", "Henrique Teixeira", "Miguel Almeida"),
)

#set page(
  paper: "a4",
  margin: (x: 2.5cm, y: 2.5cm),
  numbering: "1",
  header: context {
    if counter(page).get().first() > 1 [
      #set text(size: 8pt, fill: luma(130))
      #grid(
        columns: (1fr, 1fr),
        align(left)[Autonomous Vehicle Lane Following],
        align(right)[Project Proposal],
      )
      #line(length: 100%, stroke: 0.4pt + luma(180))
    ]
  }
)

#set text(font: "New Computer Modern", size: 11pt, lang: "en")
#set heading(numbering: "1.")
#set par(justify: true, leading: 0.7em)
#show heading.where(level: 1): it => {
  v(1.2em)
  block(it)
  v(0.4em)
}
#show heading.where(level: 2): it => {
  v(0.8em)
  block(it)
  v(0.3em)
}

// ─── Title Block ────────────────────────────────────────────────────────────
#align(center)[
  #v(1.5cm)
  #text(size: 20pt, weight: "bold")[
    Autonomous Vehicle Lane Following \
    with \
    Obstacle Avoidance
  ]
  #v(0.4cm)
  #text(size: 13pt, fill: luma(60))[Project Proposal - Group A]
  #v(0.8cm)
  #grid(
    columns: (1fr, 1fr, 1fr),
    gutter: 0.4cm,
    align(center)[
      #text(weight: "semibold")[João Ferreira] \
      #text(size: 9pt, fill: luma(80))[up202306717]
    ],
    align(center)[
      #text(weight: "semibold")[Henrique Teixeira] \
      #text(size: 9pt, fill: luma(80))[up202306640]
    ],
    align(center)[
      #text(weight: "semibold")[Miguel Almeida] \
      #text(size: 9pt, fill: luma(80))[up202303926]
    ],
  )
  #v(1cm)
  #line(length: 80%, stroke: 1pt + luma(180))
  #v(1.5cm)
]

#pagebreak()

// ─── 1. Goals ────────────────────────────────────────────────────────────────
= Goals

The objective of this project is to train an autonomous vehicle using Reinforcement Learning (RL) to follow the yellow centre line on a road and avoid obstacles. Obstacles may be static (e.g. barrels) or dynamic/moving (e.g. pedestrians or other vehicles).

The agent will process data from a *Camera* and *LiDAR* sensor, along with the current speed and the vehicle's angle relative to the yellow line, and output either a discrete or continuous action controlling acceleration/braking and steering. The RL model will likely be built around a Convolutional Neural Network (CNN).

Three experiments will be conducted to evaluate key design choices:

#block(inset: (left: 1em))[
  1. *Action Space*: discrete versus continuous control.
  2. *Reward Function*: sparse versus dense reward signals.
  3. *Camera Robustness*: varying field of view and applying environmental distortion filters.
]

#pagebreak()

// ─── 2. Proposed Approach ────────────────────────────────────────────────────
= Proposed Approach

== Setup

#grid(
  columns: (auto, 1fr),
  gutter: 0.5em,
  [*Simulator:*], [Webots (2023b or newer)],
  [*Robot:*], [E-puck],
)

#v(0.4em)
The E-puck is equipped with:
- *Camera*: RGB at 64 × 64 resolution for line detection.
- *LiDAR*: 360° point cloud or 1-D radial scan for proximity sensing.
- *Actuators*: Differential-drive motors controlled via velocity commands.

The environment is a custom Webots world featuring a *looped track* with a yellow centre line. Procedural Generation spawns primitives (Barrels, Pedestrians, Vehicles) at random intervals so the model generalises rather than memorising a fixed path. Each primitive is described by its centroid and type-specific parameters:

#grid(
  columns: (auto, 1fr),
  gutter: 0.4em,
  inset: (left: 1em),
  [*Barrel:*], [radius],
  [*Pedestrian:*], [radius],
  [*Vehicle:*], [bounding rectangle],
  [*Road edge:*], [discrete sequence of points (no centroid)],
)

#v(0.6em)
*Minimum deliverables:*

- A functional Webots environment controllable via a Python API (OpenAI Gym / Gymnasium wrapper).
- The Observation Space pipeline, extracting the vehicle's angle relative to the yellow line.
- A Baseline Agent capable of completing a simple circular track with no obstacles using a basic PPO or DQN approach.

== Algorithms and Architecture

For the *discrete action space*, Deep Q-Learning (DQN) is used for its proven effectiveness with finite action sets, experience replay, and target networks.

For the *continuous action space*, Proximal Policy Optimization (PPO) serves as the industry standard for continuous robotic control, owing to its stability and trust-region updates.

Both approaches share a common backbone: a CNN that processes RGB frames, whose output is concatenated with a flattened vector containing LiDAR readings, current velocity, and the alignment angle before passing through fully-connected layers.

#pagebreak()

// ─── 3. Reward Functions ─────────────────────────────────────────────────────
= Reward Functions

== Dense Reward

Provides feedback at every timestep, accelerating convergence.

#align(center)[
  $"Reward" = "Progress" - "Alignment Penalty" - "Collision Penalty"$
]

#grid(
  columns: (auto, 1fr),
  gutter: 1.0em,
  inset: (left: 1em),
  [*Progress* $(w_1 dot V cos theta)$:],
  [Rewards high velocity $V$ when heading $theta$ is aligned with the yellow line. Prevents reward-farming by spinning.],
  [*Alignment Penalty* $(w_2 dot |d|)$:],
  [Penalty proportional to lateral distance $d$ from the yellow line.],
  [*Collision Penalty* $(w_3 dot C)$:],
  [Large negative constant (e.g. $-100$) triggered when LiDAR detects a distance below the collision threshold.],
)

== Time-to-Collision (TTC) Reward

Augments the dense reward with a proactive safety term.

#align(center)[
  $"Reward" = "Progress" - "Deviation Penalty" - "Safety Term" - "Terminal Penalty"$
]

#grid(
  columns: (auto, 1fr),
  gutter: 1.0em,
  inset: (left: 1em),
  [*Progress* $(w_1 dot V cos theta)$:],
  [Same as dense reward.],
  [*Deviation Penalty* $(w_2 dot d^2)$:],
  [Quadratic penalty on lateral distance, penalising large deviations more heavily.],
  [*Safety Term* $(w_3 dot max(0, 1 - D_"min" / D_"safe")^2)$:],
  [Zero beyond $D_"safe"$; grows quadratically as the agent approaches an obstacle, encouraging early evasive action.],
  [*Terminal Penalty* $(w_4 dot C_"terminal")$:],
  [Large discrete penalty on actual collision; ends the episode.],
)

== Sparse Reward

Used for comparison only.

#block(inset: (left: 1em))[
  - $"Reward" = +1$ if a checkpoint or lap is completed.
  - $"Reward" = -1$ if a collision occurs.
  - $"Reward" = 0$ otherwise.
]

Since a raw sparse signal is unlikely to converge, curriculum learning is applied:

#block(inset: (left: 1em))[
  - *Stage 1* — No obstacles; sparse $+1 \/ -1$ only. Agent learns basic lap completion.
  - *Stage 2* — Add static obstacles (barrels).
  - *Stage 3* — Add dynamic obstacles (pedestrians, vehicles).
]

#pagebreak()

// ─── 4. Empirical Evaluation ─────────────────────────────────────────────────
= Empirical Evaluation

Each experiment is run *100 times* and results are averaged.

== Evaluation Metrics

#grid(
  columns: (auto, 1fr),
  gutter: 1.0em,
  [*Collisions*:], [Total number of collisions during the track.],
  [*Success Rate (%)*:], [Percentage of trials completed without collision.],
  [*Cross-Track Error (m)*:], [Average lateral distance from the yellow line.],
  [*Mean Lap Time (s)*:], [Time to complete one full circuit.],
  [*Safety Score*:], [Ratio of distance travelled to near-misses recorded by LiDAR.],
)

== Experiment 1 — Action Space Comparison

*Goal:* Compare discrete vs. continuous action spaces.

#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  block(
    fill: luma(245),
    inset: 0.8em,
    radius: 4pt,
  )[
    *Discrete (DQN)* \
    Actions: Left, Right, Straight, Brake
  ],
  block(
    fill: luma(245),
    inset: 0.8em,
    radius: 4pt,
  )[
    *Continuous (PPO)* \
    Steering $theta in [-1, 1]$,
    Throttle $a in [-1, 1]$
  ],
)

*Measured:* Sample efficiency (reward vs. training steps), trajectory smoothness, recovery behaviour.

*Hypothesis:* Continuous control will outperform discrete control, as it grants finer-grained authority over steering and throttle rather than being limited to a fixed set of moves.

*Reasoning:* We believe continuous control will perform better than discrete control because it gives the model finer authority over steering and throttle for different obstacle scenarios, whilst discrete control restricts the model to a predefined set of moves which may hinder its performance in situations that demand more nuanced responses.

== Experiment 2 — Reward Function Impact

Both reward functions are crossed with both action spaces, forming a *2 × 2 matrix*:

#align(center)[
  #table(
    columns: (auto, auto, auto),
    inset: 0.6em,
    stroke: 0.4pt + luma(180),
    [], [*Dense*], [*Sparse*],
    [*Discrete*], [✓], [✓],
    [*Continuous*], [✓], [✓],
  )
]

*Measured:* Sample efficiency, laps completed per episode, collision rate, and speed of convergence to a viable driving policy.

*Hypothesis:* Dense reward functions will converge faster and yield higher performance across both action spaces, as they provide a learning signal at every timestep instead of forcing the agent to credit-assign over entire episodes.

*Reasoning:* Dense rewards provide a learning signal at every timestep, allowing the agent to associate specific actions with immediate feedback. On the other hand, a sparse reward forces the agent to infer which of all of the actions taken during a lap were responsible for the outcome, which is harder and slower to learn.

== Experiment 3 — Camera Input Distortion

*Part A — Field of View:* Test FOV distances of 15 m, 40 m, and 80 m ahead. Very short FOV causes reactive late braking; very long FOV introduces noise from irrelevant distant objects.

*Part B — Distortion Filters:* Fog, rain, and low-light filters applied to the RGB feed to simulate real-world conditions.

*Measured:* Robustness score (percentage increase in collision rate and lap time relative to clean-input baseline), lap time and collision rate per FOV distance and distortion type.

*Hypothesis:* Training with varied FOV and distortion filters produces a more robust policy. A model trained only on clean visuals overfits to pixel-perfect features absent in real deployment; exposing it to corrupted inputs forces reliance on distortion-invariant structural features.

*Reasoning:* A model trained only on clean visuals learns to exploit pixel-perfect features that do not exist in real deployment. By exposing the agent during training to corrupted inputs — such as blur, brightness shifts, and noise — it is forced to rely on more structural, distortion-invariant features of the scene (such as the general shape and position of the yellow line rather than its exact colour gradient). Varying the FOV prevents the agent from overfitting: too narrow and it learns reactive behaviours; too wide and it may overfit to distant context.

#pagebreak()

// ─── 5. Weekly Objectives ────────────────────────────────────────────────────
= Weekly Objectives

#let week(label, items) = {
  block(
    width: 100%,
    inset: (x: 0.8em, y: 0.6em),
    radius: 4pt,
    stroke: 0.4pt + luma(200),
    fill: luma(250),
  )[
    #text(weight: "bold")[#label]
    #v(0.3em)
    #for item in items [
      - #item
    ]
  ]
  v(0.5em)
}

#week("Week 1 — March 30 – April 5", (
  "Set up Webots environment and dependencies.",
  "Load and explore the Webots City demo.",
  "Implement basic vehicle control.",
  "Start Gymnasium wrapper for the environment.",
))

#week("Week 2 — April 6 – April 12", (
  "Implement observation space (camera + LiDAR + vehicle state).",
  "Compute vehicle alignment relative to the yellow line.",
  "Define initial reward functions (dense or sparse).",
  "Implement the action space (discrete or continuous).",
))

#week("Week 3 — April 13 – April 19", (
  "Implement PPO or DQN baseline agent.",
  "Train on simplified City environment (no obstacles).",
  "Validate lap completion.",
  "Start logging metrics.",
))

#week("Week 4 — April 20 – April 26", (
  "Refine baseline RL agent training and reward function.",
  "Improve observation preprocessing.",
  "Introduce static obstacles in the City environment.",
))

#week("Week 5 — April 27 – May 3", (
  "Train the selected RL algorithm in more complex scenarios.",
  "Begin Experiment 2 (dense vs. sparse rewards).",
  "Collect preliminary results.",
))

#week("Week 6 — May 4 – May 10", (
  "Implement the alternative action space.",
  "Implement corresponding baseline agent (PPO or DQN).",
  "Start Experiment 1 (continuous vs. discrete).",
  "Continue collecting metrics.",
))

#pagebreak()

#week("Week 7 — May 11 – May 17 ★ Intermediate Checkpoint (May 14)", (
  "Demonstrate working agent in City environment.",
  "Show preliminary PPO / DQN results.",
  "Show obstacle interaction.",
  "Present partial experiment results.",
))

#week("Week 8 — May 18 – May 24", (
  "Expand Experiment 1.",
  "Continue Experiment 2.",
  "Implement Experiment 3 (camera distortion / FOV).",
))

#week("Week 9 — May 25 – June 2", (
  "Continue experiments with multiple runs.",
  "Compute intermediate metrics.",
  "Generate initial plots.",
  "Start writing paper.",
))

#week("Week 10 — June 3 – June 7", (
  "Finalise experiments.",
  "Compute final metrics.",
  "Complete plots and tables.",
))

#week("Week 11 — June 8 – June 14", (
  "Finalise paper.",
  "Clean codebase.",
  "Write instructions.txt.",
  "Test reproducibility.",
))

#week("Week 12 — June 15 – June 19 ★ Final Submission", (
  "Prepare presentation and demo.",
  "Final adjustments.",
  "Submit project.",
))

#pagebreak()

// ─── 6. Intermediate Delivery ────────────────────────────────────────────────
== Intermediate Delivery (May 14)

By the intermediate checkpoint the group must have completed:

- A working Webots simulation based on the City demo.
- A Gymnasium-compatible environment.
- Full observation space: camera, LiDAR, and vehicle state.
- At least one reward function implemented.
- At least one trained RL baseline agent (PPO or DQN), capable of completing the track in a simplified scenario; initial comparison results if both are implemented.
- Initial obstacle integration.
- Preliminary results: reward evolution during training, success rate in simplified scenarios.
