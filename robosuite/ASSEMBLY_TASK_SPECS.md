# Assembly Task Specifications — extended demos

This document expands the three shortlisted tasks (Multi-nut + Moving Pegs, Conveyor-fed Multi-part Assembly, Stack-and-Reorient with Regrasping) with concrete details for implementation in robosuite, recommended observations/sites, rewards, success metrics, and integration notes so you can implement them as extensions of the existing `run_nutassembly.py` or your stacking scenes.

---

## 1) Multi-nut + Moving Pegs (recommended quick extension)

- Summary: Reuse the existing NutAssembly assets but make pegs kinematic bodies that translate/rotate (e.g., on a short circular conveyor or rotating table). The robot must place each nut into its corresponding peg while the pegs move, requiring prediction and replanning.

- Objects / assets:
  - Existing nut objects (SquareNut, RoundNut) — reuse MuJoCo bodies and sites.
  - Pegs: make them separate body elements that are kinematically animated (via `set_body_xpos` or a simple periodic function in step loop).
  - Table / base: reuse.

- Scene layout:
  - Two or more pegs located on a rigid ring or moving platform. Pegs move with known kinematic trajectory (circular or linear back-and-forth). Start positions randomized per episode.

- Task flow:
  1. Observe nuts on table.
  2. Plan pick and timed-place to meet moving peg at insertion window.
  3. If peg position drifts or nut slips, replan insertion XY and timing.

- Observations (suggested):
  - `robot0_eef_pos`, `robot0_eef_quat` (existing)
  - Nut observables: `<nut>_pos`, (existing `<nut>_handle_site` / `<nut>_center_site` if available)
  - Peg kinematics: expose `peg{i}_body_xpos` and `peg{i}_body_xvel` via sim access or add small site markers like `peg1_center_site` (so planner sees future motion)
  - Time-step / elapsed time to help timing-based planning
  - Optional: approximate contact/force proxy from gripper sensors (if available)

- Sites to add/ensure:
  - `peg{n}_center_site`, `peg{n}_top_site` (for alignment)
  - existing nut handle/center sites already used in `run_nutassembly.py`

- Action space:
  - Same as your nut script: Cartesian position deltas and yaw control plus gripper open/close.

- Reward & success metrics:
  - Success: nut center within small XY radius of peg center and vertical clearance below threshold (simple boolean after placement)
  - Reward shaping: per-subtask rewards (pick success, lift, being within XY tolerance at insertion time, insertion complete). Penalize missed timing windows and large replans.
  - Metrics to collect: success rate, time-to-place, number of replans, wasted attempts, grasp slips.

- Implementation notes:
  - Implement pegs as kinematic bodies by advancing their transform in the environment loop (either by setting `sim.model.body_pos`/`sim.data.body_xpos` or by using a separate kinematic site that you move each step). Ensure obs reflect current peg pose.
  - Use your dynamic model to predict peg pose T steps into the future and plan timing accordingly; on deviation, replan.
  - Minimal code changes: reuse `HeuristicNutAssemblyPolicy`, add timing logic, and change `get_current_state` to read peg site/body positions each step.

- Effort: low→moderate. Fast to prototype and strong demo value for predictive replanning.

---

## 2) Conveyor-fed Multi-part Assembly (recommended showcase for replanning)

- Summary: Parts (subcomponents) arrive on a conveyor belt; robot must pick, assemble them to a staging area, and execute a multi-step assembly (e.g., place base → place spacer → place gear → fasten). New parts can arrive in unpredictable order or with variable timing, requiring reprioritization and replanning.

- Objects / assets:
  - Conveyor belt (kinematic; move parts along a line path)
  - Part catalog: `base_plate`, `spacer`, `gear`, `screw` (rigid bodies). Each should include `*_pos`, `*_handle_site`, optionally `*_center_site`.
  - Assembly fixture: a fixed jig or peg board where parts must be assembled in sequence. Add sites for each assembly slot.

- Scene layout:
  - Conveyor on one side, staging/assembly fixture on table center, robot workspace near both.

- Task flow:
  1. Parts spawn at conveyor start and move toward robot work zone.
  2. Robot picks parts as they arrive, places them into the fixture in correct order.
  3. Some parts require rotational alignment (e.g., gear), or regrasp to orient; screwing can be approximated as rotate-and-place.
  4. Unused parts may need to be moved to discard zone if the order differs.

- Observations:
  - For each part: `<part>_pos`, `<part>_quat` (or handle/center sites)
  - Conveyor velocity/pose or part linear velocity from sim data
  - Fixture slot site positions: `slot1_site`, `slot2_site`, etc.
  - Part type identifiers (one-hot) so planner knows which part it is.

- Sites to add:
  - `part_handle_site` for each part, `part_center_site`
  - `conveyor_marker_start`, `conveyor_marker_end` or per-part site on belt
  - `fixture_slot{i}_site`

- Action space:
  - As before: Cartesian + yaw + gripper, optionally allow small wrist rotation for alignment.

- Reward & success metrics:
  - Per-subtask reward (pick, place in correct slot, orientation within tolerance)
  - Final assembly success when all slots filled and parts stably resting in fixtures.
  - Measure throughput (parts assembled per minute), average replans per part, timeouts, and failure modes (dropped part, wrong slot).

- Implementation notes:
  - Conveyor can be simulated by kinematically updating each part's world position along a path each step; spawn parts using `sim.model` or by creating a pool of part bodies and enabling/disabling them.
  - Use your dynamic model to predict where a part will be after N steps to compute an intercept grasp plan.
  - Assembly ordering and scheduling offers opportunities to demonstrate replanning: when a high-priority part arrives earlier, pause/replan current task.

- Variations:
  - Variable conveyor speed, occlusions, noisy part poses.
  - Multiple conveyors with different part types.

- Effort: moderate (requires creating a few new bodies and a conveyor controller) but highest demo value for dynamic replanning.

---

## 3) Stack-and-Reorient with Regrasping (recommended if you want to extend stacking)

- Summary: Build a complex tower/structure from irregular blocks that require regrasping and mid-assembly reorientations to fit tighter slots — extends typical stacking by adding reorientation subgoals and regrasp planning.

- Objects / assets:
  - A set of block types with asymmetric geometry (L-shaped, T-shaped, slotted blocks), each with `*_pos`, `*_handle_site`, `*_center_site`.
  - Assembly base with slots that accept blocks only when oriented with certain yaw/pitch.

- Scene layout:
  - Blocks placed randomly on table; base fixture at center with slots and walls constraining placement.

- Task flow:
  1. Pick block from table.
  2. If current grasp orientation incompatible, perform regrasp: place block on table or on intermediate support, release, regrasp with different approach.
  3. Insert block into slot with precise alignment.
  4. Repeat until structure complete.

- Observations:
  - Block poses and handle/center site positions.
  - Slot site positions and orientations.
  - Optional contact proxies as tilt/force signals.

- Sites to add:
  - `block{n}_handle_site`, `block{n}_center_site`, `slot{i}_site`

- Action space:
  - Cartesian + full orientation control (not only yaw) + gripper open/close.

- Reward & success metrics:
  - Reward for each correctly placed block. Penalty for dropping or mis-inserting.
  - Success = all blocks placed and structure stable for K steps.
  - Metrics: number of regrasp actions, total time, success rate, precision at insertion.

- Implementation notes:
  - Regrasp behavior can be scripted heuristically (place-on-table → reapproach) or left to planner/learning.
  - To keep simulation stable, use simple convex meshes or approximate shapes with compound primitives.

- Effort: moderate. Good extension of stacking; regrasp logic reuses many control primitives.

---

## Common Implementation Recommendations

- Sites: Standardize naming so your policies and dynamic model can query e.g., `<obj>_handle_site`, `<obj>_center_site`, `<obj>_horizontal_radius_site`.
- Kinematic motion: Animate conveyors/pegs via a small kinematic controller in the env step loop; ensure observations reflect updated poses.
- Observation timing: If you plan to demonstrate predictive planning, expose recent velocity estimates (finite differences) or body_xvel from sim.
- Logging: instrument episodes with event timestamps (pick start/end, place start/end, replans count). Save plan snapshots to analyze replanning triggers.
- Heuristic policy reuse: Your `HeuristicNutAssemblyPolicy` is a good starting scaffold — factor out common helpers (get_nut_center, compute_insertion_xy, compute_yaw_action) into a shared helper module to reuse for new tasks.

---

## Estimated effort & prioritization

- Quickest (lowest effort): **Multi-nut + Moving Pegs** — small code changes and maximum reuse of existing assets.
- Best demo for predictive replanning: **Conveyor-fed Multi-part Assembly** — moderate effort, highest payoff.
- Best extension for stacking: **Stack-and-Reorient with Regrasping** — moderate effort, reuses stacking logic and demonstrates regrasp strategies.

---

## Next steps (implementation plan)

1. Pick the target task (which of the three to implement first).
2. Create or adapt MuJoCo XML assets for the new objects and sites.
3. Add an env wrapper or small step-time kinematic controller for moving pegs/conveyor.
4. Extend the heuristic policy or planner to use future-state predictions and replan triggers.
5. Run and log episodes, iterate on observation/site tuning.


---

File created to help you implement these demos. If you pick one, I can produce the exact MuJoCo XML snippet, site names, and a `run_*.py` scaffold to integrate with your existing code.
