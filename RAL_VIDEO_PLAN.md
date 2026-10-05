# RA-L video submission plan (2026-10-01)

Supersedes the real-robot section of `video_submission_guide.md`. That guide is CoRL-era and sim-only, and its
real-robot section proposes scripted overlays with invented critic scores. Drop that: there are now real logged plans,
imaginations and scores for 170 episodes, and RA-L reviews the video together with the paper.

## 1. What RA-L requires

- **One video only**, in mpeg/mpg/mp4 with standard codecs (H.264 + AAC is the safe choice).
- **One zip of at most 50 MB**, containing the video plus `ReadMe.txt` (what is needed to play it) and `Summary.txt`
  (what it shows).
- **Aim for 1-3 minutes**, with title, authors and affiliations at the start and credits at the end.
- **English voice-over or captions**; subtitles are recommended.
- **No supplemental text or figures** passed off as multimedia: it must not read as extra pages.

Sources:
- https://www.ieee-ras.org/publications/ra-l/ra-l-information-for-authors/
- https://www.ieee-ras.org/publications/video-submission-guidelines/

## 2. What exists and what is missing

- **Logs:** every episode has the camera frame, the real2sim render, each candidate's imagined frames, critic scores and
  the chosen plan under `verify2act/output/real/twin/<variant>/<session>_ep_XXX/imagination_logs/planning_call_NN/`
  (`request_image.png`, `real2sim_render.png`, `candidate_NN/horizon_NN/{imagine_frame.png, action.txt,
  temporal_critic.json, goal_critic.json}`, `candidate_plans.json`, `response.json`). Before/after stills are in
  `verify2act/output/real/results/<variant>/<session>/ep_XXX/`.
- **Footage:** no video file from the real eval was found, only stills. This was checked only in
  `verify2act/output/real`, `verify2act/output/real_old` and `~/Videos`; a repo-wide search timed out, so footage stored
  elsewhere would not have shown up. If the arm was filmed with an external camera, that is the footage to use; if not,
  a handful of episodes need re-shooting.
- **Tooling:** `ffmpeg` is not installed on this machine.

## 3. Suggested cut (about 2.5 minutes)

| # | time | segment | content |
|---|---|---|---|
| 1 | 0-15 s | Title and problem | Title card, then one line on why VLM plans need verifying before execution. |
| 2 | 15-40 s | Method | Architecture diagram, revealed stage by stage: propose, imagine in latent space, verify, reflect. |
| 3 | 40-95 s | Real-robot hero episode | External footage next to a panel built from the logs: camera frame, real2sim render, imagined frames per candidate, the real goal-head score, reject, replan, accept, then execution at 4-8x with a speed label. |
| 4 | 95-125 s | Same layout, different variant | V2A next to VLM-only on one layout where VLM-only failed. |
| 5 | 125-145 s | Results | Real success rates, plus one CALVIN or nut-assembly sim clip if the paper keeps those. |
| 6 | 145-160 s | Failure case and credits | One honest grasp failure, then credits. |

## 4. Episodes worth using

V2A runs that rejected a plan, replanned and succeeded (`output/real/results/v2a_wm/*/episodes.jsonl`):

- **task1b ep 1, 6, 8:** three goal rejections, one replan, success.
- **task2a ep 4, 5, 9 and task2b ep 4:** rejection, replan, success.

For the side-by-side, task1b ep 1 and task2a ep 1 are the candidates: VLM-only failed both ("failed to grasp green
block" / "failed to grasp the red block") and V2A succeeded. This assumes episode k is layout Lk in both sessions;
confirm against the session notes.

### 4a. Shot list for the re-shoot (2026-10-02, replaces the picks above)

Checked against the logs: every VLM-only failure on task1b and task2a is a grasp failure with the same plan V2A
executed, so a "VLM-only fails, V2A succeeds" side-by-side on those episodes would credit the verifier with grasp luck.
The task2a rejections are also weak: in ep 7 the same plan scored 0.05, 0.26 and 1.00 over three WM samples. The clean
story is task1b: the VLM proposes binning the red block (warm, must stay), the goal head scores all three candidates
0.08-0.29, and the replan drops red (V2A ep 1, 6, 8; VLM-only executed the red plan in ep 2 and 5).

| shot | variant | task, layouts | keep filming until | use |
|---|---|---|---|---|
| A | `v2a_wm` | task1b, L1 / L6 / L8 | 2 takes with reject, replan, success | hero (segment 3) |
| B | `vlm_only` | task1b, same layout as the best A take | 1 take that puts red in the bin | contrast (segment 4) |
| C | `v2a_wm` | task2a L3, task3a L1 | 1 success each | breadth, at 8x |
| D | any | any | nothing staged: keep one grasp failure from A-C | failure case (segment 6) |
| E | – | 10 s of the whole setup, arm at home | – | title background |

The proposal step is stochastic (the red plan showed up in 3/10 V2A and 2/5 VLM-only task1b episodes), so expect about
6-8 takes for A and 3-5 for B. `replans` and `goal_rejections` in the session's `episodes.jsonl` say right after each
take whether it is a keeper.

Filming: tripod, landscape, 1080p 30 fps, arm + sheet + bin in frame, camera not moved between takes; one clip per
episode from before the session's Enter to the arm back at home. Name clips `<task>-L<k>-<variant>-take<n>.mp4`.
Sync the video sessions' server logs and Jetson results into `verify2act/output/real_video/`, **not** `output/real/`:
`make_real_figs.py` reads everything under `output/real/results` and would count them in the paper's numbers.

## 5. Results card

From `verify2act/output/paper/real_numbers.json`:

| Variant | Success | Macro rate |
|---|---|---|
| V2A | 42/55 | 0.77 |
| RLA-WM | 34/55 | 0.62 |
| VLM-only | 16/30 | 0.53 |
| Diffusion | 14/30 | 0.47 |

The episode counts are unequal: V2A and RLA-WM ran 10 layouts per task (5 for task2b), the other two ran 5, and task1c
was dropped. Either show n on the card or restrict all four variants to the shared layouts L1-L5.

Most V2A failures are grasp failures rather than planning failures (13 execution failures, 0 plan failures), which is
worth one caption.

## 6. Next steps

1. Settle the footage question: is there external-camera video of the eval runs, and where?
   - If yes: write a script that renders the log panel for the chosen episodes, composite it beside the footage, and
     encode under 50 MB.
   - If no: re-shoot the four or five chosen layouts from the layout sheets (`verify2act/real_eval_layouts/`) with the
     live server, so the logs match the footage.
2. Install `ffmpeg` (or encode on another machine).
3. Write `ReadMe.txt` and `Summary.txt`, then zip them with the video.

## 7. Footage and offline replay (2026-10-02)

Footage is in `verify2act/output/real_video/` (clips sit in `twin/<variant>/<session>_ep_XXX/`, Jetson results in
`robot_results_video/`). It was filmed **without the sheet and without `--real2sim`**, so the live V2A logs are not
usable: imaginations are smears and correct plans scored near 0 and ran unverified. `Real2Sim` also fails silently
without a sheet (no blocks found, rms 0.0, empty render passes the 2.5 px check).

Salvage: `real_video/replay_nosheet/` holds `nosheet.py` (camera fixed at the calibrated home pose, whole-frame colour
masks tuned for these dim frames), `replay.py` (re-verifies the plans the VLM proposed live, no VLM call) and the
resulting logs in the usual `imagination_logs` layout. `replay.py` reproduces the 2026-10-01 eval scores on the sheet
episodes (task1b ep 1, 6). Replay matches the executed plan in:

| use | take | replay result |
|---|---|---|
| hero | task1b take 2 (`IMG_8670.MOV`) | three red plans rejected (0.14, 0.00, 0.21), replan green+blue accepted 0.82, real success |
| breadth | task2a take 1 (`IMG_8686.MOV`), task3a take 3 (`IMG_8691.MOV`) | accepted 0.98 / 1.00, real success |
| failure | task3a take 1 or 2, task2a take 2 | plan accepted, execution failed |
| contrast | VLM-only task1b take 2 (`IMG_8675.MOV`) | live log valid: bins red |

Not usable: task1b take 1 (replay accepts a red plan at 0.64), takes 3-5 and task2a take 3 (replay's plan differs from
the one executed). Any panel built from the replay must be captioned as recomputed offline from the recorded frame.
`ffmpeg` lives in the conda env `ffmpeg`.

## 8. First cut (2026-10-02)

`verify2act/output/real_video/video_build/build.py` renders the cards and assembles `verify2act_video.mp4` (2:02,
16.8 MB, anonymous, metadata stripped); `verify2act_video_submission.zip` adds `ReadMe.txt` and `Summary.txt`. Trim
points, speeds and captions are in `build.py::main`. The user chose no on-screen note about the offline replay; it is
stated in `Summary.txt`. The failure case is task3a take 2 (`IMG_8690.MOV`), not task2a take 2 (its outcome is unclear
on camera).
