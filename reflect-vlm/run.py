import os
import time
import pandas as pd
from PIL import Image
import wandb
from absl import app, flags
import random
import uuid
import copy
import numpy as np
import imageio
import mujoco

from roboworld.utils.config import define_flags, get_user_flags
from roboworld.utils.logger import WandBLogger, set_random_seed
from roboworld.envs.generator import generate_xml
from roboworld.envs.asset_path_utils import full_path_for
from roboworld.envs.mujoco.franka.franka_assembly import (
    FrankaAssemblyEnv, AssemblyOracle
)
from roboworld.agent.oracle import OracleAgent
from roboworld.agent.utils import parse_act_txt, get_prompt

FLAGS = flags.FLAGS

FLAGS_DEF = define_flags(
    agent_type=("expert", "string", "Type of agent (llava, expert, random)"),
    camera_name=("table_back", "string", "Camera name."),
    model_path=("/path/to/model", "string", "Path to model"),
    model_base=(None, "string", "Path to base model"),
    revise_action=(False, "bool", "Revise an initially proposed action"),
    dummy_revised_action=(False, "bool", "If True, use original action as revised action. "
                          "This is for data collection when the agent is not trained to reflect yet "
                          "(usually the first iteration of DAgger)."),
    imagine_future_steps=(0, "integer", "Generate/simulate future image."),
    diffuser_pretrained_model=(None, "string", "Path to pretrained diffuser"),
    diffuser_unet_dir=(None, "string", "Path to unet model of diffuser"),
    diffuser_vae_dir=(None, "string", "Path to vae model of diffuser"),
    level=("all", "string", "Level of difficulty (medium, hard, or all)"),
    seed=(42, "integer", "Random seed."),
    reset_seed_start=(0, "integer", "first seed to reset env"),
    max_steps=(50, "integer", "Max number of decision steps in a trajectory"),
    n_trajs=(100, "integer", "Number of trajectories."),
    repeat_per_env=(1, "integer", "Number of trajectories for each env/board."),
    save_dir=("datasets/data_v9", "string", "Directory for saving data."),
    start_traj_id=(0, "integer", "Starting trajectory index"),
    start_board_id=(0, "integer", "Starting board index"),
    oracle_prob=(0.5, "float", "Probability of executing oracle action at each timestep"),
    record=(True, "bool", "Record video."),
    record_frame_skip=(5, "integer", "Skip between recorded frames."),
    logging=WandBLogger.get_default_config(),
)


def imagine_with_sim(env, agent, first_action, goal_img, history, obj_labels, traj_dir, t):
    env_recording = env.is_recording
    if env_recording:
        env._record = False
    _env_state = env.__getstate__()
    _plan = [first_action]
    for i in range(FLAGS.imagine_future_steps):
        try:
            err = env.act_txt(_plan[-1])
            Image.fromarray(env.read_pixels(camera_name=FLAGS.camera_name)).save(
                os.path.join(traj_dir, f"sim-{t}-{'-'.join(_plan).replace(' ', '_')}.png")
            )
        except Exception as e:
            print(f"Error during imagining into future: {e}")
            break

        if i + 1 == FLAGS.imagine_future_steps or env.is_success():
            break

        _img = env.read_pixels(camera_name=FLAGS.camera_name)
        _a = agent.act(_img, goal_img, inp=get_prompt(
            version="propose", history=history + _plan, obj_labels=obj_labels
        ))
        if _a == "done":
            break
        _plan.append(_a)
    
    future_img = env.read_pixels(camera_name=FLAGS.camera_name)
    env.__setstate__(_env_state)
    if env_recording:
        env._record = True

    return _plan, future_img


def imagine_with_diffusion(diffusion_sim, agent, first_action, img, goal_img, history, obj_labels, traj_dir, t):
    _plan = [first_action]
    _img = img
    for i in range(FLAGS.imagine_future_steps):
        try:
            next_img = diffusion_sim.forward(curr_image=_img, act_text=_plan[-1])
            Image.fromarray(next_img).save(os.path.join(traj_dir, f"gen-{t}-{'-'.join(_plan).replace(' ', '_')}.png"))
        except Exception as e:
            print(f"Error during imagining into future with diffusion: {e}")
            break

        _img = next_img
        if i + 1 == FLAGS.imagine_future_steps:
            break
        
        _a = agent.act(_img, goal_img, inp=get_prompt(
            version="propose", history=history + _plan, obj_labels=obj_labels
        ))
        if _a == "done":
            break

        _plan.append(_a)

    future_img = _img

    return _plan, future_img


def build_env(env_seed, xml_filename, render_mode="offscreen"):
    xml, info = generate_xml(seed=env_seed)
    if (FLAGS.level == 'medium' and info['n_bodies'] > 5) or (FLAGS.level == 'hard' and info['n_bodies'] <= 5):
        return None, None
    xml.write_to_file(filename=xml_filename)

    board_name = "brick_1"
    fixture_name = None  # "fixture"
    peg_ids = [j + 1 for j in range(1, info['n_bodies'])]
    peg_names = [f"brick_{j+1}" for j in range(1, info['n_bodies'])]
    peg_descriptions = [info['brick_descriptions'][peg_name] for peg_name in peg_names]

    peg_labels = [" ".join(pd.split()[:1]) for pd in peg_descriptions]
    peg_labels_shuffled = peg_labels.copy()
    random.shuffle(peg_labels_shuffled)

    env = FrankaAssemblyEnv(board_name=board_name, fixture_name=fixture_name, peg_names=peg_names, peg_descriptions=peg_descriptions,
                            render_mode=render_mode, frame_skip=20, model_name=xml_filename, max_episode_length=50000,
                            magic_attaching=True)
    env_info = {
        "peg_ids": peg_ids,
        "peg_names": peg_names,
        "peg_descriptions": peg_descriptions,
        "peg_labels": peg_labels,
        "peg_labels_shuffled": peg_labels_shuffled,
        "dependencies": info["dependencies"]
    }
    
    return env, env_info


def main(_):
    variant = get_user_flags(FLAGS, FLAGS_DEF)
    wandb_logger = WandBLogger(config=FLAGS.logging, variant=variant)
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    set_random_seed(FLAGS.seed)
    env_seed = FLAGS.seed
    render_mode = 'offscreen'
    candidate_act_list = ["pick up", "put down", "insert", "reorient"]
    xml_filename = full_path_for(f"tmp_{uuid.uuid4()}.xml")

    # check flags
    assert FLAGS.agent_type in {"llava", "expert", "random"}, f"Unknown agent type `{FLAGS.agent_type}`"
    assert FLAGS.level in {"medium", "hard", "all"}, f"Unknown assembly level `{FLAGS.level}`"
    if FLAGS.revise_action:
        from roboworld.agent.diffuser import DiffusionSim
        assert FLAGS.agent_type == "llava"
        if FLAGS.diffuser_pretrained_model is not None:
            diffusion_sim = DiffusionSim(
                pretrained_model=FLAGS.diffuser_pretrained_model, 
                unet_dir=FLAGS.diffuser_unet_dir, 
                vae_dir=FLAGS.diffuser_vae_dir, 
            )
        else:
            diffusion_sim = None

    # initialize agent
    agent = None
    if FLAGS.agent_type == "llava":
        from roboworld.agent.llava import LlavaAgent
        agent = LlavaAgent(model_path=FLAGS.model_path,
                           model_base=FLAGS.model_base)

    board_cnt, traj_cnt, succ_cnt = 0, 0, 0
    data = []
    traj_id = FLAGS.start_traj_id
    board_id = FLAGS.start_board_id
    
    while traj_id < FLAGS.start_traj_id + FLAGS.n_trajs:

        env, env_info = build_env(env_seed, xml_filename, render_mode)
        if env is None:
            env_seed += 1
            continue
        oracle = AssemblyOracle(env=env, brick_ids=env_info["peg_ids"], 
                                brick_descriptions=env_info["peg_descriptions"], 
                                dependencies=env_info["dependencies"])
        oracle_agent = OracleAgent(oracle)

        idx_in_env = 0
        reset_seed = FLAGS.reset_seed_start
        while idx_in_env < FLAGS.repeat_per_env:
            print(">" * 10, f"Trajectory {traj_id}", ">" * 10)
            traj_dir = os.path.join(FLAGS.save_dir, str(traj_id))
            os.makedirs(traj_dir, exist_ok=True)
            reset_seed += 1
            env.reset(seed=reset_seed)
            goal_img = env.goal_images.get(FLAGS.camera_name, None)
            assert goal_img is not None

            # initialize data
            traj_succ = False
            exec_act_list = [None] * FLAGS.max_steps
            oracle_act_list = [None] * FLAGS.max_steps
            agent_act_list = [None] * FLAGS.max_steps
            agent_plan_list = [None] * FLAGS.max_steps
            agent_act_revised_list = [None] * FLAGS.max_steps
            traj_key_frames = []
            question_list, answer_list = [], []
            history = []
            history_list = []
            total_time, inference_time, rollout_time = 0., 0., 0.

            if FLAGS.record:
                env.record_on(record_frame_skip=FLAGS.record_frame_skip)
                env.render()

            # start rollout
            total_time_t0 = time.time()
            for t in range(FLAGS.max_steps):
                print(f"[Step {t}]")
                oracle_action = oracle_agent.act()
                oracle_act_list[t] = oracle_action
                oracle_action_primitive, oracle_brick_color = parse_act_txt(oracle_action)
                print("Oracle action:", oracle_action)
                
                img = env.read_pixels(camera_name=FLAGS.camera_name)
                traj_key_frames.append(img)

                inp = get_prompt(version="propose", history=history, obj_labels=env_info["peg_labels_shuffled"])
                print("Q:", inp)
                question_list.append(inp)
                history_list.append(copy.deepcopy(history))

                inference_t0, rollout_t0, rollout_t = None, None, None
                agent_action_revised = None

                try:
                    # get action with agent
                    if FLAGS.agent_type == "expert":
                        agent_action = oracle_action
                    elif FLAGS.agent_type == "random":
                        agent_action = f"{np.random.choice(candidate_act_list)} {np.random.choice(env_info['peg_labels'])}"
                    else:
                        inference_t0 = time.time()
                        agent_action = agent.act(img, goal_img, inp)

                        # reflect and revise action
                        if FLAGS.revise_action:                            
                            assert FLAGS.imagine_future_steps > 0

                            if diffusion_sim is None:
                                _plan, future_img = imagine_with_sim(
                                    env=env, agent=agent, first_action=agent_action, goal_img=goal_img, history=history, 
                                    obj_labels=env_info["peg_labels_shuffled"], traj_dir=traj_dir, t=t
                                )
                            else:
                                _plan, future_img = imagine_with_diffusion(
                                    diffusion_sim=diffusion_sim, agent=agent, first_action=agent_action, img=img, goal_img=goal_img, 
                                    history=history, obj_labels=env_info["peg_labels_shuffled"], traj_dir=traj_dir, t=t
                                )
                            agent_plan_list[t] = copy.deepcopy(_plan)

                            if FLAGS.dummy_revised_action:
                                agent_action_revised = agent_action
                            else:
                                inp2 = get_prompt(version="reflect", history=history, obj_labels=env_info["peg_labels_shuffled"], initial_plan=_plan)
                                print("Q2:", inp2)
                                agent_action_revised = agent.act(img, goal_img, inp2, next_image=future_img)

                            print("*" * 20, f"Step {t} reflection summary", "*" * 20)
                            print(f"Initial plan: {_plan}\n=> Revised action: {agent_action_revised}")
                            print("*" * 20, f"End of reflection summary", "*" * 20)

                        inference_time += (time.time() - inference_t0)

                    agent_act_list[t] = agent_action
                    print("A:", agent_action)

                    # process revised action
                    assert FLAGS.revise_action == (agent_action_revised is not None)
                    if agent_action_revised is not None:
                        try:
                            assert len(agent_action_revised.strip().split(' ')) <= 3, "Bad output format."
                            _p, _o = parse_act_txt(agent_action_revised)
                            assert _p in candidate_act_list and _o in env_info["peg_labels"], "Bad output format."
                        except Exception as e:
                            print(f"Error during parsing `agent_action_revised`({agent_action_revised}): {e}")
                            agent_action_revised = agent_action
                        agent_act_revised_list[t] = agent_action_revised
                        agent_action = agent_action_revised # Use the revised action as the final action to execute
                    print("A2:", agent_action)
                    
                    # parse and choose action
                    agent_action_primitive, agent_brick_color = parse_act_txt(agent_action)
                    if random.uniform(0, 1) < FLAGS.oracle_prob:
                        exec_action_primitive = oracle_action_primitive
                        exec_brick_color = oracle_brick_color
                    else:
                        exec_action_primitive = agent_action_primitive
                        exec_brick_color = agent_brick_color
                    exec_action = "done" if exec_action_primitive == "done" else " ".join([exec_action_primitive, exec_brick_color])
                    exec_act_list[t] = exec_action
                    
                    # add action to history
                    history.append(exec_action)

                    # execute
                    print("Executed action:", exec_action)
                    if exec_brick_color not in env.peg_colors:
                        print(f"Unknown object '{exec_brick_color}'")
                        continue
                    if exec_action == "done":
                        rollout_t = 0
                        break
                    else:
                        rollout_t0 = time.time()
                        err = env.act_txt(exec_action)
                        if err != 0:
                            print(f"Error {err} when executing `{exec_action}`")
                        rollout_t = time.time() - rollout_t0

                except Exception as e:
                    print(e)
                    if rollout_t is None:
                        rollout_t = time.time() - rollout_t0
                    break  # fail

                rollout_time += rollout_t

                # check success
                if env.is_success():
                    traj_succ = True
                    succ_cnt += 1
                    break
            
            last_img = env.read_pixels(camera_name=FLAGS.camera_name)
            total_time = time.time() - total_time_t0

            if env.is_recording:
                env.record_off()

            traj_key_frames.append(last_img)
            last_img_path = os.path.join(traj_dir, f"{len(question_list)}.png")
            Image.fromarray(last_img).save(last_img_path)
            goal_img_path = os.path.join(traj_dir, f"goal.png")
            Image.fromarray(goal_img).save(goal_img_path)

            # log data
            print("Success:", traj_succ)
            print(f"Inference time: {inference_time}, Rollout time: {rollout_time}")
            wandb_logger.log({
                "success": int(traj_succ), 
                "accumulated_success_cnt": succ_cnt,
                "accumulated_success_rate": succ_cnt / (traj_cnt + 1),
                "total_inference_time": inference_time,
                "total_rollout_time": rollout_time,
                "average_inference_time": inference_time / len(question_list),
                "average_rollout_time": rollout_time / len(question_list),
                "total_steps": len(question_list),
                "total_time": total_time
            }, step=traj_cnt)

            if FLAGS.record:
                wandb_logger.log_video(
                    np.stack(env.frames).transpose((0, 3, 1, 2)), fps=60,
                    caption=f"traj{traj_id}_seed{env_seed}_{reset_seed}_{['fail', 'succ'][traj_succ]}"
                )

            for i, question in enumerate(question_list):
                img_path = os.path.join(traj_dir, f"{i}.png")
                Image.fromarray(traj_key_frames[i]).save(img_path)

                entry = {
                    "trajectory_id": traj_id,
                    "board_id": board_id,
                    "env_seed": env_seed,
                    "reset_seed": reset_seed,
                    "step_id": i,
                    "action_description": exec_act_list[i],
                    "oracle_action": oracle_act_list[i],
                    "agent_action": agent_act_list[i],
                    "history": history_list[i],
                    "image": f"{traj_id}/{i}.png",
                    "next_image": f"{traj_id}/{i + 1}.png",
                    "final_goal_image": f"{traj_id}/goal.png",
                    "traj_success": int(traj_succ),
                    "traj_total_steps": len(question_list),
                    "object_descriptions": copy.deepcopy(env_info['peg_descriptions']),
                    "object_dependencies": copy.deepcopy(env_info["dependencies"])
                }
                if FLAGS.revise_action:
                    entry["agent_action_revised"] = agent_act_revised_list[i]
                    entry["agent_plan"] = copy.deepcopy(agent_plan_list[i])

                data.append(entry)

            idx_in_env += 1
            traj_id += 1
            traj_cnt += 1

        env.close()
        env_seed += 1
        board_id += 1
        board_cnt += 1

    # save data
    df = pd.DataFrame(data)
    meta_path = os.path.join(FLAGS.save_dir, "meta.csv")
    df.to_csv(meta_path)

    # log summary
    print("=" * 20)
    print("Total # boards:", board_cnt)
    print("Total # trajectories:", traj_cnt)
    print("Total # samples(steps):", len(data))
    succ_rate = succ_cnt / traj_cnt
    print(f"Success rate: {succ_rate} ({succ_cnt}/{traj_cnt})")
    wandb_logger.log({"success_rate": succ_rate})

    os.remove(xml_filename)


if __name__ == "__main__":
    app.run(main)
