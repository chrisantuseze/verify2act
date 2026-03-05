import readline
import argparse
from roboworld.envs.mujoco.franka.franka_assembly import (
    FrankaAssemblyEnv, AssemblyOracle
)
from roboworld.agent.oracle import OracleAgent
from roboworld.envs.generator import generate_xml
from roboworld.envs.asset_path_utils import full_path_for
from roboworld.constants import ACTION_PRIMITIVES


class HistoryConsole:
    def __init__(self, history_file='.command_history'):
        self.history_file = history_file
        try:
            readline.read_history_file(history_file)
            readline.set_history_length(100)
        except FileNotFoundError:
            pass
    
    def save_history(self):
        try:
            readline.write_history_file(self.history_file)
        except:
            pass

    def input(self, prompt=">>> "):
        try:
            return input(prompt)
        except (EOFError, KeyboardInterrupt):
            return "exit"


def interact(env_seed, reset_seed=1):

    render_mode = 'window'
    console = HistoryConsole()
    xml_filename = full_path_for("tmp.xml")
    xml, info = generate_xml(env_seed)
    xml.write_to_file(filename=xml_filename)

    board_name = "brick_1"
    fixture_name = None  # "fixture"
    peg_ids = [j+1 for j in range(1, info['n_bodies'])]
    peg_names = [f"brick_{j+1}" for j in range(1, info['n_bodies'])]
    peg_descriptions = [info['brick_descriptions'][peg_name] for peg_name in peg_names]
    peg_labels = [" ".join(pd.split()[:1]) for pd in peg_descriptions]

    env = FrankaAssemblyEnv(board_name=board_name, fixture_name=fixture_name, peg_names=peg_names,
                            peg_descriptions=peg_descriptions, render_mode=render_mode, frame_skip=20,
                            model_name=xml_filename, magic_attaching=True)
    oracle = AssemblyOracle(env=env, brick_ids=peg_ids, brick_descriptions=peg_descriptions,
                            dependencies=info["dependencies"])
    oracle_agent = OracleAgent(oracle)

    env.reset(seed=reset_seed)
    env.render()

    step = 0

    while True:
        step += 1
        oracle_action = oracle_agent.act()

        print(
            "=" * 80,
            f"[Step {step}]",
            f"Please enter the action in the format '<act> <obj>'.",
            f"<act>: {', '.join([f'[{i+1}]{a}' for i, a in enumerate(ACTION_PRIMITIVES)])}.",
            f"<obj>: {', '.join([f'[{i+1}]{l}' for i, l in enumerate(peg_labels)])}.",
            f"Directly press 'Enter' to perform the oracle action ({oracle_action}).",
            "-" * 80,
            sep="\n"
        )

        act, obj = None, None
        while True:
            inp = console.input()
            inp_list = inp.split()
            if len(inp_list) == 0 or inp_list[0] == "expert":
                inp_list = oracle_action.split()  # use expert output
            act = " ".join(inp_list[:max(1, len(inp_list) - 1)])
            if str.isdigit(act):
                if not 1 <= int(act) <= len(ACTION_PRIMITIVES):
                    print(f"Unknown action '{act}'")
                    continue
                else:
                    act = ACTION_PRIMITIVES[int(act) - 1]
            if act not in ACTION_PRIMITIVES + ["done", "exit", "q"]:
                print(f"Unknown action '{act}'")
                continue
            if act == "done":
                break
            if act in ["exit", "q"]:
                console.save_history()
                return
            obj = inp_list[-1]
            if str.isdigit(obj):
                if not 1 <= int(obj) <= len(peg_labels):
                    print(f"Unknown object '{obj}'")
                    continue
                else:
                    obj = peg_labels[int(obj) - 1]
            if obj not in peg_labels:
                print(f"Unknown object '{obj}'")
                continue
            break

        if act == "done":
            break

        action_text = f"{act} {obj}"
        print(f"Executing `{action_text}`...")
        env.act_txt(action_text)
        env.render()

    env.close()
    console.save_history()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_seed", type=int, default=1000000)
    parser.add_argument("--reset_seed", type=int, default=1)
    args = parser.parse_args()
    interact(args.env_seed, args.reset_seed)
