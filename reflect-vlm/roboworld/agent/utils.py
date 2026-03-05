import textwrap

def parse_act_txt(act_txt):
    if "done" in act_txt:
        return "done", None
    if str.isdigit(act_txt.split(". ")[0]):
        act_txt = act_txt.split(". ")[1]
    act_txt_splitted = act_txt.split()
    act = " ".join(act_txt_splitted[:max(1, len(act_txt_splitted) - 1)])
    obj = act_txt_splitted[-1]
    return act, obj


def get_prompt(version, history, obj_labels, initial_plan=None):

    if version == "propose":
        history_txt = f"[{', '.join(history[-10:])}]"
        obj_labels_txt = f"[{', '.join(obj_labels)}]"
        prompt = (
            f"There is a puzzle consisting of a board and several pieces with different colors on the table. The goal is to assemble the puzzle with the robot arm. "
            f"In each step, one of the following four actions can be taken: pick up [obj], put down [obj], reorient [obj], and insert [obj], "
            f"where [obj] refers to the piece to be manipulataed. "
            f"The image of the goal state is: <image>. The image of the current state is: <image>. The most recently executed actions are: {history_txt}. "
            f"What action should be taken next? Note that [obj] should be a color chosen from the following list: {obj_labels_txt}."
        )
    elif version == "reflect":
        assert initial_plan is not None
        history_txt = f"[{', '.join(history[-10:])}]"
        obj_labels_txt = f"[{', '.join(obj_labels)}]"
        init_plan_txt = f"[{', '.join(initial_plan)}]"
        prompt = (
            f"There is a puzzle consisting of a board and several pieces with different colors on the table. The goal is to assemble the puzzle with the robot arm. "
            f"In each step, one of the following four actions can be taken: pick up [obj], put down [obj], reorient [obj], and insert [obj], "
            f"where [obj] refers to the piece to be manipulataed. "
            f"The image of the goal state is: <image>. The image of the current state is: <image>. The most recently executed actions are: {history_txt}. "
            f"The next five steps planned by the model is {init_plan_txt}, from which we are going to only execute the first action. Note that if the full plan was executed sequentially, the future state would be: <image>. "
            f"What action should be taken for the immediate next step? Note that [obj] should be a color chosen from the following list: {obj_labels_txt}. You can modify the initial plan if it leads to an undesired future state."
        )
    else:
        raise ValueError(f"Unknown version `{version}`")
    return prompt


if __name__ == "__main__":
    print(parse_act_txt("done"))
    print(parse_act_txt("pick up blue"))
    print(get_prompt(history=['pick up red', 'reorient red', 'insert red'], obj_labels=['red', 'blue'], version='propose'))
    print(get_prompt(history=['pick up red', 'reorient red', 'insert red'], obj_labels=['red', 'blue'], version='reflect', initial_plan=['pick up blue', 'reorient blue', 'insert blue', 'pick up green', 'insert green']))
