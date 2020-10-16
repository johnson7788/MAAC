from algorithms.attention_sac import AttentionSAC
from envs.halite.halite_env import HaliteRunHelper
from utils.savehelper import run_setup


def run(model_name: str):
    model_path, run_num, run_dir, log_dir = run_setup(model_name, get_latest_model=True)

    if model_path is None:
        print("Couldn't find model!")
        return

    model = AttentionSAC.init_from_save(model_path)

    model.prep_rollouts(device='cpu')

    run_env: HaliteRunHelper = HaliteRunHelper()

    run_env.simulate(lambda o: model.step(o, explore=True), agent_count=2)


if __name__ == "__main__":
    run("halite")
