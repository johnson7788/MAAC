from pathlib import Path


def _get_latest_run_from(model_dir: Path, run_num: int):
    run_num = 'run%i' % run_num
    run_dir = model_dir / run_num

    if not run_dir.exists():
        return None

    incremental_dir = run_dir / "incremental"

    if not incremental_dir.exists():
        return None

    prefix = "model_ep"
    suffix = ".pt"
    nums = []
    for file in incremental_dir.iterdir():
        name = file.name
        if name.startswith(prefix) and name.endswith(suffix):
            num = int(name[len(prefix):len(name) - len(suffix)])
            nums.append(num)

    if len(nums) == 0:
        return None

    file = incremental_dir / (prefix + str(max(nums)) + suffix)
    print(file)

    return file


def run_setup(model_name: str, get_latest_model: bool=False):
    """
    Returns [model_path (None if get_latest_model is False), run_num, run_dir, log_dir]
    """
    rets = []

    model_dir = Path('./models') / model_name
    if not model_dir.exists():
        run_num = 1
    else:
        exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in
                         model_dir.iterdir() if
                         str(folder.name).startswith('run')]
        if len(exst_run_nums) == 0:
            run_num = 1
        else:
            run_num = max(exst_run_nums) + 1

    curr_run = 'run%i' % run_num
    run_dir = model_dir / curr_run

    model_path = None
    if get_latest_model:
        model_path = _get_latest_run_from(model_dir, run_num - 1)
    rets.append(model_path)

    rets.append(run_num)
    rets.append(run_dir)

    log_dir = run_dir / 'logs'
    rets.append(log_dir)

    return rets
