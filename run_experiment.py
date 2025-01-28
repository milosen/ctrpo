import argparse
import shlex
import subprocess
import torch

navi_robots = ['Ant', 'Car', 'Doggo', 'Point', 'Racecar']
navi_tasks = ['Button', 'Circle', 'Goal', 'Push']
diffculies = ['1', '2']
vel_robots = ['Ant', 'HalfCheetah', 'Hopper', 'Walker2d', 'Swimmer', 'Humanoid']
vel_tasks = ['Velocity']

benchmark = ["SafetyAntVelocity-v1", 
                   "SafetyHalfCheetahVelocity-v1",
                   "SafetyHumanoidVelocity-v1",
                   "SafetyHopperVelocity-v1",
                   "SafetyCarButton1-v0", 
                   "SafetyPointGoal1-v0",
                   "SafetyRacecarCircle1-v0",
                   "SafetyPointPush1-v0"]

algos = [
            "c-trpo",
            "cpo",
            "pcpo",
            "cppo_pid",
            "ppo_lag",
            "trpo_lag",
            "focops",
            "cup",
            "p3o",
            "ipo",
            "ppo"
]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=benchmark,
        help="the ids of the environment to benchmark",
    )
    parser.add_argument(
        "--algo",
        nargs="+",
        default=algos,
        help="the ids of the algorithm to benchmark",
    )
    parser.add_argument(
        "--num-seeds", type=int, default=5, help="the number of random seeds"
    )
    parser.add_argument(
        "--task_idx", type=int, default=None, help="the task number to run from the flattened task array"
    )
    parser.add_argument(
        "--start-seed", type=int, default=0, help="the number of the starting seed"
    )
    parser.add_argument(
        "--workers", type=int, default=1, help="the number of workers to run benchmark experimenets",
    )
    parser.add_argument(
        "--experiment", type=str, default="benchmark", help="name of the experiment"
    )
    parser.add_argument(
        "--total-steps", type=int, default=10000000, help="total number of steps"
    )
    parser.add_argument(
        "--num-envs", type=int, default=10, help="number of environments to run in parallel"
    )
    parser.add_argument(
        "--steps-per-epoch", type=int, default=20000, help="number of steps per epoch"
    )
    args = parser.parse_args()

    return args


def run_experiment(command: str):
    command_list = shlex.split(command)
    print(f"running {command}")
    fd = subprocess.Popen(command_list)
    return_code = fd.wait()
    assert return_code == 0


if __name__ == "__main__":
    args = parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    commands = []
    
    if args.task_idx is not None:
        seeds = [args.start_seed + 1000*seed for seed in range(args.num_seeds)]
        tasks = args.tasks
        algos = args.algo
        
        args.num_seeds = 1
        
        experiments = [(task, algo, seed) for task in tasks for algo in algos for seed in seeds]
        assert args.task_idx < len(experiments), "task index exceeds number of tasks"
        
        task, algo, seed = experiments[args.task_idx]
        
        print("running experiment {(task, algo, seed)}")
        
        args.start_seed = seed
        args.tasks = [task]
        args.algo = [algo]
    
    for seed in range(0, args.num_seeds):
        for task in args.tasks:
            if "Doggo" in task:
                args.total_steps = str(100000000)
                args.steps_per_epoch = str(200000)
                args.num_envs = str(20)
            for algo in args.algo:
                commands += [
                    " ".join(
                        [
                            f"python ctrpo/algo/{algo}.py",
                            "--task",
                            task,
                            "--seed",
                            str(args.start_seed + 1000*seed),
                            "--write-terminal",
                            "True",
                            "--experiment",
                            args.experiment,
                            "--total-steps",
                            str(args.total_steps),
                            "--num-envs",
                            str(args.num_envs),
                            "--steps-per-epoch",
                            str(args.steps_per_epoch),
                            "--device", device,
                        ]
                    )
                ]


    print("======= commands to run:")
    for command in commands:
        print(command)

    if args.workers > 0:
        from concurrent.futures import ThreadPoolExecutor

        executor = ThreadPoolExecutor(
            max_workers=args.workers, thread_name_prefix="safepo-benchmark-worker-"
        )
        for command in commands:
            executor.submit(run_experiment, command)
        executor.shutdown(wait=True)
    else:
        print(
            "not running the experiments because --workers is set to 0; just printing the commands to run"
        )
