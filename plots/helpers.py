import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import matplotlib as mpl


ALGOS = [ 
    "C-TRPO",
    "CPO", 
    "PCPO", 
    "CPPO-PID", 
    "PPO-Lag",
    "TRPO-Lag",
    "FOCOPS",
    "CUP",
    "P3O",
    "IPO",
]

ENVS = (
    "SafetyAntVelocity-v1", 
    "SafetyHalfCheetahVelocity-v1",
    "SafetyHumanoidVelocity-v1",
    "SafetyHopperVelocity-v1",
    "SafetyCarButton1-v0", 
    "SafetyPointGoal1-v0",
    "SafetyRacecarCircle1-v0",
    "SafetyPointPush1-v0"
)


def get_df(
    algo_subset=("c-trpo-entropy", "cpo"), 
    experiments=("c-trpo_beta_1", "c-trpo_beta_0_5", "benchmark_array"), 
    env_subset=None
):
    df = pd.DataFrame()
    for exp in experiments:
        try:
            bm_folder = os.path.join("..", "data", "runs", exp)
            envs = [f for f in os.listdir(bm_folder) if not f.startswith('.') and not f.endswith('.pdf')]
            envs = set(envs).intersection(env_subset) if env_subset else envs
        except FileNotFoundError:
            print("Not found")
            continue
        for env in envs:
            env_folder = os.path.join(bm_folder, env)
            algos = [f for f in os.listdir(env_folder) if not f.startswith('.') and not f.endswith('.pdf')]
            algos = set(algos).intersection(algo_subset) if algo_subset else algos
            for algo in algos:
                algo_folder = os.path.join(env_folder, algo)
                for seed in [f for f in os.listdir(algo_folder) if not f.startswith('.') and not f.endswith('.pdf')]:
                    progress_csv = os.path.join(algo_folder, seed, "progress.csv")
                    try:
                        new_df = pd.read_csv(progress_csv)
                        new_df["seed"] = seed
                        new_df["algo"] = algo + f" ({exp})"
                        new_df["env"] = env
                        new_df["exp"] = exp
                        new_df = new_df.sort_values(by=['Train/TotalSteps'], ascending=True)
                        new_df["Metrics/EpCumCostViolation"] = (new_df["Metrics/EpCost"] - 25.0).clip(lower=0)
                        new_df["Metrics/EpCumCostViolation"] = new_df["Metrics/EpCumCostViolation"].cumsum()
                        df = pd.concat([df, new_df], ignore_index=True)
                    except (pd.errors.EmptyDataError, pd.errors.ParserError) as e:
                        print(f"We have a {e.__class__.__name__} for {env}/{algo}/{seed}.")
                        continue
    #df = df.sort_values(by=['algo'], ascending=True)
    
    return df


def plot_env(df, env, xmax=10e6, diagnostics=True, cost_ci=True, 
             alpha=1., start_others_at=2, plot_cost_limit=True):
    
    df = df[(df["env"] == env)]
    
    _, axs = plt.subplots(1, 3, figsize=(10, 3), layout = 'tight')
    
    sns.lineplot(df, x="Train/TotalSteps", y="Metrics/EpCost", ax=axs[1], hue="algo")
    sns.lineplot(df, x="Train/TotalSteps", y="Metrics/EpRet", ax=axs[0], hue="algo")
    sns.lineplot(df, x="Train/TotalSteps", y="Metrics/EpCumCostViolation", ax=axs[2], hue="algo")
    
    axs[1].plot([0, xmax],[25,25], linestyle="dashed", label="cost limit", color="black")
    
    axs[1].set_ylim(0, ymax=100)
    # axs[2].set_ylim(0, ymax=1e6)
    
    axs[0].set_xlim(0, xmax=xmax)
    axs[1].set_xlim(0, xmax=xmax)
    axs[2].set_xlim(0, xmax=xmax)

    handles, labels = axs[1].get_legend_handles_labels()
    
    axs[0].legend(handles=handles, labels=labels, prop={'size': 6.5})
    axs[1].legend([],[], frameon=False)
    axs[2].legend([],[], frameon=False)
    
    r_axs = axs
     
    if diagnostics:
        metrics = ["Loss/Loss_cost_critic", "Train/KL", "Train/target_div", "Train/div"]
        _, axs = plt.subplots(1, len(metrics), figsize=(10, 3))

        for ax, metric in zip(axs, metrics):
            if metric in df.columns:
                sns.lineplot(df[df["algo"] == "C-TRPO"], x="Train/TotalSteps", y=metric, ax=ax, hue="algo")
    
    axs[0].set_ylabel("Return")
    axs[1].set_ylabel("Cost")
    axs[2].set_ylabel("Cost Regret")

    for ax in axs:
        ax.set_xlabel("Steps")
    
    return r_axs


def load_benchmark_data(load_ppo=True):
    df = get_df(algo_subset = ("c-trpo", "cpo", "pcpo", "cppo_pid", "ppo_lag", "trpo_lag", "focops", "cup", "p3o", "ipo", "ppo"),
            env_subset = ENVS,
            experiments = ("benchmark", "ablation"))
    
    algos = ALGOS + ["PPO"] if load_ppo else ALGOS  
    algos_replace = [
        "c-trpo (ablation)",  # during ablation we noticed we did not use beta=1 in the benchmark
        "cpo (benchmark)", 
        "pcpo (benchmark)", 
        "cppo_pid (benchmark)", 
        "ppo_lag (benchmark)",
        "trpo_lag (benchmark)",
        "focops (benchmark)",
        "cup (benchmark)",
        "p3o (benchmark)",
        "ipo (benchmark)",
    ]
    algos_replace = algos_replace + ["ppo (benchmark)"] if load_ppo else algos_replace  

    df = df.replace(algos_replace, algos)

    df['algo'] = pd.Categorical(df['algo'], algos)

    df = df.sort_values(by=['algo'])
    
    return df


ALGOS_ABLATION = [
    "C-TRPO (no hyst.)", 
    "C-TRPO (hyst.)", 
    "CPO (hyst.)", 
    "CPO (no hyst.)"
]


def load_ablation_data(load_ppo=True):
    df = get_df(algo_subset = ("c-trpo", "c-trpo-hyst", "cpo", "cpo-hyst", "ppo"),
            env_subset = ENVS,
            experiments = ("ablation", "benchmark"))
    
    algos = ALGOS_ABLATION + ["PPO"] if load_ppo else ALGOS  
    algos_replace = [
        "c-trpo (ablation)", 
        "c-trpo-hyst (ablation)", 
        "cpo-hyst (ablation)", 
        "cpo (benchmark)"
    ]
    algos_replace = algos_replace + ["ppo (benchmark)"] if load_ppo else algos_replace  

    df = df.replace(algos_replace, algos)

    df['algo'] = pd.Categorical(df['algo'], algos)

    df = df.sort_values(by=['algo'])

    df['algo'] = pd.Categorical(df['algo'], algos)

    df = df.sort_values(by=['algo'])
    
    return df
