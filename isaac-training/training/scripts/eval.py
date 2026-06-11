"""Evaluation entrypoint: load a checkpoint and run deterministic evaluation once."""

import datetime
import os

import hydra
import torch.nn as nn
import wandb
from omni.isaac.kit import SimulationApp
from torchrl.envs.transforms import Compose, TransformedEnv
from torchrl.envs.utils import ExplorationType


FILE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cfg")


def print_eval_metrics(eval_info, prefix="eval", label="eval"):
    metric_keys = [
        f"{prefix}/success_rate",
        f"{prefix}/collision_rate",
        f"{prefix}/wall_collision_rate",
        f"{prefix}/below_bound_rate",
        f"{prefix}/above_bound_rate",
        f"{prefix}/deadlock_rate",
        f"{prefix}/deadlock_escape_success_rate",
        f"{prefix}/conditioned/success_given_deadlock",
        f"{prefix}/time_limit_rate",
        f"{prefix}/deadlock_steps",
        f"{prefix}/deadlock_step_ratio",
        f"{prefix}/stats.return",
        f"{prefix}/stats.episode_len",
        f"{prefix}/stats.reach_goal",
        f"{prefix}/stats.collision",
        f"{prefix}/stats.wall_collision",
        f"{prefix}/stats.stuck",
        f"{prefix}/stats.stuck_steps",
        f"{prefix}/stats.truncated",
    ]
    metric_parts = []
    for key in metric_keys:
        value = eval_info.get(key)
        if value is not None:
            metric_parts.append(f"{key}={value:.4f}")
    if metric_parts:
        print(f"[NavRL]: {label} metrics | " + " | ".join(metric_parts))


def set_policy_eval_mode(policy):
    # PPO.train() is overloaded for optimization, so avoid calling policy.eval().
    nn.Module.train(policy.feature_extractor, False)
    nn.Module.train(policy.actor, False)
    nn.Module.train(policy.critic, False)
    nn.Module.train(policy.value_norm, False)
    policy.training = False


@hydra.main(config_path=FILE_PATH, config_name="train", version_base=None)
def main(cfg):
    sim_app = SimulationApp({"headless": cfg.headless, "anti_aliasing": 1})

    checkpoint_value = cfg.get("checkpoint", None)
    checkpoint_name = (
        os.path.splitext(os.path.basename(os.path.expanduser(str(checkpoint_value))))[0]
        if checkpoint_value is not None
        else "missing_checkpoint"
    )

    run = wandb.init(
        project=cfg.wandb.project,
        name=f"{cfg.wandb.name}/eval/{checkpoint_name}/{datetime.datetime.now().strftime('%m-%d_%H-%M')}",
        entity=cfg.wandb.entity,
        config=cfg,
        mode=cfg.wandb.mode,
        id=wandb.util.generate_id(),
    )
    run.define_metric("eval/*")

    from env import NavigationEnv
    from omni_drones.controllers import LeePositionController
    from omni_drones.utils.torchrl.transforms import VelController
    from ppo import PPO
    from utils import evaluate, load_policy_checkpoint, resolve_eval_style

    env = NavigationEnv(cfg)

    controller = LeePositionController(9.81, env.drone.params).to(cfg.device)
    vel_transform = VelController(controller, yaw_control=False)
    transformed_env = TransformedEnv(env, Compose(vel_transform)).eval()
    transformed_env.set_seed(cfg.seed)

    policy = PPO(cfg.algo, transformed_env.observation_spec, transformed_env.action_spec, cfg.device)
    load_policy_checkpoint(policy, cfg.checkpoint, cfg.device, required=True)
    set_policy_eval_mode(policy)

    eval_task_mode, eval_label = resolve_eval_style(cfg)
    eval_info = evaluate(
        env=transformed_env,
        policy=policy,
        seed=cfg.seed,
        cfg=cfg,
        exploration_type=ExplorationType.MEAN,
        prefix="eval",
        eval_task_mode=eval_task_mode,
    )
    eval_info["checkpoint"] = str(checkpoint_value or "")

    run.log(eval_info)
    print_eval_metrics(eval_info, prefix="eval", label=eval_label)

    wandb.finish()
    sim_app.close()


if __name__ == "__main__":
    main()
