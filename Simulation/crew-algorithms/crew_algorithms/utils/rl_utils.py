import os
import platform

import torch
import torchvision.transforms.functional as F
import wandb
from torchrl.collectors import aSyncDataCollector, SyncDataCollector, MultiaSyncDataCollector
from torchrl.envs import EnvBase

from crew_algorithms.envs.channels import ToggleTimestepChannel, WrittenFeedbackChannel
from crew_algorithms.envs.configs import EnvironmentConfig
from crew_algorithms.envs.unity import UnityEnv
from crew_algorithms.utils.common_utils import find_free_port
from mlagents_envs.side_channel.engine_configuration_channel import (
    EngineConfigurationChannel,
)


def make_base_env(
    env_cfg: EnvironmentConfig,
    device: str,
    toggle_timestep_channel: ToggleTimestepChannel | None = None,
    written_feedback_channel: WrittenFeedbackChannel | None = None,
):
    """Creates the base Unity Environment to be used without any transforms applied.

    Args:
        env_cfg: The environment configuration.
        device: The device to perform environment operations on.
        toggle_timestep_channel: A Unity side channel that can be used
            to play/pause games.
        written_feedback_channel: A Unity side channel that can be used
            to share written feedback at the end of each episode.

    Returns:
        A `UnityEnv` object that can be used to interact with the environment.
    """
    # env_cfg.unity_server_build_path = env_cfg.unity_server_build_path_linux
    # env_cfg.unity_server_build_path = env_cfg.unity_server_build_path_osx
    if "Linux" in platform.system():
        env_cfg.unity_server_build_path = env_cfg.unity_server_build_path_linux
    elif "Darwin" in platform.system():
        env_cfg.unity_server_build_path = env_cfg.unity_server_build_path_osx

    num_player_args = []
    if hasattr(env_cfg, "num_agents"):
        num_player_args = ["-NumAgents", f"{env_cfg.num_agents}"]
    elif hasattr(env_cfg, "num_hiders") and hasattr(env_cfg, "num_seekers"):
        num_player_args = [
            "-NumHiders",
            f"{env_cfg.num_hiders}",
            "-NumSeekers",
            f"{env_cfg.num_seekers}",
        ]

    channel_args = []
    side_channels = []
    if toggle_timestep_channel:
        side_channels.append(toggle_timestep_channel)
        channel_args += [
            "-ToggleTimestepChannelID",
            str(toggle_timestep_channel.channel_id),
        ]
    if written_feedback_channel:
        side_channels.append(written_feedback_channel)
        channel_args += [
            "-WrittenFeedbackChannelID",
            str(written_feedback_channel.channel_id),
        ]

    engine_configuration_channel = EngineConfigurationChannel()
    side_channels.append(engine_configuration_channel)

    if env_cfg.time_scale > 1.0:
        engine_configuration_channel.set_configuration_parameters(
            width=900,
            height=600,
            quality_level=10,
            time_scale=env_cfg.time_scale,
            target_frame_rate=50,
        ) 

    base_env = UnityEnv(
        str(env_cfg.unity_server_build_path),
        no_graphics=env_cfg.no_graphics,
        side_channels=side_channels,
        additional_args=[
            *channel_args,
            *num_player_args,
        ],
        log_folder=str(env_cfg.log_folder_path),
        base_port=find_free_port(),
        timeout_wait=60 * 60 * 24,
        device=device,
        frame_skip=1,
    )
    return base_env


def convert_tensor_to_pil_image(img: torch.Tensor):
    """Converts a PyTorch tensor to a PIL image.

    Args:
        img: The tensor to convert.

    Returns:
        The PIL image.
    """
    pil_image = F.to_pil_image(img)
    return pil_image


def unsqueeze_images_from_channel_dimension(
    env_cfg, tensor: torch.Tensor, dim=-3
) -> torch.Tensor:
    """Unsqueezes images that are concatenated along the channel dimension.

    Takes a tensor of shape (..., num_images * num_channels,
    width, height) and produces a tensor of shape (..., num_images,
    num_channels, width, height).

    Args:
        env_name: The name of the environment the tensor came from.
        tensor: Tensor with dimensions
            (..., num_images * num_channels, width, height).
        dim: The dimension along which to unsqueeze.

    Returns:
        Tensor with dimensions (..., num_images, num_channels, width, height).
    """
    # print(tensor.shape, dim)
    num_images = tensor.shape[dim] // env_cfg.num_channels
    return tensor.unflatten(dim, (num_images, env_cfg.num_channels))


def make_collector(cfg, env: EnvBase, policy, device: str):
    """Makes a collector to collect samples from the environment.


    Args:
        cfg: The collector configuration.
        env: The environment to collect data from.
        device: The device to run the collector on.

    Returns:
        A collector to collect data samples.
    """

    # env = lambda: env
    # from tensordict.nn import TensorDictModule
    # from torch import nn
    # from torchrl.envs.libs.gym import GymEnv

    # policy = TensorDictModule(nn.Linear(3, 1),
    #   in_keys=["observation"], out_keys=["action"])

    # env = lambda: GymEnv("Pendulum-v1", device="cpu")
    # from crew_algorithms.envs.unity import UnityEnv

    # env = lambda: UnityEnv(
    #     "../crew-dojo/Builds/FindTreasure-StandaloneLinux64-Server/Unity.x86_64",
    #     # str(env_cfg.unity_server_build_path),
    #     # side_channels=side_channels,
    #     additional_args=[
    #         *channel_args,
    #         *num_player_args,
    #     ],
    #     # log_folder=str(env_cfg.log_folder_path),
    #     base_port=find_free_port(),
    #     timeout_wait=60 * 60 * 24,
    #     device=device,
    #     frame_skip=1,
    # )
    # from crew_algorithms.multimodal_feedback.utils import make_env
    # env = lambda: make_env(cfg.envs, None, device)

    collector = aSyncDataCollector(
        env,
        policy,
        frames_per_batch=cfg.frames_per_batch,
        total_frames=cfg.total_frames,
        device=device,
        split_trajs=False,
        exploration_mode=cfg.exploration_mode,
    )
    return collector


def log_policy(cfg, policy, logger):
    """Logs the policy.

    Args:
        cfg: The configuration settings.
        policy: The policy to log.
        logger: The logger to use for logging.
    """
    save_path = os.path.join(logger.experiment.dir, "policy.pth")
    torch.save(policy.state_dict(), save_path)

    policy_artifact = wandb.Artifact(
        f"{cfg.wandb.project}-{cfg.envs.name}-policy", type="model"
    )
    policy_artifact.add_file(save_path)

    logger.experiment.log_artifact(policy_artifact)


def make_shared_model(model):
    for param in model.parameters():
        param.data = param.data.share_memory_()
    return model


def copy_params(shared_model, local_model):
    for shared_param, local_param in zip(
        shared_model.parameters(), local_model.parameters()
    ):
        if shared_param.grad is not None:
            local_param._grad = shared_param.grad.clone()
        local_param.data = shared_param.data.clone()
