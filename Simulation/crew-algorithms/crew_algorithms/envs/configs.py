
from pathlib import Path

from attrs import define
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


@define(auto_attribs=True)
class EnvironmentConfig:
    name: str = MISSING
    num_stacks: int = 1
    num_channels: int = 3
    unity_server_build_path_osx: Path = MISSING
    unity_server_build_path_linux: Path = MISSING
    unity_server_build_path: Path = MISSING
    log_folder_path: Path = MISSING
    human_delay: int = 1
    no_graphics: bool = False

@define(auto_attribs=True)
class HideAndSeekConfig(EnvironmentConfig):
    name: str = "hide_and_seek"
    num_hiders: int = 1
    num_seekers: int = 2
    start_seed: int = 1
    num_games: int = 150
    decision_frequency: float = 0.2
    num_seekers_with_policy: int = 0
    base_policy: str = "Heuristic"
    addon_policy: str = "IL"
    time_scale: float = 1.0

    @property
    def num_player_args(self) -> list[str]:
        return [
            "-NumHiders",
            f"{self.num_hiders}",
            "-NumSeekers",
            f"{self.num_seekers}",
        ]

def register_env_configs() -> None:
    cs = ConfigStore.instance()
    cs.store(group="envs", name="base_hide_and_seek", node=HideAndSeekConfig)
