import warnings
from tensordict.utils import implement_for, NestedKey
from torch import nn, Tensor
from torchrl.collectors.collectors import RandomPolicy
from torchrl.envs import (  # , StepCounter, ToTensorImage
    Compose,
    EnvBase,
    TransformedEnv,
)


from textwrap import indent
from typing import Any,Iterable
from tensordict._tensordict import unravel_key_list

import torch
import os
import wandb
from torchrl.envs import (
    StepCounter, 
    Compose,
    TransformedEnv,
    ToTensorImage, 
)
from torchrl.modules import SafeModule
from crew_algorithms.envs.channels import ToggleTimestepChannel
from crew_algorithms.envs.configs import EnvironmentConfig
from torchvision.utils import save_image
from torchrl.envs.transforms import Transform
from torchrl.data.tensor_specs import (
    TensorSpec,
)
import numpy as np

from crew_algorithms.envs.channels import ToggleTimestepChannel
from crew_algorithms.envs.configs import EnvironmentConfig
from torchvision.utils import save_image
import crew_algorithms.models as models

# from crew_algorithms.multimodal.split_key_transform import IndexSelectTransform
from crew_algorithms.utils.rl_utils import (
    convert_tensor_to_pil_image,
    make_base_env,
    unsqueeze_images_from_channel_dimension,
)


from tensordict import TensorDictBase
import random

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

def make_env(
    cfg: EnvironmentConfig,
    toggle_timestep_channel: ToggleTimestepChannel,
    device: str,
):
    """Creates an environment based on the configuration that can be used for
    a random policy.

    Args:
        cfg: The environment configuration to be used.
        toggle_timestep_channel: A Unity side channel that can be used
            to play/pause games.
        device: The device to perform environment operations on.

    Returns:
        The environment that can be used for the random policy.
    """
    env = TransformedEnv(
        make_base_env(cfg, device, toggle_timestep_channel=toggle_timestep_channel),
        Compose(
            # IndexSelectTransform([[torch.tensor([0])]], [[1]],
            # in_keys=[("agents", "observation", "obs_0_0")],
            # out_keys=[("agents", "observation", "feedback")])
            # ToTensorImage(in_keys=[("agents", "observation")], unsqueeze=True),
        ),
    )
    return env



def make_agent(device,num_of_seekers):
    model = models.IL(5,0.2,num_of_seekers).to(device)
    model = Make_model(
        model,
        in_keys = [('agents','observation','obs_0'),('agents','observation','obs_3')],
        out_keys = [('agents','action')],
    )

    return model.to(device)

class Make_model(SafeModule):
    def __init__(
        self,
        model,
        in_keys,
        out_keys,
    ):
        super().__init__(module = model, in_keys =in_keys,out_keys = out_keys)
        
    def forward(
        self,
        tensordict: TensorDictBase,
        *args,
        tensordict_out: TensorDictBase | None = None,
        **kwargs: Any,
    ) -> TensorDictBase:
        """When the tensordict parameter is not set, kwargs are used to create an instance of TensorDict."""
        try:
            if len(args):
                tensordict_out = args[0]
                args = args[1:]
                warnings.warn(
                    "tensordict_out will be deprecated soon.",
                    category=DeprecationWarning,
                )
            if len(args):
                raise ValueError(
                    "Got a non-empty list of extra agruments, when none was expected."
                )
            if self._kwargs is not None:
                kwargs.update(
                    {
                        kwarg: tensordict.get(in_key, None)
                        for kwarg, in_key in zip(self._kwargs, self.in_keys)
                    }
                )
                tensors = ()
            else:
                tensors = tuple(tensordict.get(in_key, None) for in_key in self.in_keys)
            try:
                tensors = self._call_module(tensors, **kwargs)
            except Exception as err:
                if any(tensor is None for tensor in tensors) and "None" in str(err):
                    none_set = {
                        key
                        for key, tensor in zip(self.in_keys, tensors)
                        if tensor is None
                    }
                    raise KeyError(
                        "Some tensors that are necessary for the module call may "
                        "not have not been found in the input tensordict: "
                        f"the following inputs are None: {none_set}."
                    ) from err
                else:
                    raise err
            if isinstance(tensors, (dict, TensorDictBase)):
                if isinstance(tensors, dict):
                    keys = unravel_key_list(list(tensors.keys()))
                    values = tensors.values()
                    tensors = dict(zip(keys, values))
                tensors = tuple(tensors.get(key, None) for key in self.out_keys)
            if not isinstance(tensors, tuple):
                tensors = (tensors,)
            
            final_tensors = (torch.zeros((int(self.module.num_of_seekers),2)).cuda()+51,)

            tensors = final_tensors 

            tensordict_out = self._write_to_tensordict(
                tensordict, tensors, tensordict_out
            )

            return tensordict_out
        
        except Exception as err:
            module = self.module
            if not isinstance(module, nn.Module):
                try:
                    import inspect

                    module = inspect.getsource(module)
                except OSError:
                    # then we can't print the source code
                    pass
            module = indent(str(module), 4 * " ")
            in_keys = indent(f"in_keys={self.in_keys}", 4 * " ")
            out_keys = indent(f"out_keys={self.out_keys}", 4 * " ")
            raise RuntimeError(
                f"TensorDictModule failed with operation\n{module}\n{in_keys}\n{out_keys}."
            ) from err
    
    def _write_to_tensordict(
        self,
        tensordict: TensorDictBase,
        tensors: list[Tensor],
        tensordict_out: TensorDictBase | None = None,
        out_keys: Iterable[NestedKey] | None = None,
    ) -> TensorDictBase:
        if out_keys is None:
            out_keys = self.out_keys_source
        if tensordict_out is None:
            tensordict_out = tensordict

        tensordict_out.set(out_keys[0], tensors[0])
        
        return tensordict_out

def save_images(env_cfg, data_view, base_directory:str, frames_per_batch:int, batch:int):
    """Saves individual images collected from yje policy.

    Saves images in a directory structure where you have the root data directory,
    and then one subfolder for each observation. The subfolder contains all of the
    images associated with that observation.

    Args:
        env_cfg: The environment configuration to be used.
        data_view: The current data collected from the data collector.
        base_directory: The root directory to store the data at.
        frames_per_batch: The number of frames in each batch from the collector.
        batch: The current batch number from the collector.
    """
    num_seeker = env_cfg['envs']['num_seekers']

    for i, single_data_view in enumerate(data_view.unbind(0)):

        os.makedirs(base_directory+f"observation", exist_ok=True)
        
        stack_obs = single_data_view["agents", "observation", "obs_2"]
        stack_agent = single_data_view["agents", "observation", "obs_3"]
        stack_next = single_data_view["next", 'agents', "observation", "obs_3"]
        

        for agent_id in range(num_seeker):
            obs = stack_obs[agent_id,:,:,:]
            os.makedirs(base_directory+f"observation/agent_{agent_id}", exist_ok=True)
            img = obs.permute(2, 0, 1)
            j = frames_per_batch * batch + i
            img = convert_tensor_to_pil_image(img)
            
            is_human_control = bool(stack_agent[agent_id,-2])
            str1 = ""
            if is_human_control:
                str1 = "H"
            img.save(base_directory+f"observation/agent_{agent_id}/{j}{str1}.png")
                
            agent_file = os.path.join(base_directory, f"agent_{agent_id}.txt")
            next_file = os.path.join(base_directory, f"next.txt_{agent_id}")

            

            agent_list = stack_agent[agent_id,[0,1,-1]].tolist()
            agent_list = [round(a,2) for a in agent_list]

            next_list = stack_next[agent_id,[0,1,-1]].tolist()
            next_list = [round(a,2) for a in next_list]


            with open(agent_file, 'a') as file:
                file.write(str(agent_list) + '\n')

            # Writing to next.txt
            with open(next_file, 'a') as file:
                file.write(str(next_list) + '\n')


