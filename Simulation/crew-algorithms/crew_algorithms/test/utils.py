import warnings
from tensordict.utils import implement_for, NestedKey
from torch import nn, Tensor

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

from crew_algorithms.utils.rl_utils import (
    convert_tensor_to_pil_image,
    make_base_env,
    unsqueeze_images_from_channel_dimension,
)


from tensordict import TensorDictBase
import models
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
    a imitation learning

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
            MakeBinaryMask(
                in_keys=[("agents", "observation","obs_0"),('agents','observation','obs_3')],
                out_keys=[("agents", "observation",'obs_0')],
                num_of_seekers= cfg['num_seekers']
            ),
            StepCounter(),
        ),
    )
    return env

def make_agent(device,num_of_frames,num_of_seekers,num_of_seeker_with_policy,decision_frequency,base_policy,addon_policy):
    model = models.Mix(base_policy = base_policy,addon_policy = addon_policy,num_of_frames=num_of_frames,decision_frequency=decision_frequency,num_of_seekers=num_of_seekers)
    
    model = Make_model(
        model,
        in_keys = [('agents','observation','obs_0'),('agents','observation','obs_3')],
        out_keys = [('agents','action')],
        num_seeker_with_policy=num_of_seeker_with_policy
    )

    return model.to(device)

class Make_model(SafeModule):
    def __init__(
        self,
        model,
        in_keys,
        out_keys,
        num_seeker_with_policy,
    ):
        super().__init__(module = model, in_keys =in_keys,out_keys = out_keys)
        self.num_seeker_with_policy = num_seeker_with_policy
        
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
                # we will get rid of tensordict_out as a regular arg, because it
                # blocks us when using vmap
                # with stateful but functional modules: the functional module checks if
                # it still contains parameters. If so it considers that only a "params" kwarg
                # is indicative of what the params are, when we could potentially make a
                # special rule for TensorDictModule that states that the second arg is
                # likely to be the module params.
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
            
            final_tensors = (torch.zeros((int(self.module.num_of_seekers),2)).cuda(),)
            for n in range(self.module.num_of_seekers):
                if n >= self.num_seeker_with_policy:
                    final_tensors[0][n] = tensors[0][n][:2]
                else:
                    final_tensors[0][n] = tensors[0][n][2:]

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


class MakeBinaryMask(Transform):

    def __init__(
        self,
        in_keys,
        out_keys,
        num_of_seekers
    ):

        super().__init__(in_keys=in_keys, out_keys=out_keys)
        self.num_of_seekers = num_of_seekers



    def _apply_transform(self, obs):
        image = obs[0].squeeze().permute(2,0,1)
        position1 = obs[1]
        binary_mask = torch.zeros((1,156,156)).to('cuda')
        x = int((-position1[1].item()+28.99)/57.98*155)
        y = int((position1[0].item()+29.09)/58.18*155)
        flipped = False
        for j in range(x-3,x+4):
            for m in range(y-3,y+4):
                if j < 0 or j >= 156 or m < 0 or m >= 156:
                    continue
                if image[0,j,m] >= 0.85 and image[1,j,m] <= 0.22 and image[2,j,m] <= 0.22:
                    binary_mask[:,j,m] = 1
                    flipped = True
        
        if not flipped:
            binary_mask[:,x,y] = 1

        image  = torch.cat((image ,binary_mask),dim = 0)
        obs = image.unsqueeze(dim = 0)
        return obs 
        

    def _call(self, tensordict: TensorDictBase) -> TensorDictBase:
        obs_list = []
        for i in range(self.num_of_seekers):  #fix me generalize to number of seekers
            observation = []
            for in_key in self.in_keys:
                if in_key in tensordict.keys(include_nested=True):
                    observation.append(tensordict.get(in_key)[i])
                    # observation.append(tensordict.get(in_key)[i])
                
                elif not self.missing_tolerance:
                    raise KeyError(
                        f"{self}: '{in_key}' not found in tensordict {tensordict}"
                    )
            obs1 = self._apply_transform(observation)
            obs_list.append(obs1)
            
        obs1 = torch.cat(obs_list,dim = 0)
        tensordict[self.out_keys[0]] = obs1



        return tensordict
    
    forward = _call

    def transform_observation_spec(self, observation_spec: TensorSpec) -> TensorSpec:
        return observation_spec
