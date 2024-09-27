import torch
from torchrl.data.utils import DEVICE_TYPING
from torchrl.envs.transforms.transforms import (
    Compose,
    FlattenObservation,
    ToTensorImage,
)


class CatUnitySensorsAlongChannelDimTransform(Compose):
    def __init__(
        self,
        in_keys: list[str] | None = None,
        out_keys: list[str] | None = None,
    ):
        self._device = None
        self._dtype = None
        in_keys = in_keys if in_keys is not None else ["observation"]
        out_keys = out_keys if out_keys is not None else ["observation"]

        transforms = []

        # ToTensor

        # Input data of the form:
        # (num_agents, num_sensors, width, height, num_stacks * channels).
        # Converts it to Tensor format.
        totensor = ToTensorImage(in_keys=in_keys)
        transforms.append(totensor)

        # Our data is currently in the form (num_agents, num_sensors,
        # num_stacks * channels, width, height). We want our data to
        # be stored in the form (num_agents,
        # num_sensors * num_stacks * channels, width, height).
        #
        # We flatten the observation so that multiple sensors per observation
        # are stored along the channel dimension rather than on the observation
        # dimension.
        flatten = FlattenObservation(in_keys=in_keys, first_dim=-4, last_dim=-3)
        transforms.append(flatten)

        super().__init__(*transforms)

        if self._device is not None:
            self.to(self._device)
        if self._dtype is not None:
            self.to(self._dtype)

    def to(self, dest: DEVICE_TYPING | torch.dtype):
        if isinstance(dest, torch.dtype):
            self._dtype = dest
        else:
            self._device = dest
        return super().to(dest)

    @property
    def device(self):
        return self._device

    @property
    def dtype(self):
        return self._dtype
