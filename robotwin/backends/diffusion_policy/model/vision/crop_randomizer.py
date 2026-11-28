import torch.nn as nn
import torchvision.transforms.functional as ttf


class CropRandomizer(nn.Module):
    """Center-crop image observations for evaluation."""

    def __init__(
        self,
        input_shape,
        crop_height,
        crop_width,
        num_crops=1,
        pos_enc=False,
    ):
        super().__init__()

        assert len(input_shape) == 3
        assert crop_height < input_shape[1]
        assert crop_width < input_shape[2]

        self.input_shape = input_shape
        self.crop_height = crop_height
        self.crop_width = crop_width
        self.num_crops = num_crops
        self.pos_enc = pos_enc

    def output_shape_in(self, input_shape=None):

        out_c = self.input_shape[0] + 2 if self.pos_enc else self.input_shape[0]
        return [out_c, self.crop_height, self.crop_width]

    def output_shape_out(self, input_shape=None):

        return list(input_shape)

    def forward_in(self, inputs):
        out = ttf.center_crop(inputs, output_size=(self.crop_height, self.crop_width))
        if self.num_crops > 1:
            batch, channels, height, width = out.shape
            out = out.unsqueeze(1).expand(batch, self.num_crops, channels, height, width)
            out = out.reshape(-1, channels, height, width)
        return out

    def forward_out(self, inputs):
        if self.num_crops <= 1:
            return inputs
        batch = inputs.shape[0] // self.num_crops
        return inputs.reshape(batch, self.num_crops, *inputs.shape[1:]).mean(dim=1)

    def forward(self, inputs):
        return self.forward_in(inputs)

    def __repr__(self):
        header = "{}".format(str(self.__class__.__name__))
        msg = header + "(input_shape={}, crop_size=[{}, {}], num_crops={})".format(self.input_shape, self.crop_height,
                                                                                   self.crop_width, self.num_crops)
        return msg
