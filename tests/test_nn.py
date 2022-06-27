import os
import random
import sys

import numpy as np
import timm
import torch

import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.nn.encoder.efficientnet import (
    create_variable_length_efficientnet)


DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


@pytest.mark.parametrize('model_name', [
    "tf_efficientnetv2_b0",
    "tf_efficientnetv2_b1",
    "tf_efficientnetv2_b2",
    "tf_efficientnetv2_b3",
    "tf_efficientnetv2_s",
    "tf_efficientnetv2_m",
    "tf_efficientnetv2_l",
])
@pytest.mark.parametrize('features_only', [False, True])
@pytest.mark.parametrize('list_input', [False, True])
def test_create_variable_length_efficientnet(
    model_name,
    features_only,
    list_input,
):
    torch.manual_seed(0)
    input = torch.randn(5, 3, 64, 64, device=DEVICE)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    net1 = create_variable_length_efficientnet(
        in_chans=3, model_name=model_name, features_only=features_only)
    net1.to(DEVICE)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    net2 = timm.create_model(model_name, features_only=features_only)
    net2.to(DEVICE)

    # In the evaluation mode, the outputs of original EfficientNet and
    # variable-length EfficientNet "exactly" match.
    net1.eval()
    net2.eval()
    if list_input:
        output1 = net1(
            [input[i:i+1] for i in range(input.size(0))])
        if features_only:
            output1 = torch.concat(output1, dim=0)
    else:
        output1 = net1(input)
    output2 = net2(input)[-1] if features_only else net2(input)
    if not list_input:
        # Conv2d sometimes outputs slightly different results when conv2d is
        # inputted on a per-sample basis and on a per-minibatch basis.
        assert torch.all(output1 == output2)

    # In the training mode, the outputs of original EfficientNet and
    # variable-length EfficientNet "roughly" match. The subtle differences are
    # due to differences in the definition of ordinary BatchNorm2d and
    # variable-length BatchNorm2d.
    net1.train()
    net2.train()
    if list_input:
        output1 = net1(
            [input[i:i+1] for i in range(input.size(0))])
        if features_only:
            output1 = torch.concat(output1, dim=0)
    else:
        output1 = net1(input)
    output2 = net2(input)[-1] if features_only else net2(input)
    assert torch.allclose(output1, output2, rtol=5e-2, atol=1e-3)
