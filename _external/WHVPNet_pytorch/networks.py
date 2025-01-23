import torch
from .vp_functions import *
from typing import Any, Callable, Optional
from enum import Enum

def _fcnn(n_in: int, n_channels: int, n_hiddens: list[int], n_out: int,
          nonlinear: Callable[[], torch.nn.Module] = torch.nn.ReLU,
          device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) \
          -> list[torch.nn.Module]:
    layers = [torch.nn.Flatten()] # flatten channels
    n0 = n_in * n_channels
    for n in n_hiddens:
        layers.append(torch.nn.Linear(n0, n, device=device, dtype=dtype))
        layers.append(nonlinear().to(device=device, dtype=dtype))
        n0 = n
    layers.append(torch.nn.Linear(n0, n_out, device=device, dtype=dtype))
    layers.append(torch.nn.Softmax(-1))
    return layers

def FCNN(n_in: int, n_channels: int, n_hiddens: list[int], n_out: int,
         nonlinear: Callable[[], torch.nn.Module] = torch.nn.ReLU,
         device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) \
         -> torch.nn.Sequential:
    '''
    Builder function for simple fully connected neural network (MLP).
    
    Input:
        n_in: int                       Input dimension.
        n_channels: int                 Number of channels.
        n_hiddens: list[int]            Neurons of hidden layers.
        n_out: int                      Output dimension.
        nonlinear: Callable[[], torch.nn.Module]
                                        Builder of nonlinear activation.
                                        Default: torch.nn.ReLU
        device: Optional[torch.device]  Pytorch device. Default: None
        dtype: Optional[torch.dtype]    Tensor data type. Default: None
    Output:
        fcnn: torch.nn.Sequential       Fully connected neural network of
                                        linear layers and nonlinear activation.
    '''
    layers = _fcnn(n_in, n_channels, n_hiddens, n_out, nonlinear, device, dtype)
    return torch.nn.Sequential(*layers)

def CNN1D(n_in: int, n_channels: list[int],
        conv_kernel_sizes: list[int], pool_kernel_sizes: list[int],
        n_hiddens: list[int], n_out: int,
        conv_nonlinear: Callable[[], torch.nn.Module] = torch.nn.ReLU,
        nonlinear: Callable[[], torch.nn.Module] = torch.nn.ReLU,
        device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) \
        -> torch.nn.Sequential:
    assert len(n_channels) >= 2
    assert len(n_channels) == len(conv_kernel_sizes)+1
    assert len(conv_kernel_sizes) == len(pool_kernel_sizes)
    '''
    Builder function for simple convolutional neural network in 1D.
    
    Input:
        n_in: int                       Input dimension.
        n_channels: list[int]           List of number of channels: for input
                                        and for convolutional layers.
        conv_kernel_sizes: list[int]    List of convolutional kernel sizes.
        pool_kernel_sizes: list[int]    List of pooling kernel sizes.
        n_hiddens: list[int]            Neurons of fully connected layers.
        n_out: int                      Output dimension.
        conv_nonlinear: Callable[[], torch.nn.Module]
                                        Builder of nonlinear activation for
                                        the convolutional part.
                                        Default: torch.nn.ReLU
        nonlinear: Callable[[], torch.nn.Module]
                                        Builder of nonlinear activation for
                                        the fully connected part.
                                        Default: torch.nn.ReLU
        device: Optional[torch.device]  Pytorch device. Default: None
        dtype: Optional[torch.dtype]    Tensor data type. Default: None
    Output:
        cnn: torch.nn.Sequential        Convolutional neural network of 
                                        convolutional layers, nonlinear 
                                        activation, and max pooling,
                                        followed by a fully connected part.
    '''
    conv_layers = []
    n0 = n_in
    for i in range(len(n_channels)-1):
        conv_layers.append(torch.nn.Conv1d(n_channels[i], n_channels[i+1],
            conv_kernel_sizes[i], device=device, dtype=dtype))
        conv_layers.append(conv_nonlinear().to(device=device, dtype=dtype))
        n0 = n0 - conv_kernel_sizes[i] + 1
        conv_layers.append(torch.nn.MaxPool1d(pool_kernel_sizes[i]))
        n0 = n0 // pool_kernel_sizes[i]
    fcnn_layers = _fcnn(n0, n_channels[-1], n_hiddens, n_out, nonlinear, device, dtype)
    return torch.nn.Sequential(*conv_layers, *fcnn_layers)

class VPTypes(Enum):
    '''Output selector for VPNet.'''
    FEATURES = 0
    APPROXIMATION = 1
    RESIDUAL = 2

class VPNet(torch.nn.Module):
    '''Simple Variable Projection Network (VPNet).'''
    def __init__(self, n_in: int, n_channels: int, n_vp: int, vp_type: VPTypes,
                 params_init: list[int | float] | torch.Tensor,
                 fun_system: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
                 n_hiddens: list[int], n_out: int,
                 nonlinear: Callable[[], torch.nn.Module] = torch.nn.ReLU,
                 device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> None:
        '''
        Constructs a Variable Projection Network (VPNet) instance as a
        combination of a VP layer and a fully connected network. Depending on
        vp_type, forwards the coefficients, approximation, or the residuals to
        the subsequent fully connected part.
        
        Input:
            n_in: int                       Input dimension.
            n_channels: int                 Number of channels.
            n_vp: int                       Number of VP coefficients.
            vp_type: VPTypes                VP output type.
            params_init: list[int | float] | torch.Tensor
                                            Initial values of VP parameters.
            fun_system: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]
                                            Function system and derivative builder.
                                            See FunSystem.__call__
            n_hiddens: list[int]            Neurons of fully connected layers.
            n_out: int                      Output dimension.
            nonlinear: Callable[[], torch.nn.Module]
                                            Builder of nonlinear activation for
                                            the fully connected part.
                                            Default: torch.nn.ReLU
            device: Optional[torch.device]  Pytorch device. Default: None
            dtype: Optional[torch.dtype]    Tensor data type. Default: None
        '''
        super().__init__()
        params_init = torch.tensor(params_init, device=device, dtype=dtype)
        self.vp_layer = VPLayer(params_init, fun_system)
        self.vp_type = vp_type
        n_in_fcnn = n_vp if vp_type == VPTypes.FEATURES else n_in
        self.fcnn = FCNN(n_in_fcnn, n_channels, n_hiddens, n_out, nonlinear, device, dtype)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        '''
        Forward operator.
        
        Input:
            x: torch.Tensor     Input tensor.
        Output:
            y: torch.Tensor     Network output.
            reg: torch.Tensor   Normalized L2 error for regularization.
        '''
        outs = self.vp_layer(x) # coeffs, x_hat, res, r2
        y = self.fcnn(outs[self.vp_type.value])
        reg = (outs[3] / (x ** 2).sum(dim=-1)).mean(dim=0)
        return y, reg

class VPLoss(torch.nn.Module):
    '''Loss wrapper with Variable Projection error penalty.'''
    def __init__(self, criterion: Callable[[torch.Tensor], torch.Tensor], vp_penalty: float) -> None:
        '''
        Constructs the lost wrapper.
        
        Input:
            criterion: Callable[[torch.Tensor], torch.Tensor]
                                    Loss criterion for the network output.
            vp_penalty: float       VP regularization penalty.
        '''
        super().__init__()
        self.criterion = criterion
        self.vp_penalty = vp_penalty

    def forward(self, outputs: tuple[torch.Tensor, torch.Tensor], target: torch.Tensor) -> torch.Tensor:
        '''Forward operator. Combines the network loss with VP regularization penalty.'''
        y, reg = outputs
        return self.criterion(y, target) + self.vp_penalty * reg

    def extra_repr(self) -> str:
        return f'(vp_penalty): {self.vp_penalty}'
