# https://pytorch.org/tutorials/intermediate/custom_function_double_backward_tutorial.html
# Show case that single pass 2nd-derivative is equivalent to double pass 2nd-derivative
import torch
import torchviz


def cube_forward(x):
    return x**3


def cube_backward(grad_out, x):
    return grad_out * 3 * x**2


def cube_backward_backward(grad_out, sav_grad_out, x):
    return grad_out * sav_grad_out * 6 * x


def cube_backward_backward_grad_out(grad_out, x):
    return grad_out * 3 * x**2


class CubeDoubleBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return cube_forward(x)

    @staticmethod
    def backward(ctx, grad_out):
        (x,) = ctx.saved_tensors
        grad = CubeBackward.apply(grad_out, x)
        return grad


class CubeBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, grad_out, x):
        ctx.save_for_backward(x, grad_out)
        return cube_backward(grad_out, x)

    @staticmethod
    def backward(ctx, grad_out):
        x, sav_grad_out = ctx.saved_tensors
        dx = cube_backward_backward(grad_out, sav_grad_out, x)
        dgrad_out = cube_backward_backward_grad_out(grad_out, x)
        return dgrad_out, dx


class CubeSingleBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        out = x**3
        der = 3 * x**2
        ctx.save_for_backward(x, der)
        return out, der

    @staticmethod
    def backward(ctx, grad_out_1st, grad_out_2nd):
        (x, der) = ctx.saved_tensors
        grad = grad_out_1st * der + grad_out_2nd * 6 * x
        return grad


def double_grad(x, viz=False):
    out = CubeDoubleBackward.apply(x)
    ones = torch.ones_like(out, requires_grad=False)
    (grad_x,) = torch.autograd.grad(out, x, grad_outputs=ones, create_graph=True)
    loss = grad_x + out
    loss.sum().backward(retain_graph=True)

    if viz:
        torchviz.make_dot(
            (grad_x, x, ones, out, loss),
            params={"grad_x": grad_x, "x": x, "ones": ones, "out": out, "loss": loss},
            show_saved=False,
        ).render("double_grad", format="png")
    return x.grad


def single_grad(x, viz=False):
    x = torch.tensor(2.0, requires_grad=True, dtype=torch.double)
    out, grad_x = CubeSingleBackward.apply(x)
    loss = grad_x + out
    loss.sum().backward(retain_graph=True)

    if viz:
        torchviz.make_dot(
            (x, out, grad_x, loss),
            params={"x": x, "out": out, "grad_x": grad_x, "loss": loss},
            show_saved=False,
        ).render("single_grad", format="png")
    return x.grad


def _block(x):
    x0 = x.clone()
    x1 = x.clone()
    x0_grad = double_grad(x0)
    x1_grad = single_grad(x1)
    assert torch.allclose(x0_grad, x1_grad)

def test_single_vs_double_grad():
    _block(torch.tensor(2.0, requires_grad=True, dtype=torch.double))
    _block(torch.rand(10, 3, requires_grad=True, dtype=torch.double))
    _block(torch.rand(10, 3, 4, requires_grad=True, dtype=torch.double))
