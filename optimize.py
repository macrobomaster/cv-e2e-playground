import shelve
from typing import cast
from tinygrad.nn.state import get_parameters

from tinygrad.tensor import Tensor
from tinygrad.ops import LoadOps, Device, Compiled
from tinygrad.codegen.linearizer import Linearizer
from tinygrad.features.search import (
    time_linearizer,
    beam_search,
)
from tinygrad.helpers import ansilen, getenv
from tinygrad.lazy import vars_from_ast
from tinygrad.shape.symbolic import sym_infer

from model import Model


def sched_for_inference(foundation, model):
    x = foundation(Tensor.empty(1, 480, 640, 3))
    model(x, Tensor.empty(1, 1))[0].lazydata.schedule(seen := set())
    x = foundation(Tensor.empty(1, 480, 640, 3))
    sched = model(x, Tensor.empty(1, 1))[0].lazydata.schedule(seen)
    return [x for x in sched if x.ast.op not in LoadOps]


def apply_optimizations_inference(foundation, model):
    device = cast(Compiled, Device[Device.DEFAULT])
    db = shelve.open("./cache/opt_db")

    for _, si in enumerate(sched_for_inference(foundation, model)):
        lin = Linearizer(si.ast, device.linearizer_opts)
        if (key := str(lin.ast)) in db:
            for ao in db[key]:
                lin.apply_opt(ao)
            device.method_cache[lin.ast] = device.to_program(lin)


def sched_for_training(model, bs):
    from train import loss_fn

    # forward pass
    loss_fn(
        model(
            Tensor.empty(bs, 256, 15, 20, requires_grad=False),
            Tensor.empty(bs, 1, requires_grad=False),
        ),
        Tensor.empty(bs, 4, requires_grad=False),
    ).lazydata.schedule(seen := set())
    sched = loss_fn(
        model(
            Tensor.empty(bs, 256, 15, 20, requires_grad=False),
            Tensor.empty(bs, 1, requires_grad=False),
        ),
        Tensor.empty(bs, 4, requires_grad=False),
    ).lazydata.schedule(seen)

    # backward pass
    for param in get_parameters(model):
        if param.requires_grad is None:
            param.requires_grad = True

    loss_fn(
        model(
            Tensor.empty(bs, 256, 15, 20, requires_grad=False),
            Tensor.empty(bs, 1, requires_grad=False),
        ),
        Tensor.empty(bs, 4, requires_grad=False),
    ).backward()
    for param in get_parameters(model):
        if param.grad is not None:
            sched += param.grad.lazydata.schedule(seen)

    return [x for x in sched if x.ast.op not in LoadOps]


def apply_optimizations_training(model, bs):
    device = cast(Compiled, Device[Device.DEFAULT])
    db = shelve.open("./cache/opt_db")

    for _, si in enumerate(sched_for_training(model, bs)):
        lin = Linearizer(si.ast, device.linearizer_opts)
        if (key := str(lin.ast)) in db:
            for ao in db[key]:
                lin.apply_opt(ao)
            device.method_cache[lin.ast] = device.to_program(lin)


if __name__ == "__main__":
    from main import get_foundation

    beam_db = shelve.open("./cache/beam_cache")
    db = shelve.open("./cache/opt_db")

    device = cast(Compiled, Device[Device.DEFAULT])
    print(f"optimizing for {Device.DEFAULT}")

    if getenv("TRAIN"):
        Tensor.training = True
        Tensor.no_grad = False
        model = Model()
        sched = sched_for_training(model, getenv("BS", 32))
    else:
        Tensor.training = False
        Tensor.no_grad = True
        foundation, model = get_foundation(), Model()
        sched = sched_for_inference(foundation, model)

    print(f"found {len(sched)} kernels")

    if getenv("KERNEL", -1) >= 0:
        sched = sched[getenv("KERNEL", -1) : getenv("KERNEL", -1) + 1]
        print(f"optimizing kernel {getenv('KERNEL', -1)}")

    total_tm = 0
    running_gflops = 0
    for i, si in enumerate(sched):
        rawbufs = [device.buffer(si.out.st.size(), si.out.dtype)] + [
            device.buffer(x.st.size(), x.dtype) for x in si.inputs
        ]
        lins = []

        # hand coded
        lin = Linearizer(si.ast, device.linearizer_opts)
        lin.hand_coded_optimizations()
        lins.append(lin)

        # beam search
        lin = Linearizer(si.ast, device.linearizer_opts)
        if str(lin.ast) in beam_db:
            for ao in beam_db[str(lin.ast)]:
                lin.apply_opt(ao)
        else:
            lin = beam_search(lin, rawbufs, 4, True)
            beam_db[str(lin.ast)] = lin.applied_opts
        lins.append(lin)

        choices = []
        for lin in lins:
            tm = time_linearizer(lin, rawbufs, allow_test_size=False, cnt=10)
            gflops = (
                sym_infer(lin.info.flops, {k: k.min for k in vars_from_ast(lin.ast)})
                * 1e-9
                / tm
            )
            choices.append((tm, gflops, lin.linearize()))

            print(
                f"                 kernel {i:2d} {lin.display_name+' '*(37-ansilen(lin.display_name))} {str(lin.global_size):18s} {str(lin.local_size):12s} takes {tm*1000:7.2f} ms, {gflops:6.0f} GFLOPS"
            )
        tm, gflops, lin = sorted(choices, key=lambda x: x[0])[0]
        db[str(lin.ast)] = lin.applied_opts
        print(
            f"*** {total_tm*1000:7.2f} ms : kernel {i:2d} {lin.display_name+' '*(37-ansilen(lin.display_name))} {str(lin.global_size):18s} {str(lin.local_size):12s} takes {tm*1000:7.2f} ms, {gflops:6.0f} GFLOPS"
        )
        total_tm += tm
        running_gflops += gflops * tm
    print(
        f"******* total {total_tm*1000:.2f} ms, {running_gflops/total_tm:6.0f} GFLOPS"
    )
