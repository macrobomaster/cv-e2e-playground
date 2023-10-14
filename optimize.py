import shelve
from typing import cast

from tinygrad.tensor import Tensor
from tinygrad.ops import LoadOps, Device, Compiled
from tinygrad.codegen.linearizer import Linearizer
from tinygrad.codegen.search import time_linearizer, get_linearizer_actions
from tinygrad.helpers import ansilen, getenv, flatten
from tinygrad.lazy import vars_from_ast
from tinygrad.shape.symbolic import sym_infer

from model import Head


def sched_for_inference(foundation, head):
    head(foundation(Tensor.empty(1, 480, 640, 3)), Tensor.empty(1, 1))[
        0
    ].lazydata.schedule(seen := set())
    sched = head(foundation(Tensor.empty(1, 480, 640, 3)), Tensor.empty(1, 1))[
        0
    ].lazydata.schedule(seen)
    return [x for x in sched if x.ast.op not in LoadOps]


def apply_optimizations_inference(foundation, head):
    device = cast(Compiled, Device[Device.DEFAULT])
    db = shelve.open("./cache/opt_db")

    for _, si in enumerate(sched_for_inference(foundation, head)):
        lin = Linearizer(si.ast, device.linearizer_opts)
        if (key := str(lin.ast)) in db:
            for ao in db[key]:
                lin.apply_opt(ao)
            device.method_cache[lin.ast] = device.to_program(lin)


def sched_for_training(head, bs):
    head(Tensor.empty(bs, 512, 15, 20), Tensor.empty(bs, 1))[0].lazydata.schedule(
        seen := set()
    )
    sched = head(Tensor.empty(bs, 512, 15, 20), Tensor.empty(bs, 1))[
        0
    ].lazydata.schedule(seen)
    return [x for x in sched if x.ast.op not in LoadOps]


def apply_optimizations_training(head, bs):
    device = cast(Compiled, Device[Device.DEFAULT])
    db = shelve.open("./cache/opt_db")

    for _, si in enumerate(sched_for_training(head, bs)):
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

    foundation, head = get_foundation(), Head()
    if getenv("TRAIN"):
        Tensor.training = True
        Tensor.no_grad = False
        sched = sched_for_training(head, getenv("BS", 32))
    else:
        Tensor.training = False
        Tensor.no_grad = True
        sched = sched_for_inference(foundation, head)

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
            best_tm = float("inf")
            beam = [lin]
            while True:
                acted_lins = flatten(
                    [get_linearizer_actions(lin).items() for lin in beam]
                )
                timed_lins = [
                    (v, time_linearizer(v, rawbufs)) for k, v in acted_lins if k != 0
                ]
                opts = sorted(timed_lins, key=lambda x: x[1])
                if len(opts) == 0 or best_tm <= opts[0][1]:
                    break
                best_tm = opts[0][1]
                beam = [x[0] for x in opts[:4]]
                print(
                    f"{opts[0][1]*1e3:10.2f} ms from {len(opts):3d} actions",
                    beam[0].colored_shape(),
                )
            lin = beam[0]
            beam_db[str(lin.ast)] = lin.applied_opts
        lins.append(lin)

        choices = []
        for lin in lins:
            tm = time_linearizer(
                lin, rawbufs, allow_test_size=False, cnt=10, should_copy=False
            )
            gflops = (
                sym_infer(lin.info.flops, {k: k.min for k in vars_from_ast(lin.ast)})
                * 1e-9
                / tm
            )
            choices.append((tm, gflops, lin))

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
