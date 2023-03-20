import sys

import os
import os.path as osp
import torch
import numpy as np
import onnx
import onnxruntime
import argparse

from motion.config import get_config
from motion.engine.builder import TRAINERS
from motion.utils.utils import load_ckpt


def to_numpy(tensor):
    return (
        tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    )


if __name__ == "__main__":
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=1, user_api="blas"):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--config",
            type=str,
            required=True,
            help="path to config yaml containing info about experiment",
        )
        parser.add_argument(
            "opts",
            default=None,
            nargs=argparse.REMAINDER,
            help="Modify config options from command line",
        )
        parser.add_argument(
            "--epoch",
            default=0,
            type=str,
            help="Choose test model with epoch",
        )
        parser.add_argument(
            "--postfix",
            default="ONNX",
            type=str,
            help="Choose test model with epoch",
        )
        args = parser.parse_args()
        cfg = get_config(args.config, args.opts)

    trainer = TRAINERS.get(cfg.TYPE)

    cfg = cfg.cfg
    onnx_path = os.path.join(cfg.output_dir, "onnx", args.epoch)
    model_name = cfg.output_dir.split("/")[-1]

    cfg.MODEL.TYPE += args.postfix
    model_cfg = cfg.MODEL.cfg

    model, device = trainer.build_model(cfg)
    # device = torch.device("cpu")
    # model = model.cpu()
    model = load_ckpt(model, cfg.output_dir, args.epoch)

    os.makedirs(onnx_path, exist_ok=True)
    model_path = osp.join(onnx_path, f"{model_name}" + args.postfix[:-4] + ".onnx")

    model.eval()

    onnx_cfg = cfg.ONNX
    inputs = []

    for inp_name, inp_shape in zip(onnx_cfg.input_names, onnx_cfg.input_shapes):
        inp = torch.randn([1] + inp_shape, device=device, dtype=torch.float32)
        inputs.append(inp)

    input_names = onnx_cfg.input_names
    output_names = onnx_cfg.output_names
    torch.onnx.export(
        model,
        tuple(inputs),
        model_path,
        verbose=False,
        opset_version=9,
        input_names=input_names,
        output_names=output_names,
    )
    print("**** ONNX model has been exported to {}".format(model_path))
    torch_out = model(*inputs)
    print("output dim = {}".format(torch_out.shape))

    onnx_model = onnx.load(model_path)
    onnx.shape_inference.infer_shapes(onnx_model)
    ort_session = onnxruntime.InferenceSession(model_path)
    ort_inputs = ort_session.get_inputs()
    ort_inputs = {
        ort_inp.name: to_numpy(inp) for ort_inp, inp in zip(ort_inputs, inputs)
    }

    ort_outs = ort_session.run(None, ort_inputs)
    print("ONNX output shape {}".format(ort_outs[0].shape))
    # compare ONNX Runtime and PyTorch results
    np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)
    print("Exported model has been tested with ONNXRuntime, and the result looks good!")
