import argparse
import importlib
import os
import sys
from collections import defaultdict


PRJ_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PRJ_PATH not in sys.path:
    sys.path.append(PRJ_PATH)


def parse_args():
    parser = argparse.ArgumentParser(description="Count TSMTrack model parameters.")
    parser.add_argument("--script", type=str, default="tsmtrack", choices=["tsmtrack"],
                        help="model/config family name")
    parser.add_argument("--config", type=str, default="tsmtrack",
                        help="config name under experiments/<script>/, with or without .yaml")
    parser.add_argument("--config-path", type=str, default=None,
                        help="explicit yaml path; overrides --script/--config")
    parser.add_argument("--training", action="store_true",
                        help="build with training=True; this may load cfg.MODEL.PRETRAIN_FILE")
    parser.add_argument("--include-buffers", action="store_true",
                        help="also print non-parameter buffer count")
    return parser.parse_args()


def resolve_config_path(args):
    if args.config_path is not None:
        return os.path.abspath(args.config_path)

    config_name = args.config
    if config_name.endswith(".yaml"):
        config_name = config_name[:-5]
    return os.path.join(PRJ_PATH, "experiments", args.script, config_name + ".yaml")


def count_tensor_values(tensors):
    return sum(t.numel() for t in tensors)


def format_count(value):
    return f"{value:,} ({value / 1e6:.3f} M)"


def top_level_name(param_name):
    if "." not in param_name:
        return "<root>"
    return param_name.split(".", 1)[0]


def build_model(script, cfg, training):
    model_module = importlib.import_module("lib.models")
    if script == "tsmtrack":
        return model_module.build_tsmtrack(cfg, training=training)
    raise NotImplementedError(f"Unsupported script: {script}")


def print_module_table(model):
    table = defaultdict(lambda: {"total": 0, "trainable": 0})
    for name, param in model.named_parameters():
        key = top_level_name(name)
        numel = param.numel()
        table[key]["total"] += numel
        if param.requires_grad:
            table[key]["trainable"] += numel

    print("\nParameter breakdown by top-level module:")
    print(f"{'module':<28} {'total':>18} {'trainable':>18} {'frozen':>18}")
    print("-" * 85)
    for module_name, values in sorted(table.items(), key=lambda item: item[1]["total"], reverse=True):
        total = values["total"]
        trainable = values["trainable"]
        frozen = total - trainable
        print(f"{module_name:<28} {total:>18,} {trainable:>18,} {frozen:>18,}")


def main():
    args = parse_args()
    config_path = resolve_config_path(args)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    config_module = importlib.import_module(f"lib.config.{args.script}.config")
    cfg = config_module.cfg
    config_module.update_config_from_file(config_path)

    model = build_model(args.script, cfg, training=args.training)
    model.eval()

    total_params = count_tensor_values(model.parameters())
    trainable_params = count_tensor_values(p for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    print(f"Config: {config_path}")
    print(f"Build mode: training={args.training}")
    print(f"Total parameters:     {format_count(total_params)}")
    print(f"Trainable parameters: {format_count(trainable_params)}")
    print(f"Frozen parameters:    {format_count(frozen_params)}")

    if args.include_buffers:
        buffers = count_tensor_values(model.buffers())
        print(f"Buffers:              {format_count(buffers)}")

    print_module_table(model)


if __name__ == "__main__":
    main()
