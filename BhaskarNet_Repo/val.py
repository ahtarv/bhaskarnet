import argparse
from ultralytics import YOLO
import ultralytics.nn.tasks as tasks
import ultralytics.nn.modules as mods
import ultralytics.nn.modules.block as block

# Import custom modules
from models.TuaBottleneck import TuaBottleneck
from models.saclseq import Scalseq
from models.zoomcat import Zoomcat

def register_modules():
    """Register custom modules in the Ultralytics namespace."""
    for module in (block, mods, tasks):
        setattr(module, 'TuaBottleneck', TuaBottleneck)
        setattr(module, 'Scalseq', Scalseq)
        setattr(module, 'Zoomcat', Zoomcat)
    print("✓ Custom modules registered in Ultralytics.")

def patch_detect_head():
    """Patch Ultralytics parser to ensure robust construction of head."""
    orig_parse_model = tasks.parse_model

    def custom_parse_model(d, ch, *args, **kwargs):
        from ultralytics.nn.modules.head import Detect
        model, save = orig_parse_model(d, ch, *args, **kwargs)
        m = model[-1]
        if type(m).__name__ in ["Detect", "Segment", "Pose", "OBB"]:
            target_ch = [64, 64, 64]
            nc = int(m.nc[0]) if isinstance(m.nc, (list, tuple)) else int(m.nc)
            new_m = Detect(nc, ch=target_ch)
            new_m.f, new_m.i, new_m.type = m.f, m.i, m.type
            new_m.nc = nc
            if hasattr(m, 's'):
                new_m.s = m.s
            model[-1] = new_m
        return model, save

    tasks.parse_model = custom_parse_model

def main():
    parser = argparse.ArgumentParser(description="Evaluate Bhaskar-NET")
    parser.add_argument("--weights", type=str, required=True, help="Path to trained model weights (.pt)")
    parser.add_argument("--data", type=str, default="VisDrone.yaml", help="Path to dataset YAML")
    parser.add_argument("--imgsz", type=int, default=800, help="Input image size")
    parser.add_argument("--batch", type=int, default=4, help="Batch size")
    parser.add_argument("--split", type=str, default="val", help="Dataset split to evaluate on (val/test)")
    args = parser.parse_args()

    register_modules()
    patch_detect_head()

    # Load model weights and evaluate
    model = YOLO(args.weights)
    metrics = model.val(
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        split=args.split
    )
    print("Validation Complete.")
    print(f"mAP50: {metrics.results_dict['metrics/mAP50(B)']:.4f}")
    print(f"mAP50-95: {metrics.results_dict['metrics/mAP50-95(B)']:.4f}")

if __name__ == "__main__":
    main()
