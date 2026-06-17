import sys
import os
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
    """
    Patch Ultralytics parser to ensure that the Detect head is reconstructed
    correctly with a standard channel dimension of (64, 64, 64), avoiding
    the positional argument channel inflation bug.
    """
    orig_parse_model = tasks.parse_model

    def custom_parse_model(d, ch, *args, **kwargs):
        from ultralytics.nn.modules.head import Detect
        
        # Build the model list using the original parser first
        model, save = orig_parse_model(d, ch, *args, **kwargs)
        
        # Inspect final layer (which is the Detect head)
        m = model[-1]
        if type(m).__name__ in ["Detect", "Segment", "Pose", "OBB"]:
            target_ch = [64, 64, 64]
            nc = int(m.nc[0]) if isinstance(m.nc, (list, tuple)) else int(m.nc)
            print(f"Rebuilding Detect head: nc={nc}, channels={target_ch}")
            
            # Reconstruct the head safely passing ch as a keyword argument
            new_m = Detect(nc, ch=target_ch)
            new_m.f, new_m.i, new_m.type = m.f, m.i, m.type
            new_m.nc = nc
            if hasattr(m, 's'):
                new_m.s = m.s
            model[-1] = new_m
            
        return model, save

    tasks.parse_model = custom_parse_model
    print("✓ Ultralytics parse_model patched.")

def main():
    parser = argparse.ArgumentParser(description="Train Bhaskar-NET")
    parser.add_argument("--model", type=str, default="cfg/bhaskar_net.yaml", help="Path to model YAML")
    parser.add_argument("--data", type=str, default="VisDrone.yaml", help="Path to dataset YAML")
    parser.add_argument("--epochs", type=int, default=40, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=800, help="Input image size")
    parser.add_argument("--batch", type=int, default=4, help="Batch size")
    parser.add_argument("--project", type=str, default="BhaskarNet_Runs", help="Project output folder")
    parser.add_argument("--name", type=str, default="BhaskarNet_Experiment", help="Run name")
    parser.add_argument("--amp", action="store_true", default=True, help="Enable automatic mixed precision")
    args = parser.parse_args()

    register_modules()
    patch_detect_head()

    # Load from YAML and train
    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
        amp=args.amp,
        save=True
    )

if __name__ == "__main__":
    main()
