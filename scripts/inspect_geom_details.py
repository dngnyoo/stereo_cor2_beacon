from cme_pipeline.core.config import PipelineConfig
from cme_pipeline.core.context import PipelineContext
from cme_pipeline.stages.detection.frozen_detection import load_frozen_detection_inputs

cfg = PipelineConfig()
cfg.date_str = "20101212"
ctx = PipelineContext(cfg)
det_out = load_frozen_detection_inputs(cfg, ctx)
geom = det_out.geometry
print("GeometryContext attributes:", dir(geom))
for attr in ["rsun_px", "center_xy", "pa_raster", "radius_raster", "patch_pa", "patch_rad"]:
    val = getattr(geom, attr, None)
    if hasattr(val, "shape"):
        print(f"  {attr}: shape={val.shape}, min={val.min():.3f}, max={val.max():.3f}")
    else:
        print(f"  {attr}: {val}")
