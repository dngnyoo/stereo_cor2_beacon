"""CME Products Stage: independent candidate catalog and diagnostic figures."""

from __future__ import annotations

from pathlib import Path
import hashlib

import pandas as pd

from cme_pipeline.core.stage import PipelineStage, SegmentationOutput, CMEProductsOutput
from cme_pipeline.stages.segmentation.cme_catalog import save_cme_catalog, build_cme_catalog
from cme_pipeline.utils.io import compute_dict_hash


class CMEProductsStage(PipelineStage):
    name = "products"
    version = "1.1.0"

    def compute_fingerprint(self, inputs, config, context):
        mask_hash = hashlib.sha256()
        mask_hash.update(inputs.final_pixel_segmentation_union.tobytes())
        for fid in sorted(inputs.family_pixel_masks):
            mask_hash.update(fid.encode("utf-8"))
            mask_hash.update(inputs.family_pixel_masks[fid].tobytes())
        return compute_dict_hash({
            "stage": self.name,
            "version": self.version,
            "upstream_pixels": int(inputs.total_segmented_pixels),
            "mask_sha256": mask_hash.hexdigest(),
            "products": config.products.__dict__,
        })

    def outputs_exist(self, output: CMEProductsOutput) -> bool:
        """Detect missing side-effect files even when the pickle cache is valid."""
        if not isinstance(output, CMEProductsOutput):
            return False
        return all(
            Path(path).is_file() for path in output.artifact_files
        )

    def run(self, inputs: SegmentationOutput, config, context):
        context.info("Running CMEProductsStage (independent catalog + diagnostics)...")
        if not isinstance(inputs, SegmentationOutput):
            raise TypeError("products expects SegmentationOutput")
        artifacts: list[str] = []

        if config.products.catalog:
            event_file, frame_file = save_cme_catalog(inputs, context.tables_dir)
            events, frames = build_cme_catalog(inputs)
            artifacts.extend([str(event_file), str(frame_file)])
        else:
            events = pd.DataFrame()
            frames = pd.DataFrame()
        if config.products.visualization:
            if frames.empty:
                _, frames = build_cme_catalog(inputs)
            from cme_pipeline.stages.segmentation.visualization import create_visualizations
            figures = create_visualizations(
                inputs, frames, context.visualizations_dir,
                make_animation=config.products.animation,
            )
            artifacts.extend(str(path) for path in figures)

        context.info(
            f"CME products: {len(events)} catalog events; "
            f"{len(artifacts)} output files generated."
        )
        return CMEProductsOutput(
            segmentation_output=inputs,
            event_catalog_df=events,
            frame_measurements_df=frames,
            artifact_files=artifacts,
        )
