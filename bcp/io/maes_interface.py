from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from .config import settings


@dataclass
class MaesInterface:
    """Data exchange between BCP and the MAES project.

    Paths and behavior are configured via environment variables (see Settings).
    """

    maes_root: Path = settings.maes_project_path
    incoming_dir: Path = settings.control_panel_data_import
    outgoing_dir: Path = settings.control_panel_data_export
    version: str = settings.shared_format_version

    def import_maes_data(self, filename: str = "xenopus_preprocessed.npz") -> Optional[Dict[str, Any]]:
        """Import data from MAES .npz and convert to standardized JSON structure.

        Returns the standardized dict or None if file not found.
        """
        source_path = self.maes_root / filename
        if not source_path.exists():
            return None

        data = np.load(source_path)

        standardized: Dict[str, Any] = {
            "metadata": {
                "source": "MAES_GPT_INVENTS",
                "timestamp": datetime.now().isoformat(),
                "experiment_type": "xenopus_bioelectric",
                "version": self.version,
            },
            "parameters": {
                "dt": float(data.get("dt", 0.0)),
                "lam2": float(data.get("lam2", 0.0)),
                "sync": float(data.get("sync", 0.0)),
                "Fproxy": float(data.get("Fproxy", 0.0)),
            },
            "data": {},
        }

        # Optional keys with fallbacks
        if "traces" in data:
            traces = data["traces"]
            standardized["data"]["traces"] = traces.tolist()
            standardized["data"]["n_cells"] = int(traces.shape[0]) if traces.ndim == 2 else 0
            standardized["data"]["n_timesteps"] = int(traces.shape[1]) if traces.ndim == 2 else 0

        if "C" in data:
            standardized["data"]["coupling_matrix"] = data["C"].tolist()

        # Persist to incoming dir for traceability
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        import_path = self.incoming_dir / f"import_{ts}.json"
        with import_path.open("w") as f:
            json.dump(standardized, f, indent=2)

        return standardized

    def export_to_maes(self, payload: Dict[str, Any], experiment_name: str) -> Path:
        """Export control results to MAES import area and BCP outgoing.

        Returns the path written under MAES (mirrored copy) if possible, else just outgoing.
        """
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"control_export_{experiment_name}_{ts}.json"

        export_data = {
            "metadata": {
                "source": "Bioelectric_Control_Panel",
                "timestamp": datetime.now().isoformat(),
                "experiment_name": experiment_name,
                "version": self.version,
            },
            "control_parameters": payload.get("control_parameters", {}),
            "simulation_data": payload.get("simulation_data", {}),
        }

        # Write to BCP outgoing
        outgoing_path = self.outgoing_dir / filename
        with outgoing_path.open("w") as f:
            json.dump(export_data, f, indent=2)

        # Attempt to mirror into MAES data_export if present
        maes_export_dir = self.maes_root / "data_export"
        try:
            maes_export_dir.mkdir(parents=True, exist_ok=True)
            maes_path = maes_export_dir / filename
            maes_path.write_text(outgoing_path.read_text())
            return maes_path
        except Exception:
            return outgoing_path

