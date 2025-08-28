"""
Minimal hooks the SACP UI can call:
- get_ib3d_line_sweep(...)
- get_grid(...)
- summarize_bioelectric(...)
"""
from ..sweep.parameter_sweep import sweep_a31_line, sweep_grid_a21_a31
from ..pipelines.bcp_integration import analyze_bioelectric_timeseries

def get_ib3d_line_sweep(**kw):
    return sweep_a31_line(**kw)

def get_grid(a21_grid, a31_grid):
    return sweep_grid_a21_a31(a21_grid, a31_grid)

def summarize_bioelectric(ts):
    return analyze_bioelectric_timeseries(ts)

