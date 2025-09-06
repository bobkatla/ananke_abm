import os
from click import command, option
from ananke_abm.utils.traj_fig.make_buffer_grid import make_buffer_grid
from ananke_abm.utils.traj_fig.fig_stacked_traj import fig_stacked_traj
from ananke_abm.utils.traj_fig.fig_specific_trajs import fig_specific_trajs
from ananke_abm.utils.traj_fig.fig_primary_lunch_time import fig_primary_lunch_time

@command()
@option("-traj_csv", "--traj_csv", type=str, default=None)
@option("-buffer_csv", "--buffer_csv", type=str, required=True)
@option("-out_dir", "--out_dir", type=str, default=None)
@option("-dpi", "--dpi", type=int, default=300)
def visualize_combined_traj(traj_csv, buffer_csv, out_dir, dpi):
    if traj_csv is not None:
        make_buffer_grid(traj_csv, buffer_csv, maxtime=1800, step=5)
    fig_stacked_traj(buffer_csv, os.path.join(out_dir, "stacked_traj.png") if out_dir else None)
    fig_specific_trajs(buffer_csv, out_dir if out_dir else None, 0.002, 0.0015, dpi)
    fig_primary_lunch_time(buffer_csv, out_dir if out_dir else None, 600, 840, dpi, False)