"""
Command Line Interface for Ananke ABM.

This module provides the main entry point for the ananke command line tool.
"""

import click
from ananke_abm import __version__
from ananke_abm.cli.run_models.traj_embed import traj_embed
from ananke_abm.cli.run_models.gen_n_val_traj import gval_traj

@click.group()
@click.version_option(version=__version__, prog_name="ananke")
def main():
    """Ananke ABM - Agent-Based Model for synthetic population data and activity predictions."""
    pass
main.add_command(traj_embed)
main.add_command(gval_traj)

@main.command()
def info():
    """Display information about the Ananke ABM package."""
    click.echo(f"Ananke ABM version {__version__}")
    click.echo("Agent-Based Model: using synthetic population data for activity predictions")
