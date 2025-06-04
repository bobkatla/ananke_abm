"""
Command Line Interface for Ananke ABM.

This module provides the main entry point for the ananke command line tool.
"""

import click
from ananke_abm import __version__


@click.group()
@click.version_option(version=__version__, prog_name="ananke")
def main():
    """Ananke ABM - Agent-Based Model for synthetic population data and activity predictions."""
    pass


@main.command()
def info():
    """Display information about the Ananke ABM package."""
    click.echo(f"Ananke ABM version {__version__}")
    click.echo("Agent-Based Model: using synthetic population data for activity predictions")
