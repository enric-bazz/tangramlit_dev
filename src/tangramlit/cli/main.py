"""Main entry point for tangramlit CLI commands."""

import argparse
import warnings
from typing import Optional

from . import train_mapper, tune_hyperparameters, crossvalidate_genes


def main(argv: Optional[list] = None):
    warnings.filterwarnings("ignore")

    # Top level parser
    parser = argparse.ArgumentParser(
        prog="tangramlit",
        description="Tangram-Lightning command-line interface",
    )
    
    # Subcommand container
    subparsers = parser.add_subparsers(
        dest="command",  # name of attribute storing subcommands
        required=True,  # exits if command is missing   
        help="Available commands", 
                                       )

    # Register aubcommands
    train_mapper.register(subparsers)
    tune_hyperparameters.register(subparsers)
    crossvalidate_genes.register(subparsers)
    
    # Parse arguments
    args = parser.parse_args(argv)
    
    # Dispatch to the correct command
    args.func(args) 

if __name__ == '__main__':
    main()