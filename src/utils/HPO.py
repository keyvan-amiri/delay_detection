# -*- coding: utf-8 -*-
"""
Created on Fri Sep 19 14:48:03 2025
@author: Keyvan Amiri Elyasi
"""
from ax.service.ax_client import AxClient
from ax.service.utils.instantiation import ObjectiveProperties
from ax.modelbridge.generation_strategy import GenerationStrategy, GenerationStep
from ax.modelbridge.registry import Models

def get_hpo_client():
    # Define HPO strategy: first 5 trials Sobol + remaining trials Bayesian
    gs = GenerationStrategy(
        steps=[
            GenerationStep(model=Models.SOBOL, num_trials=5),
            GenerationStep(model=Models.GPEI, num_trials=-1),
            ]
        )
    # Initialize AX client with this strategy
    ax_client = AxClient(generation_strategy=gs)
   
    # Define the search space
    ax_client.create_experiment(
        name="HPO_DIR",
        parameters=[
            {"name": "lr", 
             "type": "range", 
             "bounds": [1e-5, 1e-2], 
             "value_type": "float",
             "log_scale": True},
        ],
        objectives={
            "valid_loss": ObjectiveProperties(minimize=True)
        },
        parameter_constraints=[],
        outcome_constraints=[],
    ) 
    return ax_client