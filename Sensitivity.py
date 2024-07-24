import itertools
import random
import pandas as pd
from model import AdaptationModel

# Iterate over each combination of parameters
def analyse(run, financial, trait, ext):
    # Reset the random seed for each iteration
    random.seed(10)

    # Initialize the model with the new parameters
    model = AdaptationModel(number_of_households=50, flood_map_choice="500yr", network="watts_strogatz",
                            w_financial=financial, w_trait=trait,
                            w_ext=ext)  # Assuming the model accepts these parameters

    # Run the model for 100 steps
    for step in range(100):
        model.step()

    # Assuming you have a method to collect relevant results
    model_data = model.datacollector.get_model_vars_dataframe()
    agent_data = model.datacollector.get_agent_vars_dataframe()
    return [run, financial, trait, ext, agent_data, model_data]
