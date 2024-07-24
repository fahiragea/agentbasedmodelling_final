# Importing necessary libraries
import random
from mesa import Agent
from shapely.geometry import Point
from shapely import contains_xy
import numpy as np
import model
import math

# Import functions from functions.py
from functions import generate_random_location_within_map_domain, get_flood_depth, calculate_basic_flood_damage, floodplain_multipolygon

random.seed(10)

# Define the Households agent class
class Households(Agent):
    """
    An agent representing a household in the model.
    Each household has a flood depth attribute which is randomly assigned for demonstration purposes.
    In a real scenario, this would be based on actual geographical data or more complex logic.
    """

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.is_adapted = False  # Initial adaptation status set to False
        self.w_financial = model.w_financial
        self.w_trait = model.w_trait
        self.w_ext = model.w_ext
        # getting flood map values
        # Get a random location on the map
        loc_x, loc_y = generate_random_location_within_map_domain()
        self.location = Point(loc_x, loc_y)

        # Check whether the location is within floodplain
        self.in_floodplain = False
        if contains_xy(geom=floodplain_multipolygon, x=self.location.x, y=self.location.y):
            self.in_floodplain = True

        # Get the estimated flood depth at those coordinates. 
        # the estimated flood depth is calculated based on the flood map (i.e., past data) so this is not the actual flood depth
        # Flood depth can be negative if the location is at a high elevation
        self.flood_depth_estimated = get_flood_depth(corresponding_map=model.flood_map, location=self.location, band=model.band_flood_img)
        self.small_flood_depth_estimated = self.flood_depth_estimated / 6
        # handle negative values of flood depth
        if self.flood_depth_estimated < 0:
            self.flood_depth_estimated = 0
            self.small_flood_depth_estimated = 0
        
        # calculate the estimated flood damage given the estimated flood depth. Flood damage is a factor between 0 and 1
        self.flood_damage_estimated = calculate_basic_flood_damage(flood_depth=self.flood_depth_estimated)
        self.small_flood_damage_estimated = self.flood_depth_estimated / 6

        # Add an attribute for the actual flood depth. This is set to zero at the beginning of the simulation since there is not flood yet
        # and will update its value when there is a shock (i.e., actual flood). Shock happens at some point during the simulation
        self.flood_depth_actual = 0
        
        #calculate the actual flood damage given the actual flood depth. Flood damage is a factor between 0 and 1
        self.flood_damage_actual = calculate_basic_flood_damage(flood_depth=self.flood_depth_actual)
        self.flood_opinion = 0

        #Give each household a savings & income (income is partially based on savings):
        if self.location.x > 250000 and self.location.x < 270000 and self.location.y > 3300000 and self.location.y < 3320000:
            self.poor = True
            self.savings = random.randint(2000,10000)
            richness = self.savings / 10000
            self.income = richness * 7000 * 3 * 0.1 #income per tick is quarter year, 10% goes to savings
        else:
            self.poor = False
            self.savings = random.randint(28000, 100000)
            richness = self.savings / 100000
            self.income = richness * 25000 * 3 * 0.1#income per tick is quarter year, 10% goes to savings

        #Give each house a certain real estate value, this is also based on income.
        #Also give them insurance or not
        if self.poor == True:
            self.real_estate_value = 340000 * richness
        else:
            self.real_estate_value = 1000000 * richness
            if self.real_estate_value < 340000:
                self.real_estate_value = 340000
        self.insured = True #bool(random.getrandbits(1))

        #Adaptation costs and efficacy
        self.adaptation_costs = self.real_estate_value * 0.1
        self.adaptation_efficacy = 1.3 #make it 30% more effective against floods

        #Personal values
        self.opinion = random.uniform(0, 1)
        self.personal_values = self.opinion

        #Social influence
        self.social_status = random.uniform(0,1)
        self.persuasiveness = random.uniform(0,1)
        #self.friends = acquintances
        #self.social_influence = insert formula TO DO

        self.adapted_time = None

    # Function to count friends who can be influencial.
    def count_friends(self, radius):
        """Count the number of neighbors within a given radius (number of edges away). This is social relation and not spatial"""
        f = self.model.grid.get_neighborhood(self.pos, include_center=False, radius=radius)
        self.friends = {}
        for agent in self.model.schedule.agents:
            if agent.unique_id in f:
                for u,v,attr in self.model.G.edges(data = True):
                    if agent.unique_id == u or agent.unique_id == v:
                        self.friends[agent] = next(iter(attr.values()))
        return len(self.friends)

    def step(self):
        # Logic for adaptation based on estimated flood damage and a random chance.
        # These conditions are examples and should be refined for real-world applications
        if self.flood_opinion > 0:
            self.flood_opinion = self.flood_opinion - 0.010
        else:
            self.flood_opinion = 0
        self.savings += self.income
        if self.is_adapted == False:
            if self.insured == True:
                flood_costs = self.flood_damage_estimated * self.real_estate_value * 0.2
                adapted_depth = self.flood_depth_estimated - 1
                adapted_costs = calculate_basic_flood_damage(
                    flood_depth=adapted_depth) * self.real_estate_value * 0.5 + self.adaptation_costs
                financial_influence = flood_costs / adapted_costs
            else:
                flood_costs = self.flood_damage_estimated * self.real_estate_value
                adapted_depth = self.flood_depth_estimated - 1
                adapted_costs = calculate_basic_flood_damage(
                    flood_depth=adapted_depth) * self.real_estate_value + self.adaptation_costs
                financial_influence = flood_costs / adapted_costs
            financial_sigmoid = 1 / (1 + np.exp(-15 * (financial_influence - 1)))
            self.financial_diff = (financial_influence - self.opinion) * self.w_financial
            # print(f'id is{self.unique_id}\n floodcost is {flood_costs}\n, adaptation costs {self.adaptation_costs}\n,financial influence is{financial_influence}\n sigmoid is {financial_sigmoid}')

            self.w_friends = 0
            freq_total = 0
            self.external_influence = random.uniform(-0.6, 1) * self.w_ext
            for friend, freq in self.friends.items():
                self.opinion_diff = friend.opinion - self.opinion
                freq_total += freq
                self.w_friends += (friend.persuasiveness + friend.social_status) / 2 * freq
            self.w_friends = self.w_friends / freq_total
            self.opinion_diff = self.opinion_diff * self.w_friends
            self.opinion = self.opinion + self.opinion_diff + self.external_influence + self.financial_diff + self.flood_opinion
            self.trait_diff = (self.personal_values - self.opinion) * self.w_trait
            self.opinion = self.opinion + self.trait_diff
            if self.opinion >= 0.75 and self.savings > self.adaptation_costs:
                self.savings -= self.adaptation_costs
                self.is_adapted = True
                self.adapted_time = self.model.schedule.steps
        else:
            if self.insured == True:
                flood_costs = self.flood_damage_estimated * self.real_estate_value * 0.2
                adapted_depth = self.flood_depth_estimated - 1
                adapted_costs = calculate_basic_flood_damage(flood_depth=adapted_depth) * self.real_estate_value * 0.5 + self.adaptation_costs
                financial_influence = flood_costs / adapted_costs
            else:
                flood_costs = self.flood_damage_estimated * self.real_estate_value
                adapted_depth = self.flood_depth_estimated - 1
                adapted_costs = calculate_basic_flood_damage(flood_depth=adapted_depth) * self.real_estate_value + self.adaptation_costs
                financial_influence = flood_costs / adapted_costs
            financial_sigmoid = 1 / (1 + np.exp(-15 * (financial_influence - 1)))
            self.financial_diff = (financial_influence - self.opinion) * self.w_financial
            #print(f'id is{self.unique_id}\n floodcost is {flood_costs}\n, adaptation costs {self.adaptation_costs}\n,financial influence is{financial_influence}\n sigmoid is {financial_sigmoid}')

            self.w_friends = 0
            freq_total = 0
            self.external_influence = random.uniform(-0.6, 1) * self.w_ext
            for friend, freq in self.friends.items():
                self.opinion_diff = friend.opinion - self.opinion
                freq_total += freq
                self.w_friends += (friend.persuasiveness + friend.social_status) / 2 * freq
            self.w_friends = self.w_friends / freq_total
            self.trait_diff = (self.personal_values - self.opinion) * self.w_trait
            self.opinion_diff = self.opinion_diff * self.w_friends
            self.opinion = self.opinion + self.opinion_diff + self.external_influence + self.financial_diff + self.flood_opinion
            self.trait_diff = (self.personal_values - self.opinion) * self.w_trait
            self.opinion = self.opinion + self.trait_diff
            #print(f'id is{self.unique_id}\n opinion is{self.opinion}\n financial is {financial_diff}\n friend opinion is{opinion_diff}\n w is {w}\n sigmoid {financial_sigmoid}\n financial influence {financial_influence}')
