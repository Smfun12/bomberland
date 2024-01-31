import torch

from components.environment.config import ACTIONS
from components.types import Observation
from components.utils.observation import (
    get_nearest_active_bomb, 
    get_nearest_obstacle,
    get_unit_activated_bombs, 
)
from components.utils.metrics import manhattan_distance


def find_my_units_alive(observation: Observation, current_agent_id: str) -> int:
    alive = 0
    for unit_props in observation['unit_state'].values():
        if unit_props['agent_id'] == current_agent_id:
            if unit_props['hp'] != 0:
                alive += 1
    return alive


def find_enemy_units_alive(observation: Observation, current_agent_id: str) -> int:
    alive = 0
    for unit_props in observation['unit_state'].values():
        if unit_props['agent_id'] != current_agent_id:
            if unit_props['hp'] != 0:
                alive += 1
    return alive


def find_my_units_hps(observation: Observation, current_agent_id: str) -> int:
    hps = 0
    for unit_props in observation['unit_state'].values():
        if unit_props['agent_id'] == current_agent_id:
            hps += unit_props['hp']
    return hps


def find_enemy_units_hps(observation: Observation, current_agent_id: str) -> int:
    hps = 0
    for unit_props in observation['unit_state'].values():
        if unit_props['agent_id'] != current_agent_id:
            hps += unit_props['hp']
    return hps


def find_current_tick(observation: Observation) -> int:
    tick = observation['tick']
    return tick


def unit_within_reach_of_a_bomb(observation: Observation, current_unit_id: str):
    unit = observation['unit_state'][current_unit_id]
    unit_coords = unit['coordinates']
    nearest_bomb = get_nearest_active_bomb(observation, current_unit_id)
    if nearest_bomb is None:
        return False
    nearest_bomb_coords = [nearest_bomb['x'], nearest_bomb['y']]
    within_reach_of_a_bomb = manhattan_distance(unit_coords, nearest_bomb_coords) <= nearest_bomb['blast_diameter']
    return within_reach_of_a_bomb


def unit_within_safe_cell_nearby_bomb(observation: Observation, current_unit_id: str):
    unit = observation['unit_state'][current_unit_id]
    unit_coords = unit['coordinates']
    nearest_bomb = get_nearest_active_bomb(observation, current_unit_id)
    if nearest_bomb is None:
        return False
    nearest_bomb_coords = [nearest_bomb['x'], nearest_bomb['y']]
    within_safe_cell_nearby_bomb = manhattan_distance(unit_coords, nearest_bomb_coords) > nearest_bomb['blast_diameter']
    return within_safe_cell_nearby_bomb


"""
Bomb definition: {'created': 74, 'x': 11, 'y': 10, 'type': 'b', 'unit_id': 'd', 'agent_id': 'b', 'expires': 104, 'hp': 1, 'blast_diameter': 3}
"""
def unit_activated_bomb_near_an_obstacle(observation: Observation, current_unit_id: str):
    unit_activated_bombs = get_unit_activated_bombs(observation, current_unit_id)
    if not len(unit_activated_bombs):
        return False
    for unit_bomb in unit_activated_bombs:
        unit_bomb_coords = [unit_bomb['x'], unit_bomb['y']]
        nearest_obstacle = get_nearest_obstacle(observation, unit_bomb_coords)
        if nearest_obstacle is None:
            continue
        nearest_obstacle_coords = [nearest_obstacle['x'], nearest_obstacle['y']]
        if manhattan_distance(unit_bomb_coords, nearest_obstacle_coords) <= unit_bomb['blast_diameter']:
            return True
    return False

def compute_time_reward(observation: Observation, current_unit_id):
    tick = observation['tick']
    coor = observation['unit_state'][current_unit_id]['coordinates']
    dist = abs(coor[0] - 8) + abs(coor[1] - 8)
    return min(-0.01, 0.003(200 - tick)/(15-dist))


"""
Reward function definition:
1. +0.25: when dealing 1 hp for 1 enemy
2. +0.5: when killing opponent
3. +1: when killing all 3 opponents
4. -0.5: when losing 1 hp for 1 teammate
5. -1: when losing teammate
6. -1: when losing all 3 teammates
7. \min\left(-0.01,\ \frac{0.003\left(200-x\right)}{15-a}\right): the longer game the bigger punishment is
8. -0.002: the unit is in a cell within reach of a bomb
9. +0.002: the unit is in a safe cell when there is an active bomb nearby 
10. +0.1: the unit activated bomb near an obstacle
11. -0.1: bumping into the wall
"""
def calculate_reward(prev_observation: Observation, action: int, next_observation: Observation, current_agent_id: str, current_unit_id: str):
    reward = 0        

    # 1. +0.25: when dealing 1 hp for 1 enemy

    prev_enemy_units_hps = find_enemy_units_hps(prev_observation, current_agent_id)
    next_enemy_units_hps = find_enemy_units_hps(next_observation, current_agent_id)
    
    enemy_units_hps_diff = prev_enemy_units_hps - next_enemy_units_hps
    if enemy_units_hps_diff > 0:
        reward += (enemy_units_hps_diff * 0.25)

    # 2. +0.5: when killing opponent

    prev_enemy_units_alive = find_enemy_units_alive(prev_observation, current_agent_id)
    next_enemy_units_alive = find_enemy_units_alive(next_observation, current_agent_id)

    if prev_enemy_units_alive > next_enemy_units_alive:
        reward += 0.5

    # 3. +1: when killing all 3 opponents

    if next_enemy_units_alive == 0:
        reward += 1

    # 4. -0.5: when losing 1 hp for 1 teammate

    prev_my_units_hps = find_my_units_hps(prev_observation, current_agent_id)
    next_my_units_hps = find_my_units_hps(next_observation, current_agent_id)

    my_units_hps_diff = prev_my_units_hps - next_my_units_hps
    if my_units_hps_diff > 0:
        reward += (my_units_hps_diff * -0.5)

    # 5. -1: when losing teammate

    prev_my_units_alive = find_my_units_alive(prev_observation, current_agent_id)
    next_my_units_alive = find_my_units_alive(next_observation, current_agent_id)

    if next_my_units_alive < prev_my_units_alive:
        reward += (-1)

    # 6. -1: when losing all 3 teammates

    if next_my_units_alive == 0:
        reward += (-1)

    # 7. -0.01: the longer game the bigger punishment is

    reward += compute_time_reward(next_observation, current_unit_id)

    # 8. -0.002: the agent is in a cell within reach of a bomb

    prev_within_reach_of_a_bomb = unit_within_reach_of_a_bomb(prev_observation, current_unit_id)
    next_within_reach_of_a_bomb = unit_within_reach_of_a_bomb(next_observation, current_unit_id)

    if not prev_within_reach_of_a_bomb and next_within_reach_of_a_bomb:
        reward += (-0.002)

    # 9. +0.002: the agent is in a safe cell when there is an active bomb nearby

    prev_within_safe_cell_nearby_bomb = unit_within_safe_cell_nearby_bomb(prev_observation, current_unit_id)
    next_within_safe_cell_nearby_bomb = unit_within_safe_cell_nearby_bomb(next_observation, current_unit_id)

    if not prev_within_safe_cell_nearby_bomb and next_within_safe_cell_nearby_bomb:
        reward += 0.002

    # 10. +0.1: the unit activated bomb near an obstacle

    prev_activated_bomb_near_an_obstacle = unit_activated_bomb_near_an_obstacle(prev_observation, current_unit_id)
    next_activated_bomb_near_an_obstacle = unit_activated_bomb_near_an_obstacle(next_observation, current_unit_id)
    
    if not prev_activated_bomb_near_an_obstacle and next_activated_bomb_near_an_obstacle:
        reward += 0.1
    
    # 11. -0.1: bump into the wall
        
    a = ACTIONS[action]
    prev_coor = prev_observation['unit_state'][current_unit_id]['coordinates']
    next_coor = next_observation['unit_state'][current_unit_id]['coordinates']
    if a in ["up", "down", "left", "right"] and prev_coor == next_coor:
        reward -= 0.1

    return torch.tensor(reward, dtype=torch.float32).reshape(1)
