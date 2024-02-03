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
    tick = observation.get('tick')
    coor = observation['unit_state'][current_unit_id]['coordinates']
    dist = abs(coor[0] - 8) + abs(coor[1] - 8)
    return min(-0.001, 0.003*(200 - tick)/(15-dist+0.0000001))

"""
Reward function definition:
1. +0.5: when dealing 1 hp for 1 enemy
2. +1: when killing opponent
3. +1: when killing all 3 opponents
4. -0.25: when losing 1 hp for 1 teammate
5. -0.5: when losing teammate
6. -1: when losing all 3 teammates
7. -0.01: the longer game the bigger punishment is
8. -0.000666: the unit is in a cell within reach of a bomb
9. +0.002: the unit is in a safe cell when there is an active bomb nearby 
10. +0.1: the unit activated bomb near an obstacle
"""
def calculate_reward(prev_observation: Observation, next_observation: Observation, current_agent_id: str, current_unit_id: str):
    reward = 0        

    # 1. +0.5: when dealing 1 hp for 1 enemy

    prev_enemy_units_hps = find_enemy_units_hps(prev_observation, current_agent_id)
    next_enemy_units_hps = find_enemy_units_hps(next_observation, current_agent_id)
    
    enemy_units_hps_diff = prev_enemy_units_hps - next_enemy_units_hps
    if enemy_units_hps_diff > 0:
        reward += (enemy_units_hps_diff * 0.5)

    # 2. +1: when killing opponent

    prev_enemy_units_alive = find_enemy_units_alive(prev_observation, current_agent_id)
    next_enemy_units_alive = find_enemy_units_alive(next_observation, current_agent_id)

    if prev_enemy_units_alive > next_enemy_units_alive:
        reward += 1

    # 3. +1: when killing all 3 opponents

    if next_enemy_units_alive == 0:
        reward += 1

    # 4. -0.25: when losing 1 hp for 1 teammate

    prev_my_units_hps = find_my_units_hps(prev_observation, current_agent_id)
    next_my_units_hps = find_my_units_hps(next_observation, current_agent_id)

    my_units_hps_diff = prev_my_units_hps - next_my_units_hps
    if my_units_hps_diff > 0:
        reward += (my_units_hps_diff * -0.25)

    # 5. -0.5: when losing teammate

    prev_my_units_alive = find_my_units_alive(prev_observation, current_agent_id)
    next_my_units_alive = find_my_units_alive(next_observation, current_agent_id)

    if next_my_units_alive < prev_my_units_alive:
        reward += (-0.5)

    # 6. -1: when losing all 3 teammates

    if next_my_units_alive == 0:
        reward += (-1)

    # 7. -0.01: the longer game the bigger punishment is

    reward += (-0.01)

    # 8. -0.000666: the agent is in a cell within reach of a bomb

    prev_within_reach_of_a_bomb = unit_within_reach_of_a_bomb(prev_observation, current_unit_id)
    next_within_reach_of_a_bomb = unit_within_reach_of_a_bomb(next_observation, current_unit_id)

    if not prev_within_reach_of_a_bomb and next_within_reach_of_a_bomb:
        reward += (-0.000666)

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

    return torch.tensor(reward, dtype=torch.float32).reshape(1)

"""
Reward function definition:
1. +0.25: when dealing 1 hp for 1 enemy
2. +0.5: when killing opponent
3. +1: when killing all 3 opponents
4. -0.5: when losing 1 hp for 1 teammate
5. -1: when losing teammate
6. -1: when losing all 3 teammates
7. \min\left(-0.001,\ \frac{0.003\left(200-x\right)}{15-a}\right): the longer game the bigger punishment is
8. -0.002: the unit is in a cell within reach of a bomb
9. +0.002: the unit is in a safe cell when there is an active bomb nearby 
10. +0.05: the unit activated bomb near an obstacle
11. -0.1: bumping into the wall
12. -0.1: placing bomb on existing bomb
13. -0.1: exiting limit in placing bombs
14. +0.15: pick up FreezePowerup
15. +0.15: pick up BlastPowerup
"""
def calculate_reward_stat(prev_observation: Observation, action: int, next_observation: Observation, current_agent_id: str, current_unit_id: str):

    reward_dict = {
        "hit enemy": 0.25,
        "kill enemy": 0.5,
        "win": 1,
        "hit ally": -0.5,
        "kill ally": -1,
        "lose": -1,
        "time": -0.001, # also function "compute_time_reward" could be used
        "danger cell": -0.002,
        "safe cell": 0.002,
        "hit obstacle": 0.05,
        "bump into wall": -0.1,
        "bomb on bomb": -0.1,
        "too much bombs": -0.1,
        "FP": +0.15, # FreezePowerup
        "BP": +0.15 #BlastPowerup 
    }

    reward_list = []
    reward = 0        

    # 1. +0.25: when dealing 1 hp for 1 enemy

    prev_enemy_units_hps = find_enemy_units_hps(prev_observation, current_agent_id)
    next_enemy_units_hps = find_enemy_units_hps(next_observation, current_agent_id)
    
    enemy_units_hps_diff = prev_enemy_units_hps - next_enemy_units_hps
    if enemy_units_hps_diff > 0:
        reward += (enemy_units_hps_diff * reward_dict['hit enemy'])
        reward_list.append({"class": "hit enemy", "reward": enemy_units_hps_diff * reward_dict['hit enemy']})

    # 2. +0.5: when killing opponent

    prev_enemy_units_alive = find_enemy_units_alive(prev_observation, current_agent_id)
    next_enemy_units_alive = find_enemy_units_alive(next_observation, current_agent_id)

    if prev_enemy_units_alive > next_enemy_units_alive:
        reward += reward_dict['kill enemy']
        reward_list.append({"class": "kill enemy", "reward": reward_dict["kill enemy"]})

    # 3. +1: when killing all 3 opponents

    if next_enemy_units_alive == 0:
        reward += reward_dict["win"]
        reward_list.append({"class": "win", "reward": reward_dict["win"]})

    # 4. -0.5: when losing 1 hp for 1 teammate

    prev_my_units_hps = find_my_units_hps(prev_observation, current_agent_id)
    next_my_units_hps = find_my_units_hps(next_observation, current_agent_id)

    my_units_hps_diff = prev_my_units_hps - next_my_units_hps
    if my_units_hps_diff > 0:
        reward += (my_units_hps_diff * reward_dict["hit ally"])
        reward_list.append({"class": "hit ally", "reward": my_units_hps_diff * reward_dict["hit ally"]})

    # 5. -1: when losing teammate

    prev_my_units_alive = find_my_units_alive(prev_observation, current_agent_id)
    next_my_units_alive = find_my_units_alive(next_observation, current_agent_id)

    if next_my_units_alive < prev_my_units_alive:
        reward += reward_dict["kill ally"]
        reward_list.append({"class": "kill ally", "reward": reward_dict["kill ally"]})

    # 6. -1: when losing all 3 teammates

    if next_my_units_alive == 0:
        reward += reward_dict["lose"]
        reward_list.append({"class": "lose", "reward": reward_dict["lose"]})

    # 7. -0.01: the longer game the bigger punishment is

    reward += reward_dict["time"] #compute_time_reward(next_observation, current_unit_id)
    reward_list.append({"class": "time", "reward": reward_dict["time"]})

    # 8. -0.002: the agent is in a cell within reach of a bomb

    #prev_within_reach_of_a_bomb = unit_within_reach_of_a_bomb(prev_observation, current_unit_id)
    next_within_reach_of_a_bomb = unit_within_reach_of_a_bomb(next_observation, current_unit_id)

    if next_within_reach_of_a_bomb:
        reward += reward_dict["danger cell"]
        reward_list.append({"class": "danger cell", "reward": reward_dict["danger cell"]})

    # 9. +0.002: the agent is in a safe cell when there is an active bomb nearby

    #prev_within_safe_cell_nearby_bomb = unit_within_safe_cell_nearby_bomb(prev_observation, current_unit_id)
    next_within_safe_cell_nearby_bomb = unit_within_safe_cell_nearby_bomb(next_observation, current_unit_id)

    if next_within_safe_cell_nearby_bomb:
        reward += reward_dict["safe cell"]
        reward_list.append({"class": "safe cell", "reward": reward_dict["safe cell"]})

    # 10. +0.1: the unit activated bomb near an obstacle

    prev_activated_bomb_near_an_obstacle = unit_activated_bomb_near_an_obstacle(prev_observation, current_unit_id)
    next_activated_bomb_near_an_obstacle = unit_activated_bomb_near_an_obstacle(next_observation, current_unit_id)
    
    if not prev_activated_bomb_near_an_obstacle and next_activated_bomb_near_an_obstacle:
        reward += reward_dict["hit obstacle"]
        reward_list.append({"class":"hit obstacle", "reward": reward_dict["hit obstacle"]})
    
    # 11. -0.1: bumping into the wall
        
    a = ACTIONS[action]
    prev_coor = prev_observation['unit_state'][current_unit_id]['coordinates']
    next_coor = next_observation['unit_state'][current_unit_id]['coordinates']
    if a in ["up", "down", "left", "right"] and prev_coor == next_coor:
        reward += reward_dict["bump into wall"]
        reward_list.append({"class": "bump into wall", "reward": reward_dict["bump into wall"]})

    # 12. -0.1: placing bomb on existing bomb
    
    bombs = list(filter(lambda entity: entity.get("type") == "b", prev_observation['entities']))
    if a == "bomb" and any([next_coor == [b.get('x'), b.get('y')] for b in bombs]):
        reward += reward_dict["bomb on bomb"]
        reward_list.append({"class": "bomb on bomb", "reward": reward_dict["bomb on bomb"]})

    # 13. -0.1: exiting limit in placing bombs
        
    if a == "bomb" and len(get_unit_activated_bombs(prev_observation, current_unit_id)) >= 3:
        reward += reward_dict["too many bombs"]
        reward_list.append({"class": "too many bomb", "reward": reward_dict["too many bomb"]})

    # 14. +0.15: pick up FreezePowerup
        
    fps = list(filter(lambda entity: entity.get("type") == "fp", prev_observation['entities']))
    if any([next_coor == [fp.get("x"), fp.get("y")] for fp in fps]):
        reward += reward_dict["FP"]
        reward_list.append({"class": "FP", "reward": reward_dict["FP"]})
        
    # 15. +0.15: pick up BlastPowerup
    
    bps = list(filter(lambda entity: entity.get("type") == "bp", prev_observation['entities']))
    if any([next_coor == [bp.get("x"), bp.get("y")] for bp in bps]):
        reward += reward_dict["BP"]
        reward_list.append({"class": "BP", "reward": reward_dict["BP"]})

    return torch.tensor(reward, dtype=torch.float32).reshape(1), reward_list
