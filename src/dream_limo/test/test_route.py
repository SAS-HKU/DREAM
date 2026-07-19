import numpy as np

from dream_limo.core.route import anchored_lane_change_y
from dream_limo.limo_scale import default_deployment_config


def test_station_anchored_route_stays_left_until_truck_is_cleared():
    config = default_deployment_config()
    arena = config.arena
    x = np.asarray(
        [
            arena.merge_request_x,
            arena.merge_path_x_min - 0.01,
            arena.merge_path_x_min,
            0.5 * (arena.merge_path_x_min + arena.merge_path_x_max),
            arena.merge_path_x_max,
            arena.merge_path_x_max + 0.20,
        ]
    )
    y = anchored_lane_change_y(
        x,
        source_y=arena.lane_centers[arena.ego_lane],
        target_y=arena.lane_centers[arena.target_lane],
        start_x=arena.merge_path_x_min,
        end_x=arena.merge_path_x_max,
    )
    assert y[0] == arena.lane_centers[arena.ego_lane]
    assert y[1] == arena.lane_centers[arena.ego_lane]
    assert y[2] == arena.lane_centers[arena.ego_lane]
    assert arena.lane_centers[arena.target_lane] < y[3] < arena.lane_centers[arena.ego_lane]
    assert y[4] == arena.lane_centers[arena.target_lane]
    assert y[5] == arena.lane_centers[arena.target_lane]
