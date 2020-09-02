from Environments.FourRoomGridWorld import FourRoomGridWorld
from Problems.LearnEightPoliciesTileCodingFeat import LearnEightPoliciesTileCodingFeat

if __name__ == "__main__":
    actions = {
        0: 'up',
        1: 'down',
        2: 'right',
        3: 'left',
    }
    env = FourRoomGridWorld()
    problem = LearnEightPoliciesTileCodingFeat()
    state = env.reset()
    env.render()
    is_terminal = False
    for step in range(10):
        a = problem.select_target_action(state, policy_id=3)
        next_state, r, is_terminal, info = env.step(a)
        x, y, x_p, y_p, is_rand, selected_action = info.values()
        print(
            f'sept:{step}, '
            f'state({state}):({x},{y}), '
            f'action: {actions[a]}, '
            f'environment_action: {actions[selected_action]}, '
            f'next_state({next_state}):({x_p},{y_p}), '
            f'stochasticity:{is_rand}, '
            f'terminal:{is_terminal}'
        )
        state = next_state
        env.render()
        if is_terminal:
            break
