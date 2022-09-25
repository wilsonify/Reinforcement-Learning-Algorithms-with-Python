"""
Policy iteration cycles between
policy evaluation and policy improvement
Eventually, the result is an optimal policy.

After an initialization phase, the outer loop iterates through policy evaluation and policy
iteration until a stable policy is found.
On each of these iterations, policy evaluation
evaluates the policy found during the preceding policy improvement steps,
which in turn use the estimated value function.
"""
import numpy as np
import gym


def eval_state_action(env, V, s, a, gamma=0.99):
    """
    update using the Bellman equation
    """
    return np.sum([p * (rew + gamma * V[next_s]) for p, next_s, rew, _ in env.P[s][a]])


def policy_evaluation(env, nS, V, policy, eps=0.0001):
    """
    Policy evaluation. Update the value function until it reach a steady state
    """
    while True:
        delta = 0
        for s in range(nS):  # loop over all states
            old_v = V[s]
            V[s] = eval_state_action(env, V, s, policy[s])
            delta = max(delta, np.abs(old_v - V[s]))

        if delta < eps:
            break


def policy_improvement(env, nS, nA, V, policy):
    """
    Policy improvement. Update the policy based on the value function
    """
    policy_stable = True
    for s in range(nS):  # loop over all states
        old_a = policy[s]
        # update the policy with the action that bring to the highest state value
        policy[s] = np.argmax([eval_state_action(env, V, s, a) for a in range(nA)])
        if old_a != policy[s]:
            policy_stable = False

    return policy_stable


def run_episodes(env, policy, num_games=100):
    """
    Run some games to test a policy
    """
    tot_rew = 0
    state = env.reset()

    for _ in range(num_games):
        done = False
        while not done:
            # select the action accordingly to the policy
            # observation, reward, done, info
            # observation  reward, terminated,  truncated: info

            next_state, reward, done, truncated, info = env.step(policy[state])
            print(f"next_state={next_state}")
            print(f"reward={reward}")
            print(f"done={done}")
            print(f"truncated={truncated}")
            print(f"info={info}")

            state = next_state
            tot_rew += reward
            if done:
                state = env.reset()

    print("Won %i of %i games!" % (tot_rew, num_games))


def main():
    # create the environment
    env = gym.make(
        "FrozenLake-v1",
        new_step_api=True  # Whether to use old or new step API (StepAPICompatibility wrapper). removed at v1.0
    )

    # unwrap it to have additional information from it
    env = env.unwrapped

    # spaces dimension
    nA = env.action_space.n
    nS = env.observation_space.n

    # initializing value function and policy
    V = np.zeros(nS)
    policy = np.zeros(nS)

    # some useful variable
    policy_stable = False
    it = 0

    while not policy_stable:
        policy_evaluation(
            env=env,
            nS=nS,
            V=V,
            policy=policy,
            eps=0.0001
        )
        policy_stable = policy_improvement(env, nS, nA, V, policy)
        it += 1

    print(f"Converged after {it} policy iterations")
    run_episodes(
        env=env,
        policy=policy,
        num_games=100

    )
    print(V.reshape((4, 4)))
    print(policy.reshape((4, 4)))


if __name__ == "__main__":
    main()
