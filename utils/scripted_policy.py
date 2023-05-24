import random
import numpy as np

from env.physical_env import Coord


def get_is_demo(step, params, num_envs=0):
    demo_annealing_start = params.scripted_policy_params.demo_annealing_start
    demo_annealing_end = params.scripted_policy_params.demo_annealing_end
    demo_annealing_coef = np.clip((step - demo_annealing_start) / (demo_annealing_end - demo_annealing_start), 0, 1)
    demo_prob_init = params.scripted_policy_params.demo_prob_init
    demo_prob_final = params.scripted_policy_params.demo_prob_final
    demo_prob = demo_prob_init + (demo_prob_final - demo_prob_init) * demo_annealing_coef

    if num_envs:
        return np.random.random(num_envs) < demo_prob
    else:
        return np.random.random() < demo_prob


class ScriptedPhysical:
    def __init__(self, env, params):
        self.env = env
        self.params = params
        self.action_dim = params.action_dim

        self.num_objects = env.num_objects
        self.num_rand_objects = env.num_rand_objects
        self.width = env.width
        self.height = env.height
        self.directions = [Coord(-1, 0),
                           Coord(0, 1),
                           Coord(1, 0),
                           Coord(0, -1)]
        self.policy_id = 0
        self.reset()

    def reset(self, *args):
        policy_id = self.policy_id
        self.mov_obj_idx = policy_id // (self.num_objects + self.num_rand_objects - 1)
        self.target_obj_idx = policy_id % (self.num_objects + self.num_rand_objects - 1)
        if self.target_obj_idx >= self.mov_obj_idx:
            self.target_obj_idx += 1
        self.direction_idx = np.random.randint(4)
        self.direction = self.directions[self.direction_idx]
        self.success_steps = 0
        self.random_policy = np.random.rand() < 0.1

        n_policies = self.num_objects * (self.num_objects + self.num_rand_objects - 1)
        self.policy_id = (policy_id + 1) % n_policies

    def get_action(self, obj_idx, offset):
        if obj_idx >= self.num_objects:
            return 5 * np.random.randint(self.num_objects)
        return 5 * obj_idx + self.directions.index(offset) + 1

    def dijkstra(self, obj_idx_to_move, target_pos):
        env = self.env
        width, height = env.width, env.height
        Q = np.ones((width, height)) * np.inf
        dist = np.ones((width, height)) * np.inf
        checked = np.zeros((width, height), dtype=bool)
        for idx, obj in env.objects.items():
            checked[obj.pos.x, obj.pos.y] = True

        Q[target_pos.x, target_pos.y] = 0

        while True:
            x, y = np.unravel_index(np.argmin(Q), Q.shape)
            q = Q[x, y]
            if q == np.inf:
                break
            dist[x, y] = Q[x, y]
            checked[x, y] = True
            Q[x, y] = np.inf

            for del_x, del_y in [[0, 1], [0, -1], [1, 0], [-1, 0]]:
                new_x, new_y = x + del_x, y + del_y
                if 0 <= new_x < width and 0 <= new_y < height and not checked[new_x, new_y]:
                    if q + 1 < Q[new_x, new_y]:
                        Q[new_x, new_y] = q + 1

        mov_obj = env.objects[obj_idx_to_move]
        mov_x, mov_y = mov_obj.pos.x, mov_obj.pos.y
        min_dist = np.inf
        min_idx = self.directions[0]
        for dir in self.directions:
            new_x, new_y = mov_x + dir.x, mov_y + dir.y
            if 0 <= new_x < width and 0 <= new_y < height:
                if dist[new_x, new_y] < min_dist:
                    min_dist = dist[new_x, new_y]
                    min_idx = dir
        return min_idx, min_dist

    def act(self, obs):
        objects = self.env.objects
        env = self.env
        mov_obj_idx = self.mov_obj_idx
        target_obj_idx = self.target_obj_idx
        mov_obj = objects[mov_obj_idx]
        target_obj = objects[target_obj_idx]

        current_pos = mov_obj.pos
        target_pos = target_obj.pos - self.direction

        map_center = Coord(self.width // 2 + 1, self.height // 2 + 1)
        # need to push the target object from outside of the map (impossible), need to adjust the target object
        if not 0 <= target_pos.x < self.width or not 0 <= target_pos.y < self.height:
            if env.valid_move(target_obj_idx, self.direction):
                return self.get_action(target_obj_idx, self.direction)
            else:
                action_idx, dist = self.dijkstra(target_obj_idx, map_center)
                return self.get_action(target_obj_idx, action_idx)

        # unable to simply move the target object, need to plan a path for it instead (by letting it move to the center)
        pushing_pos = target_obj.pos + self.direction
        if any([obj.pos == pushing_pos for obj in objects.values() if obj != mov_obj]):
            action_idx, dist = self.dijkstra(target_obj_idx, map_center)
            if dist != np.inf:
                return self.get_action(target_obj_idx, action_idx)

        if current_pos != target_pos:
            if self.random_policy:
                return self.get_action(mov_obj_idx, random.choice(self.directions))
            else:
                action_idx, dist = self.dijkstra(mov_obj_idx, target_pos)
                if dist == np.inf:
                    return self.get_action(target_obj_idx, self.direction)
                else:
                    return self.get_action(mov_obj_idx, action_idx)
        else:
            action = self.get_action(mov_obj_idx, self.direction)
            self.success_steps += 1
            if self.success_steps > 2:
                self.reset()
            return action

    def act_randomly(self,):
        return np.random.randint(self.action_dim)


class ScriptedChemical:
    def __init__(self, env, params):
        self.env = env
        self.params = params
        self.action_dim = params.action_dim

    def reset(self, *args):
        pass

    def act(self, obs):
        return self.act_randomly()

    def act_randomly(self,):
        return np.random.randint(self.action_dim)
        # below for collecting dense graph only
        # p = np.concatenate([[i * 0.5 + 1.0] * 5 for i in range(10)])
        # p /= p.sum()
        # a = np.random.multinomial(1, p).argmax()
        # return a


class ScriptedToolUseSeries:
    def __init__(self, env, params):
        self.params = params

        pick_place_params = params.scripted_policy_params.pick_place_params
        self.explore_eps = pick_place_params.explore_eps
        self.release_prob = pick_place_params.release_prob
        self.action_scale = 20.0
        self.hover_height = 0.92
        self.push_x_thre = 0.0

        self.task = self.params.env_params.env_name

        self.action_low, self.action_high = params.action_spec

        global_low, global_high = params.normalization_range
        global_low = np.array(global_low)
        global_high = np.array(global_high)
        self.global_mean = (global_high + global_low) / 2
        self.global_scale = (global_high - global_low) / 2

        self.STATES = ["TOOL_GRASPING", "TOOL_MOVING", "PUSHING",
                       "CUBE_PICKING", "CUBE_MOVING", "POT_PICKING", "POT_LIFTING"]

    def reset(self, obs):
        self.state = "TOOL_GRASPING"
        self.reached_tool_goal = False

        self.release = False
        self.release_step = 0

    def grasping_act(self, eef_pos, object_pos):
        action = np.zeros_like(self.action_low)
        action[-1] = -np.random.rand()

        placement = object_pos - eef_pos
        xy_place, z_place = placement[:2], placement[-1]

        action_noise = 0
        if np.linalg.norm(xy_place) >= 0.02:
            if eef_pos[2] < self.hover_height:
                action[2] = self.hover_height - eef_pos[2]
            else:
                action[:2] = xy_place
            action_noise = 0.05
            action[-1] = -np.random.rand()
        elif np.linalg.norm(placement) >= 0.01:
            action[:3] = placement
        else:
            action[:3] = placement
            action[-1] = np.random.rand()

        action[:3] *= self.action_scale

        return action, action_noise

    def reaching_act(self, current_pos, goal_pos):
        action = np.zeros_like(self.action_low)
        action[:3] = (goal_pos - current_pos) * self.action_scale
        action[-1] = np.random.rand()
        action_noise = 0.05

        return action, action_noise

    def act(self, obs):
        eef_pos = obs["robot0_eef_pos"] * self.global_scale + self.global_mean
        cube_pos = obs["cube_pos"] * self.global_scale + self.global_mean
        tool_pos = obs["tool_pos"] * self.global_scale + self.global_mean
        pot_pos = obs["pot_pos"] * self.global_scale + self.global_mean
        goal = obs["goal_pos"] * self.global_scale + self.global_mean

        tool_head_pos = obs["tool_head_pos"] * self.global_scale + self.global_mean
        pot_handle_pos = obs["pot_handle_pos"] * self.global_scale + self.global_mean

        cube_pos_x = cube_pos[0]
        cube_pos_z = cube_pos[2]
        tool_head_pos_x = tool_head_pos[0]
        table_height = 0.8

        tool_head_goal_pos = cube_pos + np.array([0.1, -0.1, 0])
        tool_head_goal_dist = np.linalg.norm(tool_head_pos - tool_head_goal_pos)

        cube_pot_dist_xy = np.linalg.norm(cube_pos[:2] - pot_pos[:2])

        cube_grasped = tool_grasped = pot_grasped = cube_touching_pot = False
        cube_grasped = obs["cube_grasped"]
        tool_grasped = obs["tool_grasped"]
        pot_grasped = obs["pot_grasped"]

        # determine current state
        if self.state == "TOOL_GRASPING":
            if cube_pos_x < self.push_x_thre:
                self.state = "CUBE_PICKING"
            elif tool_grasped:
                if self.reached_tool_goal:
                    self.state = "PUSHING"
                else:
                    self.state = "TOOL_MOVING"
        elif self.state == "TOOL_MOVING":
            if cube_pos_x < self.push_x_thre:
                self.state = "CUBE_PICKING"
            elif not tool_grasped:
                self.state = "TOOL_GRASPING"
            elif tool_head_goal_dist < 0.02:
                self.reached_tool_goal = True
                self.state = "PUSHING"
        elif self.state == "PUSHING":
            if cube_pos_x < self.push_x_thre:
                self.state = "CUBE_PICKING"
            elif not tool_grasped:
                self.state = "TOOL_GRASPING"
        elif self.state == "CUBE_PICKING":
            if cube_grasped:
                self.state = "CUBE_MOVING"
        elif self.state == "CUBE_MOVING":
            if not cube_grasped:
                if cube_pot_dist_xy < 0.02:
                    self.state = "POT_PICKING"
                else:
                    self.state = "CUBE_PICKING"
        elif self.state == "POT_PICKING":
            if pot_grasped:
                self.state = "POT_LIFTING"
        elif self.state == "POT_LIFTING":
            if not pot_grasped:
                self.state = "POT_PICKING"
        else:
            raise NotImplementedError

        if np.random.rand() < self.explore_eps:
            return self.act_randomly()

        action = np.zeros_like(self.action_low)
        action[-1] = -np.random.rand()

        if self.state == "TOOL_GRASPING":
            action, action_noise = self.grasping_act(eef_pos, tool_pos)
        elif self.state == "TOOL_MOVING":
            action, action_noise = self.reaching_act(tool_head_pos, tool_head_goal_pos)
        elif self.state == "PUSHING":
            tool_head_cube_dist = np.linalg.norm(tool_head_pos - cube_pos)
            action, action_noise = self.reaching_act(tool_head_pos, cube_pos)
            if tool_head_cube_dist < 0.05:
                action[0] = -1
        elif self.state == "CUBE_PICKING":
            if tool_grasped:
                action, action_noise = np.array([0, 0, 0, -1]), 0
            else:
                action, action_noise = self.grasping_act(eef_pos, cube_pos)
        elif self.state == "CUBE_MOVING":
            cube_goal_pos = pot_pos.copy()
            cube_goal_pos[2] = 0.95
            action, action_noise = self.grasping_act(cube_pos, cube_goal_pos)
            if cube_pot_dist_xy < 0.02:
                action[-1] = -np.random.rand()
            else:
                action[-1] = np.random.rand()
        elif self.state == "POT_PICKING":
            action, action_noise = self.grasping_act(eef_pos, pot_handle_pos)
        elif self.state == "POT_LIFTING":
            action, action_noise = self.reaching_act(pot_pos, goal)

        action = np.clip(action, self.action_low * 0.9, self.action_high * 0.9)
        action[:3] += np.random.uniform(-action_noise, action_noise, 3)

        return action

    def act_randomly(self):
        return np.random.uniform(self.action_low, self.action_high)


class ScriptedPickAndPlace:
    def __init__(self, env, params):
        self.params = params

        self.task = params.env_params.env_name

        pick_place_params = params.scripted_policy_params.pick_place_params
        self.explore_eps = pick_place_params.explore_eps
        self.release_prob = pick_place_params.release_prob
        self.action_scale = 20.0

        self.action_low, self.action_high = params.action_spec

        global_low, global_high = params.normalization_range
        global_low = np.array(global_low)
        global_high = np.array(global_high)
        self.global_mean = (global_high + global_low) / 2
        self.global_scale = (global_high - global_low) / 2

    def reset(self, obs):
        if "Causal" in self.task:
            self.obj_to_pick_name = "mov0"
        elif self.task == "ToolUsePickTool":
            self.obj_to_pick_name = "tool"
        else:
            self.obj_to_pick_name = "cube"

        self.release = False
        self.release_step = 0
        self.goal = self.sample_goal(obs)

    def sample_goal(self, obs):
        if self.task in ["Causal", "ToolUse"]:
            goal = np.random.rand(3) * self.global_scale + self.global_mean
        elif self.task == "CausalStack":
            goal = obs["unmov0_pos"].copy() * self.global_scale + self.global_mean
            goal[-1] = 0.9
        elif self.task == "ToolUsePickPlace":
            goal = obs["pot_pos"].copy() * self.global_scale + self.global_mean
            goal[-1] = 0.95
        else:
            goal = obs["goal_pos"] * self.global_scale + self.global_mean
        return goal

    def act(self, obs):
        eef_pos = obs["robot0_eef_pos"] * self.global_scale + self.global_mean
        object_pos = obs[self.obj_to_pick_name + "_pos"] * self.global_scale + self.global_mean
        grasped = obs[self.obj_to_pick_name + "_grasped"]

        action = np.zeros_like(self.action_low)
        action[-1] = -np.random.rand()

        if np.random.rand() < self.explore_eps:
            action = np.random.uniform(-1, 1, 4)
        elif not grasped and self.task not in ["CausalReach", "ToolUseReach"]:
            self.release = False
            placement = object_pos - eef_pos
            xy_place, z_place = placement[:2], placement[-1]

            if np.linalg.norm(xy_place) >= 0.02:
                if eef_pos[2] < 0.9:
                    action[2] = 0.9 - eef_pos[2]
                else:
                    action[:2] = xy_place
                action[-1] = np.random.uniform(-1, 1)
            elif np.linalg.norm(placement) >= 0.01:
                action[:3] = placement
            else:
                action[:3] = placement
                action[-1] = np.random.rand()

            action[:3] *= self.action_scale
        else:
            # eef position
            placement = self.goal - eef_pos
            action[:3] = placement

            # gripper
            to_release = np.random.rand() < self.release_prob
            if self.task in ["CausalStack", "ToolUsePickPlace"] and np.linalg.norm(placement) < 0.02:
                to_release = True
            self.release = self.release or to_release
            if self.release:
                action[-1] = -np.random.rand()
            else:
                action[-1] = np.random.rand()

            action[:3] *= self.action_scale

        action_noise = 0.25
        action = np.clip(action, self.action_low, self.action_high)
        mask = np.abs(action) >= 2 * action_noise
        disturbed_action = np.clip(action, self.action_low + action_noise, self.action_high - action_noise) + \
                           np.random.uniform(action_noise, action_noise, 4)
        action = np.where(mask, disturbed_action, action)

        return action

    def act_randomly(self):
        return np.random.uniform(self.action_low, self.action_high)


class ScriptedUnlock:
    def __init__(self, env, params):
        self.env = env
        self.params = params
        self.action_dim = params.action_dim

    def reset(self, *args):
        pass

    def act(self, obs):
        return self.env.generate_action()

    def act_randomly(self,):
        return np.random.randint(self.action_dim)


class ScriptWrapper:
    def __init__(self, env, script_cls, params):
        self.num_envs = params.env_params.num_envs

        self.policies = [script_cls(env, params) for _ in range(self.num_envs)]
        self.policies_inited = False

    def reset(self, obs, i=0):
        if not self.policies_inited:
            self.policies_inited = True
            for i in range(self.num_envs):
                obs_i = {key: val[i] for key, val in obs.items()}
                self.policies[i].reset(obs_i)
        else:
            obs_i = {key: val[i] for key, val in obs.items()}
            self.policies[i].reset(obs_i)

    def act(self, obs):
        actions = []
        for i in range(self.num_envs):
            obs_i = {key: val[i] for key, val in obs.items()}
            actions.append(self.policies[i].act(obs_i))
        return np.array(actions)

    def act_randomly(self):
        return np.array([self.policies[0].act_randomly() for _ in range(self.num_envs)])


def get_scripted_policy(env, params):
    env_name = params.env_params.env_name

    if env_name == "Physical":
        script_cls = ScriptedPhysical
    elif env_name == "Chemical":
        script_cls = ScriptedChemical
    elif env_name == "Unlock":
        script_cls = ScriptedUnlock
    elif env_name == "ToolUseSeries":
        script_cls = ScriptedToolUseSeries
    elif "ToolUse" in env_name or "Causal" in env_name:
        script_cls = ScriptedPickAndPlace
    else:
        raise ValueError("Unknown env_name: {}".format(env_name))

    return ScriptWrapper(env, script_cls, params)
