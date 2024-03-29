import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from scalable_mappo_lagr.envs.safety_ma_mujoco.safety_multiagent_mujoco import mujoco_env
import os
import mujoco_py as mjp
from gym import error, spaces

class CoupledHalfCheetah(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, **kwargs):
        agent_conf = kwargs.get("agent_conf")
        n_agents = int(agent_conf.split("x")[0])
        n_segs_per_agents = int(agent_conf.split("x")[1])
        self.n_segs = n_agents * n_segs_per_agents
        mujoco_env.MujocoEnv.__init__(self, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets', 'coupled_half_cheetah.xml'), 5)
        utils.EzPickle.__init__(self)

    def step(self, action):
        xposbefore1 = self.get_body_com("torso")[0]
        xposbefore2 = self.get_body_com("torso2")[0]
        xposbefore = (xposbefore1 + xposbefore2)/2.0

        # ADDED
        t = self.data.time
        wall_act = .02 * np.sin(t / 3) ** 2 - .004
        mjp.functions.mj_rnePostConstraint(self.sim.model,
                                           self.sim.data)  #### calc contacts, this is a mujoco py version mismatch issue with mujoco200
        action_p_wall = np.concatenate((np.squeeze(action), [wall_act]))
        self.do_simulation(action_p_wall, self.frame_skip)

        xposafter1 = self.get_body_com("torso")[0]
        xposafter2 = self.get_body_com("torso2")[0]
        xposafter =(xposafter1 + xposafter2)/2.0

        yposafter1 = self.get_body_com("torso")[1]
        yposafter2 = self.get_body_com("torso2")[1]
        yposafter =(yposafter1 + yposafter2)/2.0
        
        forward_reward = (xposafter - xposbefore) / self.dt

        ctrl_reward = -0.5 * np.square(action).sum()
        contact_reward = -0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 1.0
        # print("forward_reward ",forward_reward )
        # print("ctrl_reward",ctrl_reward)
        # print("contact_reward ",contact_reward )
        reward = forward_reward + ctrl_reward + contact_reward + survive_reward

        # ADDED
        # wallpos = self.data.get_geom_xpos("obj_geom")[0]
        # print("wallpos",wallpos)
        y_wallpos1 = self.data.get_geom_xpos("wall1")[1]
        y_wallpos2 = self.data.get_geom_xpos("wall2")[1]
        wallpos = [y_wallpos1, y_wallpos2]

        # ywall = np.array([-5, 5])
        if xposafter < 20:
            y_walldist = yposafter - xposafter * np.tan(30 / 360 * 2 * np.pi) + wallpos
        elif xposafter > 20 and xposafter < 60:
            y_walldist = yposafter + (xposafter - 40) * np.tan(30 / 360 * 2 * np.pi) - wallpos
        elif xposafter > 60 and xposafter < 100:
            y_walldist = yposafter - (xposafter - 80) * np.tan(30 / 360 * 2 * np.pi) + wallpos
        else:
            y_walldist = yposafter - 20 * np.tan(30 / 360 * 2 * np.pi) + wallpos

        obj_cost = (abs(y_walldist) < 2).any() * 1.0
        # if obj_cost > 0:
        #     self.model.geom_rgba[9] = [1.0, 0, 0, 1.0]
        # else:
        #     self.model.geom_rgba[9] = [1.0, 0.5, 0.5, .8]
        
        # state = self.state_vector()
        # notdone = np.isfinite(state).all()
        # qpos = self.sim.data.qpos
        # done = bool((qpos[2] < 1.0) or (qpos[2] > 2.0))

        # # done = not notdone
        # done_cost = done * 1.0

        cost = np.clip(obj_cost, 0, 1)


        # wallvel = self.data.get_body_xvelp("obj1")[0]
        # xdist = np.abs(wallpos - xposafter1)  #+ np.abs(wallpos - xposafter2) #+ (wallpos1 - xposafter1)  + (wallpos1 - xposafter2)
        # obj_cost = 0 # or int(np.abs(wallpos1 - xposafter2) < 5) or int(np.abs(wallpos1 - xposafter2) < 5)\
        # #
        # if int(np.abs(wallpos - xposafter1) < 2) or int(np.abs(wallpos - xposafter2) < 2) \
        #         or int(np.abs(y_wallpos1 - yposafter1) < 2) or int(np.abs(y_wallpos2 - yposafter2) < 2):
        #     obj_cost = 1

        # # obj_cost = int(np.abs(xdist) < 5)

        ob = self._get_obs()
        # print("ob",ob)
        done = False
        return ob, reward, done, dict(
            reward_forward=forward_reward,
            reward_ctrl=ctrl_reward,
            reward_contact=contact_reward,
            reward_survive=survive_reward,
            cost_obj=obj_cost,
            # cost_done=done_cost,
            cost=cost,
        )

    def _get_obs(self):

        #AADED
        wallvel = self.data.get_body_xvelp("obj1")[0]
        wall_f = .02 * np.sin(self.data.time / 3) ** 2 - .004
        xdist = (self.data.get_geom_xpos("obj_geom")[0] - self.sim.data.qpos[1]) / 10

        # return np.concatenate([
        #     self.sim.data.qpos.flat[2:],
        #     self.sim.data.qvel.flat[1:],
        #     [wallvel],
        #     [wall_f],
        #     np.clip([xdist], -5, 5),
        # ])

        return np.concatenate([
            self.sim.data.qpos.flat[1: self.n_segs+1],
            self.sim.data.qvel.flat[1: self.n_segs+1],
            self.sim.data.qpos.flat[self.n_segs+1: ],
            self.sim.data.qvel.flat[self.n_segs+1: ],
            [wallvel],
            [wall_f],
            # np.clip([xdist], -5, 5),
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5

    def get_env_info(self):
        return {"episode_limit": self.episode_limit}

    def _set_action_space(self):
        bounds = self.model.actuator_ctrlrange.copy().astype(np.float32)
        low, high = bounds.T
        low, high = low[:-1], high[:-1]
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        return self.action_space
