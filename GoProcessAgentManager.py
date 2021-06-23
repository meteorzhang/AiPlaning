# AgentManagerment, batch
import sys
from configparser import ConfigParser
from GoConfig import GoConfig
from GoSharedStatus import GoSharedStatus
from GoProcessAgent import GoProcessAgent
from GoAgentAction import GoAgentAction
import numpy as np


class GoProcessAgentManger:
    def __init__(self, prediction_q, training_q, episode_log_q, episode_count):
        self.cfg = GoConfig.AGENT_CONFIG_FILE
        self.prediction_q = prediction_q
        self.training_q = training_q
        self.episode_log_q = episode_log_q
        self.current_agent_num = 0
        self.max_agent_num = 0
        self.environments = []
        self.agents = []
        self.actions = GoAgentAction()
        self.id = 0
        self.share_evs = {}
        self.load_cfg()
        self.episode_count = episode_count

    def get_show_status(self):
        # if GoConfig.VISUALIZE and int(self.id / len(self.environments)) == 0:
        if GoConfig.VISUALIZE and (int(self.id) == 0 or int(self.id) == 30 or int(self.id) == 56):
            return True
        elif GoConfig.PLAY_MODE and (int(self.id) == 0):
            return True
        else:
            return False

    def enable_disable_components(self):
        print('======================>>>>>>>>>>>>.')
        temp_current_agent_num = 0
        for agent in self.agents:
            sh_ev = self.share_evs[agent['env_name']]
            while agent['cur_num'] < agent['allow_num'] and sh_ev is not None:
                    print('+++++++ agent[env_name]=%s agent[cur_num]=%d,agent[allow_num]=%d'%(agent['env_name'],agent['cur_num'],agent['allow_num']))

                    enable_show = self.get_show_status()
                    a = GoProcessAgent(self.id, sh_ev, self.actions, self.prediction_q, self.training_q,
                                       self.episode_log_q, enable_show, self.episode_count)
                    a.exit_flag = False
                    agent['agents'].append(a)
                    a.start()
                    agent['cur_num'] = agent['cur_num'] + 1
                    self.id = self.id + 1
                    print('====================')
            while agent['cur_num'] > agent['allow_num']:
                    print('------- agent[env_name]=%s agent[cur_num]=%d,agent[allow_num]=%d'%(agent['env_name'],agent['cur_num'],agent['allow_num']))
                    # a= agent['agent'][-1]
                    a = agent['agents'].pop()
                    a.exit_flag = True
                    a.join()
                    agent['cur_num'] = agent['cur_num'] - 1
            temp_current_agent_num = temp_current_agent_num+agent['cur_num']
        self.current_agent_num = temp_current_agent_num

    def random_walk(self):
        for agent in self.agents:
            direct = np.random.randint(3, size=1)-1
            agent['allow_num'] = max(1, agent['allow_num'] - direct)
        pass

    def enable_adjust(self):
        for agent in self.agents:
            if agent['allow_num'] != agent['cur_num']:
                return True
        return False

    def get_current_agent_num(self):
        return self.current_agent_num

    def terminate_agent(self):
        for agent in reversed(self.agents):
            for i in agent['agents']:
                i.exit_flag = 1
                i.terminate()
                # i.join()
                self.current_agent_num -= 1
            # agent['agents'].exit_flag.value = True
            # agent['agents'].terminate()
            # print("self.agents", self.agents)
            self.agents.pop()

    def random_delete_agent(self):
        pass

    def random_add_agent(self):
        pass

    def start_all_agent(self):
        pass

    def get_agent_by_id(self, id=-1):
        rs = None
        if id == -1:
            rs = None
        for ags in self.agents:
            for ag in ags['agents']:
                if ag is not None and ag.id == id:
                    rs = ag
                    break
        return rs

    def load_cfg(self):
        print('load_cfg')
        config = ConfigParser()
        config.read(self.cfg)
        # print(config.sections())
        a = config.get("Environments", "name")
        a = a.replace(' ', '')
        self.environments = a.split(',')
        print('self.environments', self.environments)

        for str in self.environments:
            # Agent
            agent = dict()
            nums = config.get(str, 'AgentNums')
            ev_path = "../../" + config.get(str, 'Path')
            # agent_w = config.get(str, 'AgentWidth')
            # agent_h = config.get(str, 'AgentHeight')
            agent['env_name'] = str
            agent['max_num'] = int(nums)
            # agent['w'] = int(agent_w)
            # agent['h'] = int(agent_h)
            agent['path'] = ev_path
            agent['cur_num'] = 0
            self.max_agent_num = self.max_agent_num + agent['max_num']
            # Environment
            share_ev = GoSharedStatus()
            share_ev.agents_dynamic_status.clear()
            share_ev.name = str
            # agent['share_ev']=GoSharedStatus()
            share_ev.ev_path = ev_path
            share_ev.target_range = config.get(str, 'TargetRange')
            # share_ev.one_channel_map_pic_str = share_ev.ev_path + '/init_gnuplot_map.png'
            share_ev.roads_gunplot_cfg_url = share_ev.ev_path+'/road_map.txt'
            # share_ev.allow_area_gunplot_cfg_url = share_ev.ev_path+'/tongxing_grid.txt'
            # share_ev.map_pic_str = share_ev.ev_path + '/init_road.png'  # init.png 三通道叠加形成的图片，map.png是三通道一起显示的图片
            # share_ev.map_width = float(config.get(str, 'MapWidth'))
            # share_ev.map_height = float(config.get(str, 'MapHeight'))
            # share_ev.init_map(share_ev.map_width, share_ev.map_height, GoConfig.MAP_RESOLUTION)
            share_ev.road_node_Nos, share_ev.road_node_info, share_ev.road_lines, share_ev.road_lines_num, share_ev.node_edges = \
                share_ev.init_road_network(share_ev.roads_gunplot_cfg_url)

            # agent['share_ev']=GoSharedStatus()
            # agent['share_ev'].ev_path = ev_path
            # agent['share_ev'].one_channel_map_pic_str = agent['share_ev'].ev_path + '/map.jpg'
            # agent['share_ev'].roads_gunplot_cfg_url = agent['share_ev'].ev_path+'/road_txt.txt'
            # agent['share_ev'].allow_area_gunplot_cfg_url = agent['share_ev'].ev_path+'/brook_js_t.txt'
            # agent['share_ev'].map_pic_str = agent['share_ev'].ev_path + '/init.png'
            # agent['share_ev'].init_map()

            agent['allow_num'] = agent['max_num']
            agent['agents'] = list()
            self.agents.append(agent)
            self.share_evs.update({agent['env_name']: share_ev})
