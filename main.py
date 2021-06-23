from Navigator import Navigator1
from threading import Thread

navigation = Navigator1()

#初始化
def init(path,agv_list):
    print("navi start init")
    print("navi init ing...")
    init_thread = Navi_thread(navigation, agv_list,path)
    print("navi init success!")
    init_thread.start()
    init_thread.wait()


#设置目标点
def set_goal(agv_id,stationcode,x,y,goal_x,goal_y):
    print("set_goal",agv_id,stationcode,x,y,goal_x,goal_y)
    navigation.set_goal(agv_id,stationcode,x,y,goal_x,goal_y)

#规划结果返回
def step(agv_id,step_id,action_x,action_y,status):
    print("step",agv_id,step_id,action_x,action_y,status)
    all_status = navigation.get_all_state()
    return all_status


#小车坐标更新
def update(agv_list):
    agv_list = agv_list
    navigation.updateAGV(agv_list)


class Navi_thread(Thread):
    def __init__(self, navi, car_list, map_name):
        super().__init__()
        self.navigation = navi
        self.car_list = car_list
        self.map_name = map_name

    def run(self):

        self.navigation.init_data(self.map_name, self.car_list)
