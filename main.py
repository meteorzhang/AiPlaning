from navi.Navigator import Navigator1
from threading import Thread

def init(path,agv_list):
    print("navi start init")
    navigation = Navigator1()  # 导航模块
    print("navi init ing...")
    init_thread = Navi_thread(navigation, agv_list,path)
    print("navi init success!")
    init_thread.start()
    init_thread.wait()


#设置目标点
def set_goal(agv_id,stationcode,x,y,goal_x,goal_y):
    print("set_goal",agv_id,stationcode,x,y,goal_x,goal_y)

#规划结果返回
def step(agv_id,step_id,action_x,action_y,status):
    print("step",agv_id,step_id,action_x,action_y,status)

#小车坐标更新
def update(agv_list):
    print("update",agv_list)


class Navi_thread(Thread):
    def __init__(self, navi, car_list, map_name):
        super().__init__()
        self.navigation = navi
        self.car_list = car_list
        self.map_name = map_name

    def run(self):
        self.navigation.init("navi/final/tongxing_t.txt")
        self.navigation.init_map_m(self.map_name)
        print("len(car_list): ", len(self.car_list))
        self.navigation.init_data(self.map_name, self.car_list)
