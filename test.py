#from Navigator import Navigator1
from threading import Thread

#navigation = Navigator1()

#初始化
def init(path,agv_list):
    agv_list = agv_list
    print("path:",path)
    print("agv_list:",agv_list)

    #init_thread = Navi_thread(navigation, agv_list,path)

    t = "init: "+path+", "+str(agv_list)
    write_txt(t)
    print("navi init success!")
    #init_thread.start()
    #init_thread.wait()


#设置目标点
def set_goal(agv_id,stationcode,x,y,goal_x,goal_y):
    print("set_goal",agv_id,stationcode,x,y,goal_x,goal_y)
    t = "set_goal,"+str(agv_id)+","+str(stationcode)+","+str(x)+","+str(y)+","+str(goal_x)+","+str(goal_y)
    write_txt(t)
    #navigation.set_goal(agv_id,stationcode,x,y,goal_x,goal_y)

#规划结果返回
def step():
    all_status = [1,2,10.2,33.2,0]
    print("step",all_status)
    #all_status = navigation.get_all_state()
    return all_status


#小车坐标更新
def update(agv_list):
    agv_list = agv_list
    print(agv_list)

    write_txt(agv_list)
    #navigation.updateAGV(agv_list)


# class Navi_thread(Thread):
#     def __init__(self, navi, car_list, map_name):
#         super().__init__()
#         self.navigation = navi
#         self.car_list = car_list
#         self.map_name = map_name
#
#     def run(self):
#         self.navigation.init()
#         self.navigation.init_data(self.map_name, self.car_list)
#

def write_txt(txt):
    with open("test.txt", "a") as f:
        f.write(str(txt)+"\n")
#
# if __name__ == '__main__':
#     init("aaa.ini",str([1,2,3,4]))
    #set_goal(111,222,10.1,20.22,30.44,50.99)