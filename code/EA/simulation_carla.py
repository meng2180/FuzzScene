import sys
import os
from multiprocessing import Process
import time

mutation_name = sys.argv[1]
print(mutation_name)


def work1():
    os.system("timeout 110 ../../../CARLA_0.9.13/CarlaUE4.sh -carla-streaming-port=0")


def work2():
    os.system(
        "timeout 100 python3 ../scenario_runner-0.9.13/scenario_runner.py --openscenario " + "../seed_pool/" + mutation_name + " --sync --reloadWorld")


def work3():
    os.system("timeout 85 python3 ../scenario_runner-0.9.13/manual_control.py -a")


def work4():
    os.system("timeout 85 pkill -9 python")


def timetask(times):
    time.sleep(times)
    print(time.localtime())


def works():
    proc_record = []
    p1 = Process(target=work1, args=())
    p1.start()
    print("*********************************")
    print("Carla Server Started")
    proc_record.append(p1)
    time.sleep(5)
    p2 = Process(target=work2, args=())
    p2.start()
    print("*********************************")
    print("OpenScenario Simulation Started")
    proc_record.append(p2)
    time.sleep(10)
    p3 = Process(target=work3, args=())
    p3.start()
    print("*********************************")
    print("Collecting Data")
    proc_record.append(p3)


if __name__ == '__main__':
    works()
