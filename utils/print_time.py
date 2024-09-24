import time


def print_(name: str):
    print("------------------------------------------------")
    print(name)
    # print current time in format: 2019-10-03 13:10:00
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    print("------------------------------------------------")
