import os
import psutil
import time
import math

# import setup

max_x = 10000
max_y = 10000


def main():
    print("PID :")
    print (os.getpid())

    print("CREANDO LISTAS ...")
    a = [[0 for x in range(max_y)] for y in range(max_x)]
    b = [[0 for x in range(max_y)] for y in range(max_x)]
    c = [[0 for x in range(max_y)] for y in range(max_x)]
    print("Acabado ! ")

    # print(psutil.cpu_percent(interval=1))
    print(psutil.cpu_percent(interval=1, percpu=True))

    # print (psutil.cpu_times())
    # scputimes(user=17411.7, nice=77.99, system=3797.02, idle=51266.57, iowait=732.58, irq=0.01, softirq=142.43, steal=0.0, guest=0.0, guest_nice=0.0)

    psutil.virtual_memory()

    print ("INICIALIZANDO ...")
    for i in range(max_x):
        for j in range(max_y):
            if i == 0:
                a[i][j] = 1.1
                b[i][j] = 3.2
            else:
                a[i][j] = 5.5
                b[i][j] = 8.7

    print("Fin Init !")

    # print (psutil.cpu_times())

    # print(psutil.cpu_percent(interval=1))
    print(psutil.cpu_percent(interval=1, percpu=True))

    print("MULTIPLICACION DE MATRICES ... ")
    for i in range(max_x):
        for j in range(max_y):
            c[i][j] = a[i][j] * b[i][j]

    print()
    print("Fin !")
    # print (psutil.cpu_times())

    # print(psutil.cpu_percent(interval=1))
    print(psutil.cpu_percent(interval=1, percpu=True))


main()
