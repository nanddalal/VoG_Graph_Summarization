import os
import signal
import multiprocessing as mp

def timeout_handler():
    print "ran into time limit"

def inf_loop():
    while True:
        for i in xrange(99999):
            j = i*i

signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(15)

worker_processes = mp.Pool(processes=None)

worker_processes.apply_async(inf_loop)
worker_processes.apply_async(inf_loop)
worker_processes.apply_async(inf_loop)
worker_processes.apply_async(inf_loop)
worker_processes.apply_async(inf_loop)
worker_processes.apply_async(inf_loop)
worker_processes.apply_async(inf_loop)
worker_processes.apply_async(inf_loop)

worker_processes.close()
worker_processes.join()
