from multiprocessing import Process, Queue, Pipe
import numpy as np

def main():

    def f1(q,x): q.put((x**2)/4,)
    def f2(q,x): q.put(-(x+5)**-3,)
    def f3(q,x): q.put(np.log(max(x,1)))
    def f4(q,x): q.put(np.sin(x))

    functions = [f1, f2, f3, f4]
    jobs = []
    queue = Queue()

    # Initialize jobs and pass queue
    for f in functions:
        p = Process(target=f, args=(queue, 1))
        jobs.append(p)
        p.start()
    
    # Join jobs and print results
    for proc in jobs:
        print(queue.get())
        proc.join()

if __name__ == '__main__':

    main()
    