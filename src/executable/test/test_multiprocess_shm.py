import torch
import multiprocessing as mp


def worker_main(worker_id, shm):
    torch.cuda.set_device(worker_id)
    # torch.cuda.cudart().cudaProfilerStart()
    a = torch.randn((100, 100)).cuda()
    b = torch.randn((100, 100)).cuda()
    c = a * b
    # torch.cuda.cudart().cudaProfilerStop()
    print(c)
    return c



shm_tensor = torch.zeros((1000, 32)).share_memory_()

workers = []
for worker_id in range(4):
    p = mp.Process(target=worker_main,
        args=(worker_id, shm_tensor))
    p.start()
    print(f"Worker {worker_id} pid={p.pid}")
    workers.append(p)

for each in workers:
    each.join()
