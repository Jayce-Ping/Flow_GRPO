import asyncio
from datetime import datetime
from concurrent import futures

class test:
    def __init__(self, max_concurrent: int):
        self.semaphore = asyncio.Semaphore(max_concurrent)

    def __call__(self, n):
        res = asyncio.run(self.run(n))
        return res

    async def run(self, n):
        tasks = [self.__async_call() for _ in range(n)]
        return await asyncio.gather(*tasks)

    async def __async_call(self):
        async with self.semaphore:
            time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"{time} start")
            await asyncio.sleep(1)
            print("done")
        
        return time
    


def get_fn():

    def _fn(m, n):
        t = test(m)
        res = t(n)
        return res
    
    return _fn

if __name__ == "__main__":
    fn = get_fn()
    executor = futures.ThreadPoolExecutor(max_workers=8)
    res = executor.submit(fn, 8, 10)
    
    res = res.result()
    print(res)