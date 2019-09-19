```python
from tqdm import tqdm  # 进度条库
from time import sleep

# method01
for i in tqdm(range(100)):
    sleep(0.1)

# method02
with tqdm(total=100) as pbar:
    for i in range(10):
        sleep(0.2)
        pbar.update(10)
```



Q：每次循环时，进度条新建一行。

A：可能是手动终止进程导致tqdm没有完全退出导致的，或者关掉程序重新运行。

```python
try:
    with tqdm(...) as t:
        for i in t:
            ...
except KeyboardInterrupt:
    t.close()
    raise
t.close()
```

