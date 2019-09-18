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

