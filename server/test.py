import requests
import time

start = time.time()
res = requests.post("http://localhost:8000/api/move", json={"fen": "startpos", "moves": "e2e4", "timeMs": 2000})
print(res.json(), time.time() - start)
