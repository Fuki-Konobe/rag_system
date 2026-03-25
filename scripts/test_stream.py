import requests

url = "http://localhost:8000/ask_stream"
params = {"question": "Shift_JISはどこで使われている？"}

# stream=True で接続を維持したまま受信する
with requests.post(url, params=params, stream=True) as r:
    for chunk in r.iter_content(chunk_size=None, decode_unicode=True):
        if chunk:
            print(chunk, end="", flush=True)