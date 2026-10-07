"""
检查指定用户的推荐类目分布（避免 PowerShell 乱码）
"""
import requests
from collections import Counter

API = "http://localhost:8000"

for uid in [0, 1, 100, 500, 1000, 5000]:
    try:
        r = requests.post(f"{API}/recommend",
                          json={"user_id": uid, "k": 10}, timeout=5)
        data = r.json()
        cats = [item["category"] for item in data["items"]]
        sources = [item["source"] for item in data["items"]]
        print(f"\n用户 {uid}:")
        print(f"  类目分布: {dict(Counter(cats))}")
        print(f"  来源分布: {dict(Counter(sources))}")
        print(f"  延迟: {data['latency_ms']:.1f} ms")
    except Exception as e:
        print(f"用户 {uid}: 失败 {e}")