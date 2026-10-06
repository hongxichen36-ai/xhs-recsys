"""
简单 TTL 内存缓存
接口和 Redis 类似，方便以后替换成 Redis。
"""
import time
import json
import threading


class TTLCache:
    def __init__(self, default_ttl=300):
        self.default_ttl = default_ttl
        self._store = {}
        self._lock = threading.Lock()

    def get(self, key):
        with self._lock:
            item = self._store.get(key)
            if not item:
                return None
            value, expire_at = item
            if time.time() > expire_at:
                del self._store[key]
                return None
            return value

    def set(self, key, value, ttl=None):
        ttl = ttl or self.default_ttl
        with self._lock:
            self._store[key] = (value, time.time() + ttl)

    def clear(self):
        with self._lock:
            self._store.clear()


cache = TTLCache(default_ttl=300)