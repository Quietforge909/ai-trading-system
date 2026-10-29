from __future__ import annotations

import base64
import hashlib
import hmac
import time
import urllib.parse
from decimal import Decimal, ROUND_DOWN
from typing import Any

import requests


LEGACY_ASSET_MAP = {
    "XXBT": "BTC",
    "XBT": "BTC",
    "XETH": "ETH",
    "XXRP": "XRP",
    "XLTC": "LTC",
    "XADA": "ADA",
    "XDOT": "DOT",
    "XXLM": "XLM",
    "ZUSD": "USD",
    "ZUSDT": "USDT",
}


def normalize_asset(asset: str) -> str:
    asset = (asset or "").replace(".d", "").upper()
    return LEGACY_ASSET_MAP.get(asset, asset)


def floor_decimal(value: float, decimals: int) -> str:
    quantum = Decimal("1") if decimals <= 0 else Decimal("1").scaleb(-decimals)
    floored = Decimal(str(value)).quantize(quantum, rounding=ROUND_DOWN)
    return format(floored, "f")


class KrakenClient:
    def __init__(self, api_key: str = "", api_secret: str = "", timeout: int = 30):
        self.api_key = api_key
        self.api_secret = api_secret
        self.timeout = timeout
        self.base_url = "https://api.kraken.com"

    def _sign(self, url_path: str, data: dict[str, Any]) -> str:
        post_data = urllib.parse.urlencode(data)
        encoded = (str(data["nonce"]) + post_data).encode()
        message = url_path.encode() + hashlib.sha256(encoded).digest()
        mac = hmac.new(base64.b64decode(self.api_secret), message, hashlib.sha512)
        return base64.b64encode(mac.digest()).decode()

    def public(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        response = requests.get(
            f"{self.base_url}/0/public/{method}",
            params=params or {},
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("error"):
            raise RuntimeError(payload["error"])
        return payload["result"]

    def private(self, method: str, data: dict[str, Any] | None = None) -> dict[str, Any]:
        if not self.api_key or not self.api_secret:
            raise RuntimeError("Kraken private API key/secret are missing")
        url_path = f"/0/private/{method}"
        payload = dict(data or {})
        payload["nonce"] = int(time.time() * 1000)
        headers = {
            "API-Key": self.api_key,
            "API-Sign": self._sign(url_path, payload),
        }
        response = requests.post(
            f"{self.base_url}{url_path}",
            data=payload,
            headers=headers,
            timeout=self.timeout,
        )
        response.raise_for_status()
        result = response.json()
        if result.get("error"):
            raise RuntimeError(f"Kraken private {method} failed: {result['error']}")
        return result["result"]

    def balance(self) -> dict[str, float]:
        raw = self.private("Balance")
        return {normalize_asset(k): float(v) for k, v in raw.items()}

    def open_orders(self) -> dict[str, Any]:
        return self.private("OpenOrders").get("open", {})

    def ticker(self, pairs: list[str] | None = None) -> dict[str, Any]:
        params = {"pair": ",".join(pairs)} if pairs else {}
        return self.public("Ticker", params)

    def asset_pairs(self) -> dict[str, Any]:
        return self.public("AssetPairs")

    def trade_volume(self, pairs: list[str]) -> dict[str, Any]:
        return self.private("TradeVolume", {"pair": ",".join(pairs)})

    def add_order(
        self,
        pair: str,
        side: str,
        ordertype: str,
        volume: str,
        price: str | None = None,
        validate: bool = False,
        client_order_id: str | None = None,
    ) -> dict[str, Any]:
        data: dict[str, Any] = {
            "pair": pair,
            "type": side,
            "ordertype": ordertype,
            "volume": volume,
        }
        if price is not None:
            data["price"] = price
        if validate:
            data["validate"] = "true"
        if client_order_id:
            data["cl_ord_id"] = client_order_id
        return self.private("AddOrder", data)

    def query_orders(self, txids: list[str]) -> dict[str, Any]:
        if not txids:
            return {}
        return self.private("QueryOrders", {"txid": ",".join(txids)})

    def cancel_order(self, txid: str) -> dict[str, Any]:
        return self.private("CancelOrder", {"txid": txid})
