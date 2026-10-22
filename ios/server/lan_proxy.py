"""Token-checked WebSocket relay from the Mac's LAN address to the local Realtime server.

Usage: python lan_proxy.py <lan_ip> [port]   (token read from $S2S_LAN_TOKEN_FILE)

The speech-to-speech server has no authentication and stays bound to 127.0.0.1. Phones on
the same Wi-Fi reach this relay instead; it accepts only `/v1/realtime?token=<secret>` and
answers 401 to everything else before any byte reaches the pipeline (which holds a single
session slot). Traffic is plain ws:// on the LAN, so the token is only as private as the
Wi-Fi itself.
"""

from __future__ import annotations

import asyncio
import hmac
import logging
import os
import sys
from http import HTTPStatus
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from websockets.asyncio.client import connect
from websockets.asyncio.server import ServerConnection, serve
from websockets.exceptions import ConnectionClosed
from websockets.http11 import Request, Response

UPSTREAM = os.environ.get("S2S_UPSTREAM", "ws://127.0.0.1:8765/v1/realtime")
TOKEN = Path(os.environ.get("S2S_LAN_TOKEN_FILE", "lan_token")).read_text().strip()

logger = logging.getLogger("lan_proxy")


def check_token(connection: ServerConnection, request: Request) -> Response | None:
    url = urlsplit(request.path)
    given = parse_qs(url.query).get("token", [""])[0]
    if url.path != "/v1/realtime" or not hmac.compare_digest(given.encode(), TOKEN.encode()):
        # Never log request.path: it carries the token.
        logger.warning("rejected connection from %s", connection.remote_address[0])
        return connection.respond(HTTPStatus.UNAUTHORIZED, "unauthorized\n")
    return None


async def pump(source, sink) -> None:
    async for message in source:
        await sink.send(message)


async def relay(client: ServerConnection) -> None:
    peer = client.remote_address[0]
    logger.info("session from %s", peer)
    try:
        async with connect(UPSTREAM, max_size=None) as upstream:
            tasks = [asyncio.create_task(pump(client, upstream)), asyncio.create_task(pump(upstream, client))]
            _, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in pending:
                task.cancel()
    except (ConnectionClosed, OSError) as exc:
        logger.info("session from %s ended: %s", peer, exc)
    logger.info("session from %s closed", peer)


async def main(host: str, port: int) -> None:
    async with serve(relay, host, port, process_request=check_token, max_size=None):
        logger.info("relaying ws://%s:%d/v1/realtime -> %s", host, port, UPSTREAM)
        await asyncio.get_running_loop().create_future()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    asyncio.run(main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 8766))
