"""
Websocket server speaking the BEHAVIOR eval protocol (OmniGibson's network_utils).

Protocol, as the eval's client expects it:
  GET /healthz         -> HTTP 200 "OK"; the client polls this before connecting
  on connect           -> one msgpack frame with metadata ({} is fine)
  {"reset": True}      -> policy.reset(), and NO reply (a reply would desync the client)
  observation dict     -> {"action": (1, 23) float32, "server_timing": {...}}
  error                -> traceback as a text frame (the client raises), then close 1011
Messages are msgpack, with numpy arrays encoded as in openpi's msgpack_numpy.
"""

import asyncio
import functools
import http
import logging
import time
import traceback
from pathlib import Path

import msgpack
import numpy as np
from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosed
from websockets.frames import CloseCode

logger = logging.getLogger(__name__)

LOG_EVERY = 100  # log inference time once per this many actions


def pack_array(obj):
    """
    msgpack hook: numpy arrays and scalars -> plain dicts (openpi's exact format).
    """
    if isinstance(obj, (np.ndarray, np.generic)) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"unsupported dtype: {obj.dtype}")
    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }
    if isinstance(obj, np.generic):
        return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
    return obj


def unpack_array(obj):
    """
    msgpack hook: the dicts made by pack_array -> numpy arrays and scalars.
    Arrays are read-only views of the received bytes.
    """
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


packb = functools.partial(msgpack.packb, default=pack_array)
unpackb = functools.partial(msgpack.unpackb, object_hook=unpack_array)


def health_check(connection, request):
    """
    Answer GET /healthz with 200 before any websocket upgrade; other paths continue.
    """
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None


class WebsocketPolicyServer:
    """
    Serves a policy with act(obs) -> action and reset() (e.g. B1KPolicy).
    """

    def __init__(
        self,
        policy,
        host: str = "0.0.0.0",
        port: int = 8000,
        metadata: dict | None = None,
        save_obs_path: Path | None = None,
    ):
        self.policy = policy
        self.host = host
        self.port = port
        self.metadata = metadata or {}
        self.save_obs_path = save_obs_path  # save the first observation received, then None

    def serve_forever(self) -> None:
        asyncio.run(self._run())

    async def _run(self) -> None:
        async with serve(
            self._handler,
            self.host,
            self.port,
            compression=None,  # images are already compact; compression only costs time
            max_size=None,  # observations (3 images) exceed the default 1 MB limit
            process_request=health_check,
        ) as server:
            logger.info("Serving on ws://%s:%d", self.host, self.port)
            await server.serve_forever()

    async def _handler(self, connection) -> None:
        """
        One connected client: metadata once, then request/reply until it disconnects.
        """
        logger.info("Client connected from %s", connection.remote_address)
        await connection.send(packb(self.metadata))

        infer_times = []  # ms per action since the last log line
        while True:
            try:
                obs = unpackb(await connection.recv())
                if "reset" in obs:
                    self.policy.reset()
                    continue  # no reply: the client does not read one

                if self.save_obs_path is not None:
                    self._save_obs(obs)
                
                start = time.monotonic()
                # Blocks the event loop while the model runs: fine, the eval is one client.
                action = self.policy.act(obs)
                infer_ms = (time.monotonic() - start) * 1000
                await connection.send(packb({"action": action, "server_timing": {"infer_ms": infer_ms}}))

                infer_times.append(infer_ms)
                if len(infer_times) == LOG_EVERY:
                    logger.info(
                        "infer_ms over the last %d actions: mean %.1f, max %.1f",
                        LOG_EVERY,
                        np.mean(infer_times),
                        np.max(infer_times),
                    )
                    infer_times = []

            except ConnectionClosed:
                logger.info("Client %s disconnected", connection.remote_address)
                break
            except Exception:
                logger.exception("Error while serving a request")
                # A text frame makes the client raise right away; a silent close would make
                # it reconnect and resend the same observation forever.
                await connection.send(traceback.format_exc())
                await connection.close(code=CloseCode.INTERNAL_ERROR, reason="Internal server error")
                break

    def _save_obs(self, obs: dict) -> None:
        """
        Save the first real observation (for tests and offline checks), only once.
        """
        self.save_obs_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(self.save_obs_path, **obs)
        logger.info("Saved observation to %s", self.save_obs_path)
        self.save_obs_path = None