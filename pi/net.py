"""Wire format for client.py <-> server.py: one TCP connection, length-prefixed messages.

A message is an 8-byte length, a JSON header, then the raw bytes of the numpy arrays the header lists (dtype and shape
per array). Nothing is ever unpickled: the client does not trust the server, and the server does not trust the client.
Only plain numeric dtypes are accepted."""
import hashlib, hmac, json, os, secrets, socket, struct, numpy as np

DTYPES = {"uint8", "int8", "uint16", "int16", "uint32", "int32", "int64", "float32", "float64"}
MAX_ARRAY = 1 << 34            # 16 GiB per array: refuse anything larger


class Conn:
    def __init__(self, sock):
        self.s = sock; self.sent = self.recvd = 0
        self.s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        for o in (socket.SO_SNDBUF, socket.SO_RCVBUF): self.s.setsockopt(socket.SOL_SOCKET, o, 16 << 20)

    def send(self, header, arrays=()):
        arrays = [np.ascontiguousarray(a) for a in arrays]
        header = dict(header, arrays=[[a.dtype.name, list(a.shape)] for a in arrays])
        h = json.dumps(header).encode()
        parts = [struct.pack("<Q", len(h)), h] + [memoryview(a).cast("B") for a in arrays if a.nbytes]
        for p in parts: self.s.sendall(p)
        self.sent += sum(len(p) for p in parts)

    def _exact(self, n):
        buf = bytearray(n); view = memoryview(buf); got = 0
        while got < n:
            k = self.s.recv_into(view[got:], min(n - got, 1 << 24))
            if k == 0: raise ConnectionError("connection closed")
            got += k
        self.recvd += n
        return buf

    def recv(self, limit=MAX_ARRAY):
        n = struct.unpack("<Q", self._exact(8))[0]
        if n > min(1 << 26, limit): raise ValueError("header too large")
        header = json.loads(self._exact(n)); out = []
        for dt, shape in header.pop("arrays", []):
            if dt not in DTYPES or not all(isinstance(x, int) and x >= 0 for x in shape): raise ValueError("bad array")
            size = int(np.prod(shape, dtype=np.int64)) * np.dtype(dt).itemsize
            if size > limit: raise ValueError("array too large")
            out.append(np.frombuffer(self._exact(size), dtype=dt).reshape(shape))
        return header, out


# ---------------------------------------------------------------- password (pre-shared token)
def token_from_env():
    """The password: environment variable PI_TOKEN, or the first line of the file named by PI_TOKEN_FILE (default
    ~/.config/pi/token if it exists). Never a command-line argument, which other users and shell history can see."""
    t = os.environ.get("PI_TOKEN", "")
    f = os.environ.get("PI_TOKEN_FILE") or os.path.expanduser("~/.config/pi/token")
    if not t and "PI_TOKEN" not in os.environ and os.path.exists(f):
        if os.stat(f).st_mode & 0o077: raise SystemExit(f"{f} can be read by other users: run chmod 600 {f}")
        t = open(f).readline().strip()
    if t and len(t) < 16: raise SystemExit("PI_TOKEN is too short: use at least 16 characters (see README)")
    return t

def _mac(token, nonce, role):
    return hmac.new(token.encode(), role + nonce, hashlib.sha256).hexdigest()

def server_auth(c, token):
    """Challenge-response: the password itself never crosses the network, and a recorded login can't be replayed.
    Each side also proves it knows the password, so the client can tell it reached the real server."""
    nonce = secrets.token_bytes(32)
    c.send(dict(op="challenge", nonce=nonce.hex()))
    h, _ = c.recv(limit=4096)
    cn = bytes.fromhex(h.get("nonce", ""))
    if h.get("op") != "auth" or len(cn) != 32 or not hmac.compare_digest(h.get("mac", ""), _mac(token, nonce + cn, b"client")):
        c.send(dict(op="denied")); raise PermissionError("wrong password")
    c.send(dict(op="welcome", mac=_mac(token, nonce + cn, b"server")))

def client_auth(c, token):
    h, _ = c.recv(limit=4096)
    if h.get("op") != "challenge": raise PermissionError("server did not ask for a password (server.py needs PI_TOKEN set)")
    nonce, cn = bytes.fromhex(h["nonce"]), secrets.token_bytes(32)
    c.send(dict(op="auth", nonce=cn.hex(), mac=_mac(token, nonce + cn, b"client")))
    h, _ = c.recv(limit=4096)
    if h.get("op") != "welcome": raise PermissionError("wrong password")
    if not hmac.compare_digest(h.get("mac", ""), _mac(token, nonce + cn, b"server")): raise PermissionError("server does not know the password")


def pack_bits(y, k):
    """Values in [0, 2^k), any shape -> uint8 bytes, k bits each (groups of g values fill whole bytes)."""
    y = np.asarray(y, dtype=np.uint64).reshape(-1)
    g = next(g for g in (1, 2, 4, 8) if g * k % 8 == 0); assert g * k <= 64, k
    nb = g * k // 8
    y = np.concatenate([y, np.zeros(-len(y) % g, np.uint64)]).reshape(-1, g)
    v = np.zeros(len(y), np.uint64)
    for j in range(g): v |= y[:, j] << np.uint64(j * k)
    return np.ascontiguousarray(v.astype("<u8").view(np.uint8).reshape(-1, 8)[:, :nb]).reshape(-1)

def unpack_bits(b, k, count):
    g = next(g for g in (1, 2, 4, 8) if g * k % 8 == 0); nb = g * k // 8
    if g == 1:                                                 # whole bytes per value (24, 56, ...): no regrouping
        b = b.reshape(-1, nb)[:count]; v = b[:, 0].astype(np.int64)
        for i in range(1, nb): v |= b[:, i].astype(np.int64) << (8 * i)
        return v
    b = b.reshape(-1, nb); w = np.zeros((len(b), 8), np.uint8); w[:, :nb] = b
    v = w.reshape(-1).view("<u8"); m = np.uint64((1 << k) - 1)
    return np.stack([(v >> np.uint64(j * k)) & m for j in range(g)], 1).reshape(-1)[:count].astype(np.int64)
