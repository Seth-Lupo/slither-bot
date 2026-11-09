"""What the agent sees: the live game state -> a 64x64 heading-up picture (9 channels) + 14 numbers.

The page script reports every snake near us (head, angle, speed, thickness, body points) and nearby food. We draw it on a
20-unit world grid and sample a 64x64 view around our head, rotated so ahead is up, zoomed out as we grow - exactly the
view the policy was trained on in self-play. Channels: other bodies, other heads, head speed (forward, sideways), food,
outside the map, own body, other snakes' thickness, remains of dead snakes.
"""
import math, numpy as np

CELL, V0, VB, TURN, DT = 20.0, 176.0, 425.0, math.radians(240.0), 0.1
NB, NA, OBS, NCH, NSC = 32, 64, 64, 9, 14
REM_SZ = 13              # pellets this big are remains of dead snakes
SP2UPS = 30.4            # game speed units -> world units per second
SPAN = 2700              # world units around our head the page reports
L = 2 * int(SPAN / CELL) + 8
# Reward profile the policy is told to play under: (graze, remains, kill, death). Trained ranges: 0-0.5, 1-8, 0.5-8, 2-15.
PROFILES = {"hunter": (0.3, 8.0, 2.0, 15.0),     # best so far: loves remains, hates dying (length 7468, 52 kills in 22 min)
            "careful": (0.4, 3.0, 2.0, 15.0), "balanced": (0.25, 4.0, 4.0, 10.0), "aggressive": (0.1, 6.0, 8.0, 3.0)}

PAGE_JS = """(() => { if (window.__agent) return;
  let target = null, boost = 0;
  setInterval(() => { if (target === null) return;                    // hold the commanded heading between decisions
    window.xm = Math.cos(target) * 200; window.ym = Math.sin(target) * 200;
    if (typeof window.setAcceleration === 'function') window.setAcceleration(boost); }, 50);
  const length = (s) => (s && window.fpsls && window.fmlts) ? Math.floor(15 * (fpsls[s.sct] + s.fam / fmlts[s.sct] - 1) - 5) : 0;
  window.__agent = {
    act(d, b) { const s = window.slither; if (s) { target = s.ang + d; boost = b; } },
    read(span) {
      const s = window.slither, alive = !!(s && window.playing && !s.dead && window.dead_mtm === -1);
      if (!alive) { target = null; return { alive: false }; }
      const hx = s.xx, hy = s.yy, S2 = span * span, sn = [], fd = [];
      for (const o of (window.slithers || [])) { if (!o || o.dead) continue; const px = [], py = [];
        for (const q of (o.pts || [])) { if (!q || q.dying) continue; const dx = q.xx - hx, dy = q.yy - hy;
          if (dx * dx + dy * dy < S2) { px.push(Math.round(q.xx)); py.push(Math.round(q.yy)); } }
        const dx = o.xx - hx, dy = o.yy - hy; if (!px.length && dx * dx + dy * dy >= S2) continue;
        sn.push({ me: o === s, x: o.xx, y: o.yy, a: o.ang, sp: o.sp, sc: o.sc || 1, px, py }); }
      for (const f of (window.foods || [])) { if (!f) continue; const dx = f.xx - hx, dy = f.yy - hy;
        if (dx * dx + dy * dy < S2) fd.push(Math.round(f.xx), Math.round(f.yy), f.sz); }
      return { alive: true, length: length(s), w: { grd: window.grd || 16384, sn, fd, kills: s.kill_count || 0 } }; } }; })();"""

_o = (np.arange(OBS) + 0.5 - OBS / 2) * CELL
_FI, _RJ = np.meshgrid(-_o, _o, indexing="ij")
OF, ORR = _FI.flatten(), _RJ.flatten()


def scang_of(sc): return 0.13 + 0.87 * ((7 - sc) / 6) ** 2
def zoom_of(sc): return min(2.5, max(1.0, 1 + 0.3 * (sc - 1)))


def build(w, score, a1, a2, prof):
    """w: page world, score: our length, a1/a2: our last two actions, prof: reward profile -> obs [9,64,64], sca [14], me"""
    me = next(o for o in w["sn"] if o["me"])
    hx, hy, ang = me["x"], me["y"], me["a"]; sc = min(6.0, me["sc"])
    bx, by = math.floor(hx / CELL) - L // 2, math.floor(hy / CELL) - L // 2
    owner = np.zeros((L, L), np.int32); heads = np.zeros((L, L), np.int32); food = np.zeros((L, L), np.float32); rem = np.zeros((L, L), np.float32)
    n = len(w["sn"]); scs = np.zeros(n + 2, np.float32); vel = np.zeros((n + 2, 2), np.float32)
    for k, o in enumerate(w["sn"]):
        code = 1 if o["me"] else k + 2
        scs[code] = min(6.0, o["sc"]); v = o["sp"] * SP2UPS; vel[code] = (v * math.cos(o["a"]), v * math.sin(o["a"]))
        if o["px"]:
            gx = np.floor(np.asarray(o["px"]) / CELL).astype(np.int64) - bx; gy = np.floor(np.asarray(o["py"]) / CELL).astype(np.int64) - by
            m = (gx >= 0) & (gx < L) & (gy >= 0) & (gy < L); owner[gy[m], gx[m]] = code
        gx, gy = math.floor(o["x"] / CELL) - bx, math.floor(o["y"] / CELL) - by
        if 0 <= gx < L and 0 <= gy < L: owner[gy, gx] = code; heads[gy, gx] = code
    if w["fd"]:
        f = np.asarray(w["fd"], np.float32).reshape(-1, 3)
        gx = np.floor(f[:, 0] / CELL).astype(np.int64) - bx; gy = np.floor(f[:, 1] / CELL).astype(np.int64) - by
        m = (gx >= 0) & (gx < L) & (gy >= 0) & (gy < L)
        val = (f[:, 2] / 6.0) ** 2; big = f[:, 2] >= REM_SZ       # pellet value ~ (size/6)^2: normal 0.25..2.3, remains 5+
        np.add.at(food, (gy[m & ~big], gx[m & ~big]), val[m & ~big]); np.add.at(rem, (gy[m & big], gx[m & big]), val[m & big])
        np.minimum(food, 50.0, out=food); np.minimum(rem, 50.0, out=rem)
    c, s = math.cos(ang), math.sin(ang); z = zoom_of(sc)
    G, Rm = w["grd"], w["grd"] * 0.98
    ch = None
    for df, dr in ((-.25, -.25), (-.25, .25), (.25, -.25), (.25, .25)):
        of = (OF + df * CELL) * z; orr = (ORR + dr * CELL) * z
        x = hx + of * c - orr * s; y = hy + of * s + orr * c
        gx = np.floor(x / CELL).astype(np.int64) - bx; gy = np.floor(y / CELL).astype(np.int64) - by
        ok = (gx >= 0) & (gx < L) & (gy >= 0) & (gy < L); gx = gx.clip(0, L - 1); gy = gy.clip(0, L - 1)
        own = owner[gy, gx] * ok; hd = heads[gy, gx] * ok
        b0 = (own >= 2).astype(np.float32); m0 = (own == 1).astype(np.float32); z0 = scs[own] / 6 * b0
        h1 = (hd >= 2).astype(np.float32); hv = vel[hd]
        f1 = (hv[:, 0] * c + hv[:, 1] * s) / VB * h1; r1 = (-hv[:, 0] * s + hv[:, 1] * c) / VB * h1
        fo = np.log1p(food[gy, gx]) * ok / 2; ro = np.log1p(rem[gy, gx]) * ok / 2
        e1 = ((((x - G) ** 2 + (y - G) ** 2) > Rm ** 2) | ~ok).astype(np.float32)
        cur = [b0, h1, f1, r1, fo, e1, m0, z0, ro]
        if ch is None: ch = cur
        else:
            take = h1 > ch[1]; ch[2] = np.where(take, f1, ch[2]); ch[3] = np.where(take, r1, ch[3])
            for i in (0, 1, 4, 5, 6, 7, 8): ch[i] = np.maximum(ch[i], cur[i])
    obs = np.stack(ch).reshape(NCH, OBS, OBS).astype(np.float16)
    boost = me["sp"] > 10
    d = max(1e-3, math.hypot(hx - G, hy - G)); bear = math.atan2(G - hy, G - hx) - ang
    tf = lambda h: ((h % NB) - (NB - 1) / 2) / ((NB - 1) / 2)
    g, wr, wk, wd = prof
    sca = np.array([float(boost), min(1.0, me["sp"] * SP2UPS / VB), math.log(max(score, 1)) / 10, (Rm - d) / Rm, math.sin(bear), math.cos(bear),
                    tf(a1), float(a1 // NB), tf(a2), float(a2 // NB),
                    g / 0.5, math.log(wr) / math.log(8), math.log(wk / 0.5) / math.log(16), math.log(wd / 2) / math.log(7.5)], np.float32)
    return obs, sca, me


def turn_of(a, sc, gain=1.0):
    """action (0-63) -> (heading change over the next 0.1 s in radians, boost 0/1). 32 turn bins x {cruise, boost}."""
    frac = ((a % NB) - (NB - 1) / 2) / ((NB - 1) / 2)
    return frac * TURN * scang_of(min(6.0, sc)) * DT * gain, int(a // NB)


def view_rgb(o):
    """[9,64,64] -> [64,64,3] picture: bodies green, own body blue, heads white, food purple, remains gold, edge red."""
    o = o.astype(np.float32); rgb = np.zeros((OBS, OBS, 3), np.float32)
    rgb += o[4, ..., None].clip(0, 1) * np.array([90, 70, 140])
    rgb += o[8, ..., None].clip(0, 1) * np.array([200, 170, 40])
    rgb = np.where((o[5] > 0)[..., None], np.array([90, 30, 30]), rgb)
    rgb = np.where((o[6] > 0)[..., None], np.array([60, 110, 200]), rgb)
    b = 0.5 + 0.5 * o[7]
    rgb = np.where((o[0] > 0)[..., None], np.stack([b * 80, b * 220, b * 140], -1), rgb)
    rgb = np.where((o[1] > 0)[..., None], np.array([255, 255, 255]), rgb)
    rgb[OBS // 2 - 1:OBS // 2 + 1, OBS // 2 - 1:OBS // 2 + 1] = (255, 220, 0)
    return rgb.clip(0, 255).astype(np.uint8)
