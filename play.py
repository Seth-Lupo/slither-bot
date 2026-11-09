"""Watch a trained slither.io agent play in your browser, with a live panel of what it sees and what it wants to do.

  python play.py                       # loops forever; Ctrl-C to quit
  python play.py --profile careful     # or --w GRAZE REMAINS KILL DEATH, e.g. --w 0.3 8 2 15
  python play.py --record clips/       # save every game (with the panel) as a .webm
"""
import argparse, asyncio, base64, json, os, time
import numpy as np, torch
import view as V
from net import load

NAMES = ["other bodies", "other heads", "head speed fwd", "head speed side", "food", "map edge", "own body", "other size", "remains"]
OVERLAY_JS = """(() => { if (window.__overlay) return;
  const W = 420, cv = document.createElement('canvas'); cv.width = W; cv.height = 720;
  cv.style.cssText = 'position:fixed;right:0;top:0;z-index:99999;background:rgba(8,10,16,.88);pointer-events:none';
  document.body.appendChild(cv); const g = cv.getContext('2d'), names = %s;
  const img = (b64, ch, x, y, s) => { const raw = atob(b64), n = 64 * 64, id = g.createImageData(64, 64);
    for (let i = 0; i < n; i++) { for (let c = 0; c < 3; c++) id.data[4 * i + c] = raw.charCodeAt(ch === 3 ? 3 * i + c : i); id.data[4 * i + 3] = 255; }
    const t = document.createElement('canvas'); t.width = 64; t.height = 64; t.getContext('2d').putImageData(id, 0, 0);
    g.imageSmoothingEnabled = false; g.drawImage(t, x, y, s, s); };
  window.__overlay = (d) => { g.clearRect(0, 0, W, 720); g.fillStyle = '#cbd5e1'; g.font = '12px sans-serif';
    // action dial: up = straight ahead, left/right = turn; inner bars = cruise, outer (orange) = boost
    const cx = W / 2, cy = 182, r0 = 30, L = 128, p = d.p; let mx = 1e-9; for (let t = 0; t < 32; t++) mx = Math.max(mx, p[t] + p[32 + t]);
    g.fillText('action probabilities (up = straight, sides = turn; orange = boost)', 12, 18);
    g.strokeStyle = '#334155'; g.beginPath(); g.arc(cx, cy, r0, Math.PI, 2 * Math.PI); g.stroke();
    for (let t = 0; t < 32; t++) { const f = (t - 15.5) / 15.5, a = -Math.PI / 2 + f * 1.4, c = Math.cos(a), s = Math.sin(a);
      const l0 = L * p[t] / mx, l1 = L * p[32 + t] / mx;
      g.lineWidth = 5; g.strokeStyle = '#60a5fa'; g.beginPath(); g.moveTo(cx + r0 * c, cy + r0 * s); g.lineTo(cx + (r0 + l0) * c, cy + (r0 + l0) * s); g.stroke();
      g.strokeStyle = '#f59e0b'; g.beginPath(); g.moveTo(cx + (r0 + l0) * c, cy + (r0 + l0) * s); g.lineTo(cx + (r0 + l0 + l1) * c, cy + (r0 + l0 + l1) * s); g.stroke();
      if (t === d.a %% 32) { g.fillStyle = d.a >= 32 ? '#f59e0b' : '#60a5fa'; g.beginPath(); g.arc(cx + (r0 - 10) * c, cy + (r0 - 10) * s, 5, 0, 7); g.fill(); } }
    g.lineWidth = 1; g.fillStyle = '#cbd5e1';
    g.fillText(`boost ${(100 * d.pb).toFixed(0)}%%   value ${d.v.toFixed(2)}   profile ${d.prof}`, 12, cy + 26);
    // what the policy sees
    g.fillText('policy view (heading up)', 12, 232); img(d.rgb, 3, 12, 240, 150);
    g.fillText(d.info, 175, 252);
    for (let k = 0; k < 9; k++) { const x = 12 + (k %% 3) * 134, y = 410 + Math.floor(k / 3) * 102;
      g.fillText(names[k], x, y - 4); img(d.ch[k], 0, x, y, 84); } }; })();""" % json.dumps(NAMES)


async def game(browser, net, prof, a):
    """Play one game in a fresh page (so each recording is one game). Returns (seconds, best length, kills)."""
    vp = dict(width=1600, height=760)
    ctx = await browser.new_context(viewport=vp, **(dict(record_video_dir=a.record, record_video_size=vp) if a.record else {}))
    page = await ctx.new_page(); t0, best, kills, stamp = time.time(), 0, 0, time.strftime("%Y%m%d_%H%M%S")
    try:
        await page.goto("http://slither.io/", wait_until="commit", timeout=60000)
        await page.wait_for_selector("#playh .btnt", state="visible", timeout=60000)
        await page.evaluate(V.PAGE_JS); await page.evaluate(OVERLAY_JS)
        await page.fill("#nick", a.nick); await page.click("#playh .btnt")
        rd = {"alive": False}
        for _ in range(450):                                       # wait for the game to start
            await asyncio.sleep(0.1); rd = await page.evaluate(f"() => window.__agent.read({V.SPAN})")
            if rd["alive"]: break
        a1 = a2 = V.NB // 2; k0 = rd.get("w", {}).get("kills", 0); n = 0; tn = t0 = time.time()
        while rd["alive"] and any(o["me"] for o in rd["w"]["sn"]):
            length = max(rd["length"], 1); best = max(best, length); kills = rd["w"]["kills"] - k0
            ob, sca, me = V.build(rd["w"], length, a1, a2, prof)                   # sense
            with torch.no_grad(): lg, val = net(torch.from_numpy(ob[None]).float(), torch.from_numpy(sca[None]))
            p = torch.softmax(lg[0], -1)
            act = int(p.argmax()) if a.greedy else int(torch.multinomial(p, 1))     # decide
            turn, boost = V.turn_of(act, me["sc"])
            await page.evaluate(f"() => window.__agent.act({turn}, {boost})")       # act
            if n % 2 == 0:                                                          # show
                ch = [base64.b64encode((np.clip(np.abs(ob[k].astype(np.float32)), 0, 1) * 255).astype(np.uint8).tobytes()).decode() for k in range(9)]
                await page.evaluate("d => window.__overlay(d)", dict(
                    p=p.tolist(), a=act, pb=float(p[32:].sum()), v=float(val[0]), prof=a.profile_name, ch=ch,
                    rgb=base64.b64encode(V.view_rgb(ob).tobytes()).decode(), info=f"length {length}  kills {kills}"))
            a2, a1 = a1, act; n += 1
            tn += 0.1; await asyncio.sleep(max(0.0, tn - time.time())); tn = max(tn, time.time() - 0.05)   # 10 decisions/s
            rd = await page.evaluate(f"() => window.__agent.read({V.SPAN})")
        await asyncio.sleep(1.5)
        return time.time() - t0, best, kills
    finally:
        video = page.video if a.record else None
        await ctx.close()
        if video and best:
            os.replace(await video.path(), os.path.join(a.record, f"{stamp}_len{best}_kills{kills}_{int(time.time() - t0)}s.webm"))


async def main(a):
    from playwright.async_api import async_playwright
    net = load(a.policy)
    prof = tuple(a.w) if a.w else V.PROFILES[a.profile]
    a.profile_name = "custom " + ",".join(f"{x:g}" for x in a.w) if a.w else a.profile
    if a.record: os.makedirs(a.record, exist_ok=True)
    print(f"profile {a.profile_name} {prof}, {'greedy' if a.greedy else 'sampling'}", flush=True)
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=False, args=["--mute-audio", "--disable-background-timer-throttling",
                                                                 "--disable-renderer-backgrounding", "--disable-backgrounding-occluded-windows"])
        while True:
            try:
                secs, best, kills = await game(browser, net, prof, a)
                print(f"game over: {secs:.0f} s, best length {best}, kills {kills}", flush=True)
            except Exception as e: print(f"retrying after {type(e).__name__}: {str(e)[:80]}", flush=True); await asyncio.sleep(3)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--policy", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "policy.pt"))
    p.add_argument("--profile", default="hunter", choices=list(V.PROFILES))
    p.add_argument("--w", type=float, nargs=4, metavar=("GRAZE", "REMAINS", "KILL", "DEATH"))
    p.add_argument("--sample", dest="greedy", action="store_false", help="sample actions instead of taking the most likely")
    p.add_argument("--record", default="", help="directory to save each game as .webm")
    p.add_argument("--nick", default="rl-agent")
    try: asyncio.run(main(p.parse_args()))
    except KeyboardInterrupt: pass
