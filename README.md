# slither-bot

A reinforcement-learning agent that plays [slither.io](http://slither.io), and a script to watch it play live in your browser.
It hunts down the remains of dead snakes, coils around victims, and in our best run reached length 7,468 with 52 kills
in 22 minutes.

## Run it

```bash
pip install -r requirements.txt
playwright install chromium
python play.py                     # opens a browser, plays forever (Ctrl-C to quit)
python play.py --record clips/     # also save every game as a video
python play.py --profile careful   # or set your own reward weights: --w GRAZE REMAINS KILL DEATH
```

The panel on the right shows, live, the action dial (32 steering directions; blue = cruise, orange = boost), the
policy's 64x64 view, and its 9 input channels.

## How it was made (broad strokes)

1. **Simulator.** A slither.io-like world written in PyTorch that runs entirely on the GPU: about a million snakes at
   once in ~10,000 arenas, with physics measured from the real game (speeds, turn rates, growth, control lag).
2. **Self-play.** PPO on 8 GPUs, with the agent playing against itself, older copies of itself, and scripted bots.
   Several billion decisions.
3. **Reward conditioning.** Each snake gets a random reward profile (how much it values grazing, eating dead snakes'
   remains, kills, and avoiding death), and the profile is part of its input. One network learns the whole range of
   play styles, and you choose the style when you run it (`--profile` / `--w`).
4. **Live fine-tuning.** The self-play policy then kept learning (PPO) in 180 real slither.io games running at once.

**What the agent sees:** a 64x64 picture around its head, rotated so ahead is up and zooming out as it grows (other
bodies, heads and their speeds, food, remains, the map edge, its own body), plus 14 numbers (speed, length, distance
to the edge, its last two moves, and its reward profile). It decides 10 times a second: 32 turn amounts × boost on/off.

## Files

| | |
|---|---|
| `play.py` | browser driver, sense → decide → act loop, live panel, recording |
| `view.py` | game state → the policy's view (and the in-page script that reads the game and steers) |
| `net.py` | the policy network (IMPALA-style ResNet, 2.5M parameters) |
| `policy.pt` | trained weights |

Trained on one 8×H200 node; watching runs fine on a laptop CPU. Please don't run many copies at once on public servers.
