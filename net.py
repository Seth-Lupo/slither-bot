"""The policy: an IMPALA-style ResNet over the 9x64x64 view, plus the 14 numbers -> 64 action logits and a value."""
import torch, torch.nn as nn, torch.nn.functional as F
from view import NA, NCH, NSC, OBS


class Res(nn.Module):
    def __init__(s, c):
        super().__init__(); s.a = nn.Conv2d(c, c, 3, padding=1); s.b = nn.Conv2d(c, c, 3, padding=1)
    def forward(s, x): return x + s.b(F.relu(s.a(F.relu(x))))


class Net(nn.Module):
    def __init__(s, ch=(32, 64, 64), hid=512):
        super().__init__(); L = []; c0 = NCH
        for c in ch: L += [nn.Conv2d(c0, c, 3, padding=1), nn.MaxPool2d(3, 2, 1), Res(c), Res(c)]; c0 = c
        s.cnn = nn.Sequential(*L)
        s.fc = nn.Linear(c0 * (OBS // 8) ** 2 + NSC, hid)
        s.pi = nn.Linear(hid, NA); s.v = nn.Linear(hid, 1)
    def forward(s, x, sca):
        h = F.relu(s.cnn(x)).flatten(1)
        h = F.relu(s.fc(torch.cat([h, sca], 1)))
        return s.pi(h), 10 * s.v(h).squeeze(1)


def load(path):
    net = Net().eval(); ck = torch.load(path, map_location="cpu", weights_only=False)
    net.load_state_dict(ck["sd"] if isinstance(ck, dict) and "sd" in ck else ck); return net
