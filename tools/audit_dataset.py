import json, glob, os, sys, hashlib, math
import numpy as np
root = sys.argv[1]
rows = []
hashes = {}
for f in sorted(glob.glob(os.path.join(root, 'data/landmarks/*/*.json'))):
    d = json.load(open(f))
    for c in d['clips']:
        fr = c['frames']
        n = len(fr)
        ts = np.array([x['t'] for x in fr])
        L = np.array([x['landmarks'] for x in fr], dtype=float) if n else np.zeros((0, 21, 3))
        dt = np.diff(ts) if n > 1 else np.array([0])
        fps = (n - 1) / (ts[-1] - ts[0]) if n > 1 and ts[-1] > ts[0] else 0
        pc = L[:, [0, 5, 9, 13, 17], :2].mean(1) if n else np.zeros((0, 2))
        span = np.ptp(pc, axis=0) if n else np.zeros(2)
        step = np.linalg.norm(np.diff(pc, axis=0), axis=1) if n > 1 else np.array([0])
        # hand size: wrist to middle mcp
        size = np.linalg.norm(L[:, 0, :2] - L[:, 9, :2], axis=1) if n else np.array([0])
        # bone lengths sanity: thumb tip to pinky mcp etc
        h = hashlib.md5(json.dumps([x['landmarks'] for x in fr]).encode()).hexdigest()
        hashes.setdefault(h, []).append((d['user'], d['gesture'], c['clip']))
        rows.append(dict(user=d['user'], g=d['gesture'], clip=c['clip'], bg=c['background'], li=c['lighting'],
                         di=c['distance'], dur=c['duration'], n=n, tlast=ts[-1] if n else 0, fps=fps,
                         maxdt=dt.max(), mindt=dt.min(), nan=int(np.isnan(L).sum()),
                         xmin=L[..., 0].min() if n else 0, xmax=L[..., 0].max() if n else 0,
                         ymin=L[..., 1].min() if n else 0, ymax=L[..., 1].max() if n else 0,
                         spanx=span[0], spany=span[1], maxstep=step.max(), msize=size.mean() if n else 0,
                         minsize=size.min() if n else 0, maxsize=size.max() if n else 0,
                         zmin=L[..., 2].min() if n else 0, zmax=L[..., 2].max() if n else 0,
                         w=c.get('image_width'), hh=c.get('image_height'),
                         dupframes=int((np.abs(np.diff(L, axis=0)).sum((1, 2)) == 0).sum()) if n > 1 else 0))
import collections
print('clips', len(rows))
print('per user/gesture counts:')
cnt = collections.Counter((r['user'], r['g']) for r in rows)
users = sorted({r['user'] for r in rows}); gs = sorted({r['g'] for r in rows})
print('user ' + ' '.join(f'{g[:9]:>9}' for g in gs))
for u in users:
    print(f'{u:5}' + ' '.join(f'{cnt.get((u,g),0):>9}' for g in gs))
print('\nmetadata combos per user:')
for u in users:
    print(u, dict(collections.Counter((r['bg'], r['li'], r['di']) for r in rows if r['user'] == u)))
print('\nimage sizes', collections.Counter((r['w'], r['hh']) for r in rows))
print('duration field', collections.Counter((r['g'], r['dur']) for r in rows))
print('\nclip table:')
print('user gesture      clip bg        li     di    dur   n  tlast  fps  maxdt  nan  spanx spany maxstep  size(min/mean/max)   x[min,max] y[min,max] zrange dup')
for r in rows:
    print(f"{r['user']:4} {r['g']:12} {r['clip']:>3} {r['bg']:9} {r['li']:6} {r['di']:5} {r['dur']:>4} {r['n']:>3} {r['tlast']:6.2f} {r['fps']:4.1f} {r['maxdt']:5.2f} {r['nan']:>3} "
          f"{r['spanx']:5.2f} {r['spany']:5.2f} {r['maxstep']:6.3f}  {r['minsize']:.2f}/{r['msize']:.2f}/{r['maxsize']:.2f}  [{r['xmin']:.2f},{r['xmax']:.2f}] [{r['ymin']:.2f},{r['ymax']:.2f}] [{r['zmin']:.2f},{r['zmax']:.2f}] {r['dupframes']}")
print('\nDUPLICATE clip contents:')
for h, v in hashes.items():
    if len(v) > 1: print(v)
