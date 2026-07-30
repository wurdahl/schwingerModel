"""
Gauge configuration generation driven by a TOML input file:

    python run_sim.py inputs/example.toml

Runs nChains independent HMC chains in parallel, discards the per-chain burn-in,
merges the thermalized configurations, and pickles the result. The number of
HMC substeps is either given explicitly (numSubSteps) or tuned automatically by
scanning pilot chains until the target metropolis acceptance rate is reached.
See inputs/example.toml for all recognized fields.
"""

import os
import sys
import pickle
import tomllib

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

import schwingerModel as sim


def loadInput(path):
    with open(path, 'rb') as f:
        raw = tomllib.load(f)

    cfg = {
        #required
        'outputFile': raw['outputFile'],
        'beta':  raw['physics']['beta'],
        'fMass': raw['physics']['fMass'],
        'dimx':  raw['lattice']['dimx'],
        'dimt':  raw['lattice']['dimt'],
        'targetConfigs': raw['run']['targetConfigs'],
        'burnIn':        raw['run']['burnIn'],
        #optional
        'aSpacing': raw['physics'].get('aSpacing', 1.0),
        'cgRtol':   raw['run'].get('cgRtol', 1e-5),
        'randSeed': raw['run'].get('randSeed', 0),
        'nCores':   raw['run'].get('nCores', os.cpu_count()),
        'tunneling': raw['run'].get('tunneling', False),
    }
    cfg['nChains'] = raw['run'].get('nChains', cfg['nCores'])

    sub = raw.get('substeps', {})
    cfg['numSubSteps']      = sub.get('numSubSteps')        # None -> tune automatically
    cfg['targetAcceptance'] = sub.get('targetAcceptance', 0.8)
    cfg['pilotSteps']       = sub.get('pilotSteps', 100)
    cfg['startSubSteps']    = sub.get('startSubSteps', 5)
    cfg['maxSubSteps']      = sub.get('maxSubSteps', 500)

    return cfg


def pilotAcceptance(cfg, numSubSteps):
    """Acceptance rate of a short pilot chain, measured on its second half
    so the cold-start transient does not bias the estimate."""
    model = sim.schwingerModel(
        metroSteps=cfg['pilotSteps'],
        beta=cfg['beta'], dimx=cfg['dimx'], dimt=cfg['dimt'],
        aSpacing=cfg['aSpacing'], fMass=cfg['fMass'], cgRtol=cfg['cgRtol'],
        randSeed=cfg['randSeed'], numSubSteps=numSubSteps,
        tqdmPosition=1,   #keep the pilot bar below the tuning report lines
    )
    return np.mean(model.acceptHistory[cfg['pilotSteps']//2:])


def tuneSubSteps(cfg):
    """
    Scans substep counts until the target acceptance is reached: doubling to
    bracket, then bisecting to the smallest passing count (within ~10%).
    """
    target = cfg['targetAcceptance']

    lo = None                       # last failing count
    hi = cfg['startSubSteps']
    while hi <= cfg['maxSubSteps']:
        acc = pilotAcceptance(cfg, hi)
        tqdm.write(f"  substeps {hi:4d}: acceptance {acc:.2f}")
        if acc >= target:
            break
        lo, hi = hi, hi*2
    else:
        raise RuntimeError(f"acceptance {target} not reached by {cfg['maxSubSteps']} substeps")

    if lo is None:
        return hi

    #bisect the bracket [lo (fail), hi (pass)] down to ~10% granularity
    while hi - lo > max(1, hi//10):
        mid = (lo + hi)//2
        acc = pilotAcceptance(cfg, mid)
        tqdm.write(f"  substeps {mid:4d}: acceptance {acc:.2f}")
        if acc >= target:
            hi = mid
        else:
            lo = mid

    return hi


def runChain(cfg, chainIndex, numSubSteps, stepsPerChain):
    model = sim.schwingerModel(
        metroSteps=cfg['burnIn'] + stepsPerChain,
        beta=cfg['beta'], dimx=cfg['dimx'], dimt=cfg['dimt'],
        aSpacing=cfg['aSpacing'], fMass=cfg['fMass'], cgRtol=cfg['cgRtol'],
        randSeed=cfg['randSeed'] + chainIndex, tqdmPosition=chainIndex,
        numSubSteps=numSubSteps,tunneling=cfg['tunneling']
    )
    #chain 0 carries the model object; the rest only their thermalized history
    if chainIndex == 0:
        return model
    return (model.linkHistory[cfg['burnIn']:],
            model.acceptHistory[cfg['burnIn']:],
            model.tunnelAcceptance[cfg['burnIn']:])


def main(inputPath):
    cfg = loadInput(inputPath)

    if cfg['numSubSteps'] is not None:
        numSubSteps = cfg['numSubSteps']
        print(f"using {numSubSteps} substeps (given in input file)")
    else:
        print(f"tuning substeps for acceptance >= {cfg['targetAcceptance']}:")
        numSubSteps = tuneSubSteps(cfg)
        print(f"using {numSubSteps} substeps")

    stepsPerChain = -(-cfg['targetConfigs'] // cfg['nChains'])   # ceil division

    print(f"running {cfg['nChains']} chains x ({cfg['burnIn']} burn-in + {stepsPerChain} configs) "
          f"on {cfg['nCores']} cores")

    results = Parallel(n_jobs=cfg['nCores'])(
        delayed(runChain)(cfg, i, numSubSteps, stepsPerChain) for i in range(cfg['nChains']))

    base = results[0]
    nKeep = cfg['targetConfigs']
    merged = np.concatenate([base.linkHistory[cfg['burnIn']:]] + [r[0] for r in results[1:]])[:nKeep]
    mergedAccept = np.concatenate([base.acceptHistory[cfg['burnIn']:]] + [r[1] for r in results[1:]])[:nKeep]
    mergedTunnel = np.concatenate([base.tunnelAcceptance[cfg['burnIn']:]] + [r[2] for r in results[1:]])[:nKeep]

    base.linkHistory = merged
    base.metroSteps = len(merged)
    #per-config accept flags survive the merge; only autocorrelation across chain boundaries is meaningless
    base.acceptHistory = mergedAccept
    base.tunnelAcceptance = mergedTunnel

    if cfg['tunneling']:
        print(f"tunnel step acceptance: {mergedTunnel.mean():.3f}")

    os.makedirs(os.path.dirname(cfg['outputFile']) or '.', exist_ok=True)
    with open(cfg['outputFile'], 'wb') as f:
        pickle.dump(base, f)

    print(f"saved {len(merged)} configurations to {cfg['outputFile']}")


if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit("usage: python run_sim.py <input.toml>")
    main(sys.argv[1])
