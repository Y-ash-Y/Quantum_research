"""Parameterized-loss pulse path for the existing restricted IR/PNS policies.

Ideal local gates only. At nominal eta=.15, random draws and outputs match the
original attack simulator. Original sources remain unchanged for provenance.
"""
import numpy as np

from adversarial.eavesdropper_game import AttackPolicy, Session, _lut
from bb84.honest_nuisance import HonestSettings, MU_SIGNAL, MU_DECOY, DECOY_FRACTION
from bb84.simulator import _pns_forward_prob
from bb84.telemetry import TelemetryCounts


def simulate_loss_session(policy, settings, rng):
    if not isinstance(policy, AttackPolicy) or not isinstance(settings, HonestSettings):
        raise ValueError('Expected AttackPolicy and HonestSettings')
    if not isinstance(rng, np.random.Generator):
        raise ValueError('rng must be a local numpy Generator')
    if settings.gate_noise != 0:
        raise ValueError('Attack path supports ideal local gates only')
    n, eta = settings.n_pulses, settings.eta
    lut = _lut(settings.channel_noise)
    q_pns = _pns_forward_prob(MU_SIGNAL, eta) if policy.pns_frac else 0.0
    decoy = rng.random(n) < DECOY_FRACTION
    photons = rng.poisson(np.where(decoy, MU_DECOY, MU_SIGNAL))
    pns = rng.random(n) < policy.pns_frac
    ir = (~pns) & (rng.random(n) < policy.ir_rate)
    alice, alice_basis, bob_basis = [rng.integers(0, 2, n) for _ in range(3)]
    detected = np.zeros(n, bool)
    normal = ~pns
    detected[normal] = ((photons[normal] >= 1)
                        & (rng.random(normal.sum()) < 1-(1-eta)**photons[normal]))
    detected[pns] = ((photons[pns] >= 2) & (rng.random(pns.sum()) < q_pns))
    eve_basis = np.where(rng.random(n) < policy.basis_bias, 0, 1)
    transmitted, prep_basis = alice.copy(), alice_basis.copy()
    wrong = ir & (eve_basis != alice_basis)
    transmitted[wrong] = rng.integers(0, 2, wrong.sum())
    prep_basis[ir] = eve_basis[ir]
    sift = detected & (alice_basis == bob_basis)
    z, x = sift & (alice_basis == 0), sift & (alice_basis == 1)
    if (sift.sum() < 20 or not z.any() or not x.any() or not decoy.any()
            or decoy.all() or not detected[~decoy].any()):
        return None
    bob = (rng.random(n) < lut[transmitted*4+prep_basis*2+bob_basis]).astype(int)
    error = alice != bob
    known = pns | (ir & (eve_basis == alice_basis))
    counts = TelemetryCounts(int(z.sum()), int(x.sum()), int(error[z].sum()), int(error[x].sum()),
                             int((~decoy).sum()), int(decoy.sum()),
                             int(detected[~decoy].sum()), int(detected[decoy].sum()))
    return Session(counts.features(), int(known[sift].sum()), int(sift.sum()), counts)
