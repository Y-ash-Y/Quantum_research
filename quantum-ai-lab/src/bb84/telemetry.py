"""Publicly observable simulated monitoring counts, never Eve's private labels.

The model treats sifted-bit errors as telemetry. A real protocol must account
for the cost of publicly revealing a parameter-estimation sample.
"""
from dataclasses import dataclass, fields

from bb84.simulator import _count


@dataclass(frozen=True)
class TelemetryCounts:
    z_sifted: int
    x_sifted: int
    z_errors: int
    x_errors: int
    signal_sent: int
    decoy_sent: int
    signal_detected: int
    decoy_detected: int

    def __post_init__(self):
        for field in fields(self):
            object.__setattr__(self, field.name, _count(getattr(self, field.name), field.name, 0))
        for observed, total in [('z_errors', 'z_sifted'), ('x_errors', 'x_sifted'),
                                ('signal_detected', 'signal_sent'), ('decoy_detected', 'decoy_sent')]:
            if getattr(self, observed) > getattr(self, total):
                raise ValueError(f'{observed} cannot exceed {total}')
        if self.z_sifted+self.x_sifted > self.signal_detected+self.decoy_detected:
            raise ValueError('Sifted bits cannot exceed detected pulses')

    def as_tuple(self):
        return tuple(getattr(self, f.name) for f in fields(self))

    def features(self):
        if not self.z_sifted or not self.x_sifted or not self.decoy_sent or not self.signal_detected:
            raise ValueError('Insufficient counts for the legacy four-feature telemetry')
        qz, qx = self.z_errors/self.z_sifted, self.x_errors/self.x_sifted
        signal = self.signal_detected/self.signal_sent
        return [(self.z_errors+self.x_errors)/(self.z_sifted+self.x_sifted),
                abs(qz-qx), signal, (self.decoy_detected/self.decoy_sent)/signal]


COUNT_FIELDS = tuple(f.name for f in fields(TelemetryCounts))
