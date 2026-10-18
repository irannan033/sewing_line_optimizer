import numpy as np
from src.data import generate_garment_data, clean_garment_data
from src.forecast import ProductionParticleFilter
from src.workforce_optimizer import assign_operators


def test_data():
    df = clean_garment_data(generate_garment_data(200))
    assert df["actual_productivity"].between(0.2, 1.2).all()


def test_assignment_is_valid():
    rng = np.random.default_rng(0)
    P = rng.random((6, 4))
    X = assign_operators(P, np.ones_like(P, dtype=bool))
    assert X.sum(axis=0).tolist() == [1] * 4 and (X.sum(axis=1) <= 1).all()


def test_particle_filter_tracks_rate():
    pf = ProductionParticleFilter(init_rate=0.05)
    for _ in range(120):
        pf.step(np.random.poisson(0.2), 1.0)
    assert 0.1 < pf.rate() < 0.35
