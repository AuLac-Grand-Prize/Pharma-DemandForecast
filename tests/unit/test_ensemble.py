from demandforecast.ensemble.stacker import EnsembleStacker, EnsembleWeights


def test_weights_normalize() -> None:
    w = EnsembleWeights(2, 2, 2).normalize()
    assert abs(w.prophet + w.lstm + w.tft - 1.0) < 1e-9


def test_combine_uniform() -> None:
    s = EnsembleStacker(EnsembleWeights(1, 1, 1))
    assert abs(s.combine(10, 20, 30) - 20.0) < 1e-9


def test_combine_weighted() -> None:
    s = EnsembleStacker(EnsembleWeights(0.6, 0.3, 0.1))
    assert abs(s.combine(10, 20, 30) - (6 + 6 + 3)) < 1e-9
