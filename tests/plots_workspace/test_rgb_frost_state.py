from types import SimpleNamespace

import numpy as np
import pandas as pd


def test_state_matching_is_causal_and_missing_features_remain_blank(tmp_path):
    from plots.rgb_frost_state import match_states
    from image_models.dinov2_features import DINO_COLUMNS

    points = pd.DataFrame({'cycle_name': ['one', 'two'], 'experiment_id': ['e', 'e'],
                           'selected_time': pd.to_datetime(['2026-01-01 00:01:00'] * 2)})
    metadata = pd.DataFrame({'cycle_name': ['one', 'one', 'two'], 'camera_role': ['front'] * 3,
        'file_name': ['past.jpg', 'future.jpg', 'absent.jpg'],
        'image_time': pd.to_datetime(['2026-01-01 00:00:50', '2026-01-01 00:01:01', '2026-01-01 00:00:50'])})
    cache = metadata.iloc[:2][['cycle_name', 'camera_role', 'file_name']].copy()
    cache = pd.concat([cache, pd.DataFrame(np.ones((2,384)),columns=DINO_COLUMNS)],axis=1)
    cache.to_parquet(tmp_path/'one.parquet')
    loader = SimpleNamespace(load_image_metadata=lambda n: metadata.loc[metadata.cycle_name.eq(n)],
                             dataset_root=tmp_path)
    result = match_states(points, loader, [tmp_path])
    assert result.file_name.tolist() == ['past.jpg', 'absent.jpg']
    assert result.loc[0,DINO_COLUMNS].eq(1).all()
    assert result.loc[1,DINO_COLUMNS].isna().all()


def test_common_model_cycles_excludes_missing_decisions():
    from image_models.core import common_cycle_metrics
    metrics = pd.DataFrame(dict(cycle_name=['a','a','b','b'], seed=[0]*4,
        version=['old','future','old','future'], selected_cop=[2,2,2,np.nan], rb_cop=[1]*4))
    assert set(common_cycle_metrics(metrics).cycle_name) == {'a'}


def test_fixed_optimal_reference_is_not_refit_to_model_states():
    from plots.rgb_frost_state import state_coordinates
    reference = pd.DataFrame({'dinov2_000':[1.,1.], 'dinov2_001':[0.,0.1]})
    states = pd.DataFrame({'dinov2_000':[0.], 'dinov2_001':[1.]})
    result = state_coordinates(states, reference)
    assert result.cosine_to_centroid.iloc[0] == 1
    assert result.cosine_to_reference.iloc[0] < .1


def test_optimal_cohort_is_not_limited_by_old_model_outputs(tmp_path, monkeypatch):
    import dataset_tools
    import image_models.inputs
    import plots.rgb_frost_state as module
    cycles = pd.DataFrame(dict(cycle_name=['frost_cycle_000200'],experiment_id=['new_exp']))
    loader = SimpleNamespace(list_cycles=lambda **kwargs: cycles)
    monkeypatch.setattr(dataset_tools, 'DatasetLoader', lambda dataset: loader)
    monkeypatch.setattr(image_models.inputs, 'reliable_rgb_cohort', lambda *a, **k: cycles)
    monkeypatch.setattr(module, 'match_states', lambda points, *a, **k: points.assign(
        dinov2_000=1.,dinov2_001=0.,local_image_available=True))
    cycles.assign(t_star='2026-09-01 10:00:00').to_csv(tmp_path/'cycle_comparison.csv',index=False)
    result = module.collect_states(tmp_path/'no_old_model_run', point='optimal', cohort='regular',
                                   decisions=tmp_path, feature_caches=[tmp_path])
    assert result.cycle_name.tolist() == ['frost_cycle_000200']
