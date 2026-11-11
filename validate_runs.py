"""Verify fresh run-level metrics and physical writes against their recorded traces."""
from concurrent.futures import ProcessPoolExecutor
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from memristive_cpg.metrics import collective_metrics, time_rmse
from scripts.runtime import DATA, VALIDATION, WORK

TOL = 2e-11
HULLS = None


def load_hulls():
    global HULLS
    src = pd.read_csv(DATA/'processed/control_inputs/support_halfspaces.csv', float_precision='round_trip')
    HULLS = {key: g[['a','b','c']].to_numpy() for key,g in src.groupby(['branch','delay_us','event'])}


def margin(branch, delay, event, before, after):
    h = HULLS[branch,delay,event]
    return float(np.max(h[:,:2]@np.array([before,after])+h[:,2]))



def save_conductance_history(folder, initial, actions):
    """Materialize piecewise-constant states from initial G and complete action records."""
    g = {int(item['device']):item['initial_G_mS'] for item in initial}
    rows = []

    def snapshot(time, operation, device=None, valid=True):
        rows.append(dict(time_s=time, operation=operation, device=device,
                         used_for_integration=valid, **{f'G_{k:02d}_mS':value for k,value in sorted(g.items())}))

    snapshot(0.,'prepared')
    for row in actions.itertuples():
        device = int(row.device)
        if row.kind=='ideal_coordinate_change':
            g[device] = row.mean_G_mS
            snapshot(row.time_s,row.kind,device)
        else:
            g[device] = row.after_G_mS
            snapshot(row.time_s,row.kind,device,bool(row.support_valid and row.frequency_valid))
    pd.DataFrame(rows).to_csv(folder/'conductance_states.csv.gz',index=False,float_format='%.17g',
                              compression={'method':'gzip','mtime':0})
    return len(rows)

def audit(run):
    suite, folder = run
    p = json.loads((folder/'config.json').read_text())
    m = json.loads((folder/'metrics.json').read_text())
    parameters = p['parameters']
    initial = json.loads((folder/'initialization.json').read_text())
    n = int(m.get('N',5))
    epsilon = TOL
    cols = ['time_s','q_target','q_m','target_Hz','motor_estimated_Hz','motor_true_Hz','R','Ephi_rad']
    cols.extend(f'{stem}{k}{tail}' for k in range(1,n+1) for stem,tail in [('r_',''),('phi_','_rad')])
    t = pd.read_csv(folder/'trace.csv.gz',usecols=cols,float_precision='round_trip')
    assert np.all(np.diff(t.time_s) > 0), folder
    z, e = collective_metrics(t[[f'r_{k}' for k in range(1,n+1)]].to_numpy(),
                              t[[f'phi_{k}_rad' for k in range(1,n+1)]].to_numpy(),
                              parameters['a'],parameters['kappa_rad'])
    err_r = float(np.max(np.abs(np.abs(z)-t.R)))
    err_phi = float(np.max(np.abs(e-t.Ephi_rad))) if n > 1 else 0.
    assert err_r <= epsilon and err_phi <= epsilon, (folder,err_r,err_phi)
    err_q = err_f = None
    if m['complete']:
        lo, hi = m['evaluation_start_s'], m['evaluation_end_s']
        w = t[t.time_s.between(lo-1e-9,hi+1e-9)]
        assert abs(w.time_s.iloc[0]-lo)<1e-8 and abs(w.time_s.iloc[-1]-hi)<1e-8, folder
        q = time_rmse(w.time_s,w.q_target,w.q_m,parameters['Qscale'])
        f = time_rmse(w.time_s,w.target_Hz,w.motor_estimated_Hz)
        err_q = abs(q-m['E_q_RMSE_over_Qscale'])
        err_f = abs(f-m['estimated_frequency_RMSE_Hz'])
        assert err_q <= epsilon and err_f <= epsilon, (folder,err_q,err_f)
    else:
        assert m['E_q_RMSE_over_Qscale'] is None, folder
    counts = {}
    prefix_count = 0
    for item in initial:
        device = item['device']
        counts[device] = item['initial_n']
        prefix = item['prefix']
        prefix_count += len(prefix)
        assert [x['event'] for x in prefix] == list(range(1,counts[device]+1)), folder
        for event in prefix:
            distance = margin(item['branch'],item['delay_us'],event['event'],event['before_mS'],event['after_mS'])
            ok = distance <= 1e-10 and event['before_mS'] > 0 and event['after_mS'] > 0
            assert ok == event['support'] or abs(distance-1e-10) <= 1e-14, folder
    try:
        events = pd.read_csv(folder/'events.csv',float_precision='round_trip')
    except pd.errors.EmptyDataError:
        events = pd.DataFrame()
    state_rows = 0 if suite=='sensitivity' else save_conductance_history(folder,initial,events)
    if 'kind' in events:
        events = events[events.kind=='physical_write']
    accepted = m['accepted_online_writes']
    assert len(events) == accepted and accepted <= m.get('budget_cap',m.get('budget')), folder
    if len(events):
        assert events.decision_time_s.is_unique, folder
        invalid = events[~events.support_valid | ~events.frequency_valid]
        if len(invalid):
            assert list(invalid.index)==[events.index[-1]] and not m['complete'], folder
            reason = 'realized_outside_support' if not invalid.support_valid.iloc[0] else 'realized_frequency_bound'
            assert m['failure_reason']==reason, folder
    by_id = {item['device']: item for item in initial}
    last_conductances = {item['device']:item['initial_G_mS'] for item in initial}
    for row in events.itertuples():
        device = row.device
        counts[device] += 1
        assert counts[device] == row.event and 1 <= row.event <= 10, (folder,device)
        start = by_id[device]
        assert (row.branch,row.delay_us)==(start['branch'],start['delay_us']), folder
        distance = margin(row.branch,row.delay_us,row.event,row.before_G_mS,row.after_G_mS)
        valid = distance <= 1e-10 and row.before_G_mS > 0 and row.after_G_mS > 0
        assert valid == row.support_valid or abs(distance-1e-10) <= 1e-14, folder
        if suite != 'sensitivity':
            assert abs(last_conductances[device]-row.before_G_mS) <= epsilon, folder
        last_conductances[device] = row.after_G_mS
        if 'waveform_duration_s' in events:
            assert abs(row.time_s-row.decision_time_s-(1000+row.delay_us)*1e-6)<1e-10, folder
    objective = 0
    if suite=='predictive' and (folder/'decision_log.csv.gz').stat().st_size>40:
        try:
            log = pd.read_csv(folder/'decision_log.csv.gz')
        except pd.errors.EmptyDataError:
            log = pd.DataFrame()
        for _, choices in log.groupby('decision') if len(log) else []:
            winner = choices[choices.chosen]
            permitted = choices[choices.admissible]
            assert len(winner)==1 and winner.admissible.iloc[0], folder
            assert winner.J.iloc[0]<=permitted.J.min()+1e-12+1e-15, folder
            assert (choices.last_observation_time_s <= choices.time_s+1e-9).all(), folder
            objective += 1
    return dict(suite=suite,run_id=m['run_id'],R_abs_error=err_r,Ephi_abs_error_rad=err_phi,
                displacement_rmse_abs_error=err_q,frequency_rmse_abs_error_Hz=err_f,
                checked_prefix_events=prefix_count,checked_accepted_writes=accepted,
                checked_predictive_argmins=objective,materialized_conductance_states=state_rows,
                terminated_by_realized_invalid_write=int(len(events)>0 and
                    (not events.support_valid.iloc[-1] or not events.frequency_valid.iloc[-1])))


def main(workers=4):
    expected = {'predictive':52,'original_32':32,'stochastic':2400,'stochastic_extra':3600,'sensitivity':320}
    runs = []
    for suite,n in expected.items():
        root = WORK/suite
        m = pd.read_csv(root/'metrics.csv',usecols=['run_id'])
        assert len(m)==n and m.run_id.is_unique, suite
        runs.extend((suite,root/'runs'/run_id) for run_id in m.run_id)
    with ProcessPoolExecutor(max_workers=workers,initializer=load_hulls) as pool:
        rows = list(pool.map(audit,runs,chunksize=20))
    result = pd.DataFrame(rows)
    VALIDATION.mkdir(parents=True,exist_ok=True)
    result.to_csv(VALIDATION/'run_level_checks.csv',index=False,float_format='%.17g')
    summary = dict(status='PASS',runs=len(rows),audited_suites=expected)
    for col in result.columns[2:6]:
        summary['maximum_'+col]=float(result[col].max())
    for col in result.columns[6:]:
        summary[col]=int(result[col].sum())
    (VALIDATION/'run_checks_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers',type=int,default=4)
    main(parser.parse_args().workers)
