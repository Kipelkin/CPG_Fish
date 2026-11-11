"""Compare frozen manuscript claims, archived arrays and fresh runs."""
import hashlib
import json
import math

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull

from memristive_cpg.metrics import wilson
from scripts.runtime import DATA, TABLES, VALIDATION, WORK

REF = DATA / 'reference'
SUPP = REF / 'supplement'
EXACT = 'EXACT MATCH'
CLOSE = 'WITHIN TOLERANCE'
DIFF = 'MISMATCH'
MISSING = 'NOT REPRODUCIBLE'
UNCHECKED = 'NOT VERIFIED'
FLOAT_TOL = 2e-11  # Bound used by archived independent trace/metric audits.
PHASE_TOL = 1e-7   # Independent DOP853/RK45 no-pulse control criterion.
ROWS = []


def read(path):
    return pd.read_csv(path, float_precision='round_trip', low_memory=False)


def rel(path):
    return str(path.relative_to(DATA.parent))


def add(item, metric, expected, actual, tolerance=0., units='', source='', evidence='', note='', samples=1, status=None):
    numeric = isinstance(expected, (int, float, np.integer, np.floating)) and isinstance(actual, (int, float, np.integer, np.floating))
    numeric = numeric and not (pd.isna(expected) or pd.isna(actual))
    difference = abs(float(actual)-float(expected)) if numeric else None
    relative = difference / abs(float(expected)) if numeric and expected else (0. if numeric and difference == 0 else None)
    if status is None:
        status = (EXACT if difference == 0 else CLOSE if difference <= tolerance else DIFF) if numeric else (
            EXACT if expected == actual else DIFF)
    ROWS.append(dict(item=item, metric=metric, original=expected, reproduced=actual,
                     absolute_difference=difference, relative_difference=relative, tolerance=tolerance,
                     units=units, samples=samples, status=status, reference=source,
                     reproduction=evidence, notes=note))


def compare_frame(item, expected, actual, keys, columns=None, tolerance=FLOAT_TOL):
    a, b = read(expected), read(actual)
    if keys:
        a, b = a.set_index(keys).sort_index(), b.set_index(keys).sort_index()
        assert a.index.is_unique and b.index.is_unique, item
    add(item, 'number of records', len(a), len(b), source=rel(expected), evidence=rel(actual))
    if not a.index.equals(b.index):
        add(item, 'record alignment', 'same keys', 'different keys', source=rel(expected), evidence=rel(actual))
        return
    for col in columns or a.columns:
        assert col in b, (item, col)
        x, y = a[col], b[col]
        if not x.isna().equals(y.isna()):
            add(item, col+' missing mask', 'same', 'different', source=rel(expected), evidence=rel(actual))
        mask = x.notna() & y.notna()
        if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y) and mask.any():
            px, py = x[mask].to_numpy(dtype=float), y[mask].to_numpy(dtype=float)
            delta = np.abs(px-py)
            j = int(np.argmax(delta))
            add(item, col, px[j], py[j], tolerance, source=rel(expected), evidence=rel(actual),
                samples=int(mask.sum()), note='Original and reproduced values at maximum absolute deviation.')
        else:
            deviations = int((x[mask] != y[mask]).sum())
            add(item, col+' mismatches', 0, deviations, source=rel(expected), evidence=rel(actual), samples=int(mask.sum()))


def published(item, metric, expected, actual, decimals, units, source, evidence, note=''):
    add(item, metric, expected, actual, .5*10**(-decimals)+2e-15, units, source, evidence, note)


def measurements_and_calibration():
    work = WORK / 'measurements'
    a = read(work/'archive_records.csv')
    for key, expected, actual in [
        ('POT records', 39, sum(a.branch == 'POT')), ('DEP records', 38, sum(a.branch == 'DEP')),
        ('complete cycles', 18905, a.cycles.sum()), ('conductance transitions', 189050, a.transitions.sum()),
        ('distinct delays', 19, a.delay_us.nunique()), ('archive read files matching exactly', 77, a.archive_read_exact.sum())
    ]:
        add('Raw measurements', key, expected, int(actual), source='Fish_CPG-2.pdf §2.1 and §3.1; raw archive', evidence=rel(work/'archive_records.csv'))
    if a.waveform_checked.any():
        add('Raw measurements', 'maximum waveform read-window error', 0., a.waveform_read_error_V.max(),
            1e-14, 'V', 'Archive voltage means', rel(work/'archive_records.csv'), '30 samples at 1 MHz')
    else:
        add('Raw measurements', 'waveform read-window check', 77, 0, source='Raw archive',
            evidence=rel(work/'archive_records.csv'), status=UNCHECKED, note='Run invoked with --skip-waveforms.')
    add('Raw measurements', 'maximum conductance export error', 0., a.archive_conductance_error_S.max(),
        1e-15, 'S', 'Archive pulse/COND', rel(work/'archive_records.csv'))
    add('Figure 1A', 'paired numeric voltage and current sweep', 'source numeric sweep', 'supplied bitmap',
        source='data/reference/figure1/original_IV_supplied.png', evidence=rel(work/'archive_members.csv'),
        note='Archive contains pulse voltage, conductance and endurance records. No numeric I–V sweep located.',
        status=MISSING)
    compare_frame('Figure 1B', REF/'figure1/endurance_mean_plotted.csv', work/'endurance_plotted.csv',
                  ['protocol_cycle'], tolerance=1e-11)
    for file, keys in [('record_characteristics.csv', ['record']),
                       ('eq3_recursive_errors.csv', ['branch','fold','record','event'])]:
        compare_frame('Figure 2', REF/'figure2'/file, WORK/'figure2/data'/file, keys)
    d = WORK / 'calibration/data'
    for label, file, key, columns in [
        ('coefficients_full', 'coefficients', ['branch','delay_us','fit'], None),
        ('support_full', 'support_screen', ['branch','delay_us','fit','event'], None),
        ('record_inventory', 'records', ['record'], None),
        ('validation_full', 'event_validation', ['branch','delay_us','fold','test_record','event'], None)
    ]:
        compare_frame('Supplement S1', SUPP/f'Table_S1_{label}.csv', d/f'{file}.csv', key, columns)
    compare_frame('Supplement S1 ordered residuals', REF/'calibration/ordered_residuals_all_fit.csv',
                  d/'ordered_residuals_all_fit.csv', ['record','cycle'])
    compare_frame('Supplement S1 transitions', REF/'calibration/measured_transition_pairs.csv.gz',
                  d/'measured_transition_pairs.csv.gz', None)
    compare_frame('Supplement S1 pooled RMSE', SUPP/'Table_S1_pooled_prediction_RMSE.csv',
                  WORK/'figure2/data/pooled_rmse.csv', ['branch','fold','event'])
    c = read(d/'coefficients.csv')
    s = read(d/'support_screen.csv')
    e = read(d/'event_validation.csv')
    expected = {
        'POT': [-.040364,.371013,.198951,.013939,.5296,.7606,.312,.522,81.1],
        'DEP': [.018449,-.379704,-.061464,-.006161,-.4227,-.4683,.073,.103,94.8],
    }
    table = []
    j = np.arange(1,11)
    for branch, values in expected.items():
        fit = c[(c.branch == branch)&(c.delay_us == 500)&(c.fit == 'all_records_fit')].iloc[0]
        prefix = s[(s.branch == branch)&(s.delay_us == 500)&(s.fit == 'all_records_fit')&(s.event == 10)].iloc[0]
        mu = fit.a_mS+fit.b_mS*(j==1)+fit.c_mS/j+fit.d_mS*np.log(j)
        heldout = e[(e.branch == branch)&(e.delay_us == 500)&(e.event == 10)].set_index('fold')
        cols = ['a_mS','b_mS','c_mS','d_mS','first_increment_mS','ten_event_change_mS',
                'RMSE_1_to_2_mS','RMSE_2_to_1_mS','support_pass_pct']
        actual = [*[fit[key] for key in cols[:4]], mu[0], mu.sum(),
                  heldout.loc['1_to_2','cumulative_rmse_mS'],heldout.loc['2_to_1','cumulative_rmse_mS'],
                  100*prefix.continuous_prefix_pass_fraction]
        table.append(dict(branch=branch, delay_us=500, **dict(zip(cols,actual,strict=True))))
        for key, old, new, dec in zip(cols,values,actual,[6,6,6,6,4,4,3,3,1],strict=True):
            published('Table 1', f'{branch} {key}', old, new, dec, '%' if key.endswith('pct') else 'mS',
                      'Fish_CPG-2.pdf p. 20, Table 1', rel(TABLES/'Table_1.csv'))
    pd.DataFrame(table).to_csv(TABLES/'Table_1.csv', index=False, float_format='%.17g')



def controller_inputs():
    frozen = DATA/'processed/control_inputs'
    cal = WORK/'calibration/data'
    dynamic = WORK/'phase/results'
    for folder,file,keys in [
        (cal,'coefficients.csv',['branch','delay_us','fit']),
        (cal,'ordered_residuals_all_fit.csv',['record','cycle']),
        (cal,'measured_transition_pairs.csv.gz',None),
        (cal,'screened_prefixes_with_origin.csv',['branch','delay_us','fit','initial_index']),
        (dynamic,'state_catalog.csv',['state_id']),
        (dynamic,'selected_virtual_devices.csv',['case','device_id']),
    ]:
        compare_frame('Controller input from fresh calibration',frozen/file,folder/file,keys)
    measured = read(cal/'measured_transition_pairs.csv.gz')
    rows = []
    for key,g in measured.groupby(['branch','delay_us','event'],sort=True):
        hull = ConvexHull(g[['Gprev_mS','Gnext_mS']].to_numpy())
        rows.extend([(*key,*abc) for abc in hull.equations])
    h = pd.DataFrame(rows,columns=['branch','delay_us','event','a','b','c'])
    path = cal/'support_halfspaces_recomputed.csv'
    h.to_csv(path,index=False,float_format='%.17g')
    compare_frame('Controller event hulls from measurements',frozen/'support_halfspaces.csv',path,None)
    old = json.loads((frozen/'interface.json').read_text())
    new = json.loads((dynamic/'interface.json').read_text())
    for key,value in old.items():
        add('Controller interface',key,value,new[key],FLOAT_TOL,source=rel(frozen/'interface.json'),
            evidence=rel(dynamic/'interface.json'))


def input_integrity():
    manifest = DATA/'input_inventory.json'
    entries = json.loads(manifest.read_text())
    checks = []
    for filename,entry in entries.items():
        path = DATA.parent/filename
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream,'sha256').hexdigest()
        checks.append(dict(file=filename,expected_sha256=entry['sha256'],actual_sha256=digest,
                           exact=digest==entry['sha256'] and path.stat().st_size==entry['bytes']))
    pd.DataFrame(checks).to_csv(VALIDATION/'input_integrity.csv',index=False)
    count = sum(check['exact'] for check in checks)
    add('Input manifest','file hashes and sizes matching',len(checks),count,source=rel(manifest),
        evidence=rel(VALIDATION/'input_integrity.csv'))
    if count != len(checks):
        raise ValueError('Input inventory mismatch. See results/validation/input_integrity.csv.')

def phase():
    d = WORK/'phase/results'
    metrics = read(d/'selected_metrics.csv').set_index('case')
    for case in ['nominal','weak','detuned_strong','detuned_weak']:
        file = f'trace_{case}.csv'
        compare_frame('Figure 3 trajectory '+case, REF/'phase'/file, d/file, None, tolerance=PHASE_TOL)
    for item, file, keys in [('Phase scan', 'fixed_gain_state_scan.csv', ['target_K_rad_s','mismatch_endpoint_Hz','seed']),
                             ('Finite coupling states', 'fixed_gain_K_catalog.csv', ['plus','minus'])]:
        compare_frame('Figure 3 '+item, REF/'phase'/file, d/file, keys, tolerance=PHASE_TOL)
    compare_frame('Supplement S2', SUPP/'Table_S2_phase_metrics.csv', d/'selected_metrics.csv', ['case'], tolerance=PHASE_TOL)
    for label, folder in [('unperturbed_trace.csv', 'figure4_alternative'),
                           ('supplemental_weak_reference.csv', 'figure4_enhanced'),
                           ('panel_D_comparison.csv', 'figure4_enhanced')]:
        compare_frame('Figure 3 phase memory', REF/'phase'/label, WORK/'phase'/folder/label, None, tolerance=PHASE_TOL)
    shifts = json.loads((WORK/'phase/figure4_enhanced/new_control_checks.json').read_text())
    for case, expected, label in [('nominal',1.74,'strong'),('weak',18.50,'weak')]:
        published('Figure 3', label+' recovery from end of pulse', expected, metrics.loc[case,'target_recovery_s'],
                  2, 's', 'Fish_CPG-2.pdf §3.3', rel(d/'selected_metrics.csv'))
        add('Figure 3', label+' late global phase shift', math.pi/10, shifts[f'{label}_final_phase_shift_rad'],
            PHASE_TOL, 'rad', 'Fish_CPG-2.pdf §3.3, π/10', rel(WORK/'phase/figure4_enhanced/new_control_checks.json'))
    baseline = read(d/'baseline_scan.csv')
    add('Section 3.3', 'one-oscillator baseline runs', 27, len(baseline), source='Fish_CPG-2.pdf §3.3', evidence=rel(d/'baseline_scan.csv'))
    add('Section 3.3', 'single-oscillator maximum |numeric − Eq. 7|', 0., baseline.f_error_Hz.abs().max(),
        1e-12, 'Hz', 'Eq. (7)', rel(d/'baseline_scan.csv'))


def tracking():
    prescribed = WORK/'prescribed'
    predictive = WORK/'predictive/metrics.csv'
    for item, label, file, keys in [
        ('Figure 4 historical', 'budget_metrics.csv', 'budget_metrics.csv', ['controller','programming_budget']),
        ('Figure 4 historical', 'programming_events.csv', 'programming_events.csv', ['event']),
        ('Figure 4 historical', 'trajectories.csv.gz', 'trajectories.csv.gz', None),
    ]:
        compare_frame(item, REF/'prescribed'/label, prescribed/file, keys)
    for name, file, keys in [
        ('Table S3 budgets','budget_metrics.csv',['controller','programming_budget']),
        ('Table S3 writes','programming_events.csv',['event']),
        ('Figure S1','trajectories.csv.gz',None),
    ]:
        source = {'Table S3 budgets':'Table_S3_budget_metrics_full.csv',
                  'Table S3 writes':'Table_S3_programming_events.csv',
                  'Figure S1':'Figure_S1_trajectories.csv.gz'}[name]
        if name == 'Figure S1':
            keys = ['controller','programming_budget','time_s']
            part = read(SUPP/source)[keys]
            new = read(prescribed/file).merge(part,on=keys,validate='one_to_one')
            output = prescribed/'figure_S1_selection.csv.gz'
            new.to_csv(output,index=False,float_format='%.17g')
        else:
            output = prescribed/file
        compare_frame(name, SUPP/source, output, keys)
    table = read(prescribed/'budget_metrics.csv')
    sheet = table.set_index(['controller','programming_budget']).E_q_RMSE_over_Qscale
    expected_rmse = {'S': [1.026,1.161,.863,.635], 'N':[1.027,1.114,.765,.521]}
    expected_gains = {'S':16.2, 'N':26.0}
    for architecture, vals in expected_rmse.items():
        entries = [sheet.loc[(architecture+'-Fixed',0)],sheet.loc[(architecture+'-Device',1)],
                   sheet.loc[(architecture+'-Device',10)],sheet.loc[(architecture+'-Ideal',0)]]
        for condition, old, new in zip(['fixed','device budget 1','device budget 10','ideal'],vals,entries,strict=True):
            published('Figure 4 manuscript', f'{architecture} {condition} normalized RMSE', old, new, 3, 'Qscale units',
                      'Fish_CPG-2.pdf §3.4', rel(prescribed/'budget_metrics.csv'))
        published('Figure 4 manuscript', f'{architecture} budget 10 improvement', expected_gains[architecture],
                  100*(1-entries[2]/entries[0]),1,'%', 'Fish_CPG-2.pdf §3.4', rel(prescribed/'budget_metrics.csv'),
                  '100 × (fixed RMSE − device RMSE) / fixed RMSE with full precision.')
    writes = read(prescribed/'programming_events.csv')
    published('Figure 4 earlier manuscript', 'target frequency §3.4', 2.9, writes.frequency_after_write_Hz.iloc[-1],
              1, 'Hz', 'Fish_CPG.pdf §3.4 (earlier revision)', rel(prescribed/'programming_events.csv'))
    published('Figure 4 manuscript', 'target frequency §3.4', 2.968, writes.frequency_after_write_Hz.iloc[-1],
              3, 'Hz', 'Fish_CPG-2.pdf §3.4 (latest located revision)', rel(prescribed/'programming_events.csv'))
    published('Figure 4 manuscript', 'target frequency caption', 2.968, writes.frequency_after_write_Hz.iloc[-1],
              3, 'Hz', 'Fish_CPG-2.pdf Figure 4 caption', rel(prescribed/'programming_events.csv'))
    add('Figure 4 manuscript','logged interval between scheduled writes',.2,float(np.diff(writes.completion_s).mean()),
        1e-12,'s','Fish_CPG-2.pdf §3.4 decision cadence',rel(prescribed/'programming_events.csv'),
        'Comparison of timing across distinct protocols. The historical series has no predictive decisions.',status=DIFF)
    add('Figure 4 manuscript','programming decision protocol','predictive decision every 0.2 s',
        'prescribed DEP writes spaced 0.04 s',source='Fish_CPG-2.pdf §3.4',evidence=rel(prescribed/'derived_quantities.json'))
    fresh = read(predictive)
    main = fresh[fresh.group == 'main']
    main_path = predictive.parent/'metrics_main_target.csv'
    main.to_csv(main_path,index=False,float_format='%.17g')
    compare_frame('Causal predictive replay', REF/'predictive/results_main.csv', main_path, ['controller','budget_cap'],
                  ['accepted_online_writes','ideal_coordinate_updates','complete','failure_reason',
                   'E_q_RMSE_over_Qscale','estimated_frequency_RMSE_Hz','minimum_R','maximum_Ephi_rad'])
    fresh.to_csv(TABLES/'predictive_tracking.csv',index=False,float_format='%.17g')


def stochastic():
    base = WORK/'original_32'
    compare_frame('Archived original 32', SUPP/'Table_S4_archived_32_runs.csv', base/'metrics.csv', ['seed'],
                  ['complete','failure_reason','accepted_online_writes','initial_support_failures','last_time_s'])
    h = read(base/'trace_hash_check.csv')
    add('Archived original 32','byte-identical decompressed traces',32,int(h.exact.sum()),
        source=rel(SUPP/'Table_S4_archived_32_runs.csv'),evidence=rel(base/'trace_hash_check.csv'))
    a = read(base/'metrics.csv')
    add('Section 3.5','original N-Device completed',5,int(a.complete.sum()),source='Fish_CPG-2.pdf §3.5',
        evidence=rel(base/'metrics.csv'), note='31 initial_support, 1 realized_outside_support in the available archive.')
    expected = read(SUPP/'Table_S4_frequency_only_run_results.csv')
    new = read(WORK/'stochastic/metrics.csv')
    extra = read(WORK/'stochastic_extra/metrics.csv')
    assert len(new) == 2400 and len(extra) == 3600
    all_rows = pd.concat([new.assign(study=np.where(new.evaluation_end_s==8,'heldout','long_window')),
                          extra.assign(study=np.select([extra.scenario == 'main', extra.evaluation_end_s == 20],
                                                       ['primary','long_window'],default='heldout'))],ignore_index=True)
    common = ['complete','initialization_success','physical_initialization_success','overall_success','failure_reason',
              'accepted_online_writes','E_q_RMSE_over_Qscale','estimated_frequency_RMSE_Hz','minimum_R','maximum_Ephi_rad']
    as_old = new.assign(run_id=new.reference_run_id)
    result = WORK/'stochastic/metrics_reference_keys.csv'
    as_old.to_csv(result,index=False,float_format='%.17g')
    compare_frame('Table 2 individual runs', SUPP/'Table_S4_frequency_only_run_results.csv', result, ['run_id'], common)
    keys = all_rows.reference_run_id
    old = read(REF/'stochastic/FINAL_RESULTS.csv')
    old = old[old.run_id.isin(keys)]
    assert len(old) == len(all_rows)
    old_path = WORK/'stochastic/final_results_selected.csv'
    new_path = WORK/'stochastic/final_results_recomputed.csv'
    old.to_csv(old_path,index=False,float_format='%.17g')
    all_rows.assign(run_id=keys).to_csv(new_path,index=False,float_format='%.17g')
    compare_frame('Supplement S4 archived 6000 runs', old_path,new_path,['run_id'],common)
    summaries = []
    for (study,scenario,controller,mode), group in all_rows.groupby(['study','scenario','controller','mode']):
        full = group[group.complete & group.initialization_success]
        total = len(group)
        successes = int(group.overall_success.sum())
        lo, hi = wilson(successes,total)
        summaries.append(dict(study=study,scenario=scenario,controller=controller,mode=mode,
                              evaluation_end_s=group.evaluation_end_s.iloc[0],n_runs=total,
                              physical_initialization_n=int(group.physical_initialization_success.sum()),
                              strict_initialization_n=int(group.initialization_success.sum()),completion_n=int(group.complete.sum()),
                              overall_success_n=successes,overall_success_CI95_low=lo,overall_success_CI95_high=hi,
                              conditional_error_n=len(full),E_q_RMSE_over_Qscale_median=full.E_q_RMSE_over_Qscale.median(),
                              Eomega_RMSE_rad_s_median=2*math.pi*full.estimated_frequency_RMSE_Hz.median(),
                              failure_reasons=json.dumps(group.failure_reason.fillna('no_simulator_stop').value_counts().to_dict(),sort_keys=True)))
    summary = pd.DataFrame(summaries)
    summary.to_csv(TABLES/'stochastic_summary.csv',index=False,float_format='%.17g')
    table2 = summary[summary.controller.str.contains(r'(?:Reassigned-J-Frequency|Reassigned-Guarded-Frequency)')]
    table2.to_csv(TABLES/'Table_2.csv',index=False,float_format='%.17g')
    manuscript = [(8,'J','independent',281,16,3.3,8.5,1.011),(8,'J','shared',276,20,4.4,10.1,.982),
                  (8,'Guarded','independent',300,43,10.8,18.8,1.026),(8,'Guarded','shared',300,49,12.6,20.9,1.024),
                  (20,'J','independent',281,18,3.8,9.3,1.014),(20,'J','shared',276,23,5.2,11.2,1.004),
                  (20,'Guarded','independent',300,16,3.3,8.5,1.020),(20,'Guarded','shared',300,28,6.5,13.2,1.012)]
    for end, algorithm, mode, completed, success, lo, hi, rmse in manuscript:
        row = table2[(table2.evaluation_end_s==end)&(table2.controller==f'Reassigned-{algorithm}-Frequency')&
                     table2['mode'].str.startswith(mode)].iloc[0]
        item = f'Table 2 / S4a {end} s {algorithm} {mode}'
        for metric, oldval, val, decimals, unit in [
            ('trials',300,row.n_runs,None,''),('complete',completed,row.completion_n,None,''),
            ('success',success,row.overall_success_n,None,''),
            ('Wilson low',lo,100*row.overall_success_CI95_low,1,'%'),
            ('Wilson high',hi,100*row.overall_success_CI95_high,1,'%'),
            ('median q RMSE',rmse,row.E_q_RMSE_over_Qscale_median,3,'normalized'),
        ]:
            if decimals is None:
                add(item,metric,oldval,val,source='Fish_CPG-2.pdf Table 2',evidence=rel(TABLES/'Table_2.csv'))
            else:
                published(item,metric,oldval,val,decimals,unit,'Fish_CPG-2.pdf Table 2',rel(TABLES/'Table_2.csv'))
    init = WORK/'initialization'
    ini_summary = []
    for file in ['initialization_stress.csv','own_G0_initialization_sensitivity.csv','global_initialization_sensitivity.csv']:
        compare_frame('Supplement S4b '+file, REF/'stochastic'/file, init/file, ['case','mode','seed'])
        for (case,mode), group in read(init/file).groupby(['case','mode']):
            k = int(group.initialization_success.sum())
            ini_summary.append(dict(protocol=file,case=case,mode=mode,total=len(group),successes=k,
                                    wilson_low=wilson(k,len(group))[0],wilson_high=wilson(k,len(group))[1]))
    pd.DataFrame(ini_summary).to_csv(TABLES/'initialization_summary.csv',index=False,float_format='%.17g')


def sensitivity_and_hardware():
    work = WORK/'sensitivity'
    m = read(work/'metrics.csv')
    baseline, refined = m[~m.refined], m[m.refined]
    keys = ['controller','scenario','suite','end_s','budget','pattern','drift_at20_fraction']
    columns = ['complete','initialization_success','failure_reason','accepted_online_writes','maximum_device_writes',
               'E_q_RMSE_over_Qscale','estimated_frequency_RMSE_Hz','Eomega_RMSE_rad_s','minimum_R','maximum_Ephi_rad',
               'overall_success','last_time_s','final_common_Hz','minimum_motor_amplitude']
    for name, group in [('new_results.csv',baseline),('refined_results.csv',refined)]:
        target = work/f'reproduced_{name}'
        group.to_csv(target,index=False,float_format='%.17g')
        compare_frame('Drift and direction '+name, REF/'sensitivity'/name, target, keys, columns)
    baseline[baseline.suite=='drift'].to_csv(TABLES/'conductance_drift.csv',index=False,float_format='%.17g')
    baseline[baseline.suite=='asymmetry'].to_csv(TABLES/'programming_direction.csv',index=False,float_format='%.17g')
    isolated = baseline[(baseline.suite=='drift') & (baseline.scenario=='hold') &
                        (baseline.controller=='Fixed') & (baseline.end_s==20)]
    brackets = []
    for pattern in ['frequency_common','frequency_differential','coupling_heterogeneous']:
        for sign in (-1,1):
            g = isolated[(isolated.pattern==pattern)&(isolated.drift_at20_fraction*sign>0)].copy()
            g['magnitude'] = g.drift_at20_fraction.abs()
            g = g.sort_values('magnitude')
            first = g[~g.overall_success]
            failure = first.magnitude.iloc[0] if len(first) else np.nan
            allowed = g[g.overall_success & ((g.magnitude<failure) if np.isfinite(failure) else True)]
            passed = allowed.magnitude.max()
            brackets.append(dict(pattern=pattern,sign=sign,largest_passing_drift_pct=100*passed,
                                 first_failing_drift_pct=100*failure,maximum_tested_drift_pct=100*g.magnitude.max()))
            expected = {'frequency_differential':(.5,1.), 'coupling_heterogeneous':(1.,2.)}
            if pattern in expected:
                for label,old,new in zip(('largest tested pass','first tested failure'),expected[pattern],
                                         (100*passed,100*failure),strict=True):
                    add('Section 3.5 drift brackets',f'{pattern}, sign {sign}: {label}',old,new,1e-12,'%',
                        'Fish_CPG-2.pdf §3.5; Supplementary Table S4e',rel(TABLES/'drift_brackets.csv'))
    pd.DataFrame(brackets).to_csv(TABLES/'drift_brackets.csv',index=False,float_format='%.17g')

    for file, table in [('Table_S4_drift_metrics_full.csv','conductance_drift.csv'),
                        ('Table_S4_direction_tests_full.csv','programming_direction.csv')]:
        archive = read(SUPP/file)
        if 'record_type' in archive:
            archive = archive[archive.record_type=='matched_direction_trial'] if (archive.record_type=='matched_direction_trial').any() else archive[archive.run_id.notna()]
        archive = archive.dropna(subset=['run_id'])
        available = read(TABLES/table)
        take = archive[keys].drop_duplicates()
        take = take.merge(available[keys],on=keys,validate='one_to_one')
        a = archive.merge(take,on=keys,validate='one_to_one')
        b = available.merge(take,on=keys,validate='one_to_one')
        if len(a):
            dest,src = work/('compare_'+file),work/('reference_'+file)
            a.to_csv(src,index=False,float_format='%.17g')
            b.to_csv(dest,index=False,float_format='%.17g')
            compare_frame(file,src,dest,keys,columns)
    for pattern in ['frequency_common','frequency_differential','coupling_heterogeneous']:
        row = baseline[(baseline.end_s==20)&(baseline.controller=='Fixed')&(baseline.scenario=='hold')&
                       (baseline.pattern==pattern)&np.isclose(baseline.drift_at20_fraction,.02)].iloc[0]
        compare_frame('Figure S2 '+pattern, SUPP/f'Figure_S2_{pattern}.csv.gz',
                      work/'runs'/row.run_id/'trace.csv.gz',['time_s'],
                      ['q_m','motor_true_Hz','R','Ephi_rad','accepted_writes'])
    for name, file, key in [
        ('Supplement S5 frequency states','original_frequency_graph_vertices.csv',['plus_n','minus_n']),
        ('Reachable edges','original_frequency_graph_edges.csv',['from_plus_n','from_minus_n','to_plus_n','to_minus_n']),
        ('POT/DEP increments','pot_dep_frequency_mapping.csv',['branch','arm','n_writes']),
        ('Forward vs static reachability','frequency_reachability_comparison.csv',['scenario']),
    ]:
        src = SUPP/'Table_S5_frequency_state_combinations.csv' if name.startswith('Supplement') else REF/'sensitivity'/file
        compare_frame(name,src,work/file,key)
    compare_frame('Supplement S5 conductance states',SUPP/'Table_S5_state_catalog.csv',
                  WORK/'phase/results/state_catalog.csv',['state_id'])
    compare_frame('Supplement S5 drift bounds',SUPP/'Table_S5_conditional_bounds.csv',
                  TABLES/'conditional_hardware_bounds.csv',['pattern','end_s'],
                  ['prepared_Gplus_mS','prepared_Gminus_mS','df_c_d_delta_Hz','endpoint_phase_derivative_rad_per_delta20',
                   'sufficient_delta20_abs_for_pointwise_phase_bound','phase_budget_rad'])
    h = read(TABLES/'Table_3.csv').set_index('metric').value
    for key,old,dec,unit in [
        ('virtual_devices',34,None,'devices'),('frequency_devices',2,None,'devices'),('coupling_devices',32,None,'devices'),
        ('POT_G_min',.1773,4,'mS'),('POT_G_max',3.2083,4,'mS'),('DEP_G_min',.9488,4,'mS'),('DEP_G_max',3.4542,4,'mS'),
        ('delay_count',19,None,'conditions'),('events_per_device',10,None,'writes'),('frequency_conversion',1,None,'Hz/mS'),
        ('reachable_states',21,None,'states'),('reachable_frequency_min',2.5,1,'Hz'),('reachable_frequency_max',3.261,3,'Hz'),
        ('reachable_increment_min',.0019,4,'Hz'),('reachable_increment_max',.5296,4,'Hz'),('gamma_c',1.744484,6,'rad/s/mS'),
        ('gamma_d',.722589,6,'rad/s/mS'),('coupling_sensitivity_bound',3.777,3,'rad/s/mS'),('write_waveform',1.5,1,'ms'),
        ('decision_interval',200,None,'ms'),('read_source_voltage',.20,2,'V'),('read_pulse_duration',50,None,'µs'),
        ('conditional_common_bound',1.50,2,'%'),('conditional_differential_bound',.234,3,'%'),('output_channels',1,None,'channel'),
        ('omega0',19.158469093,9,'rad/s')]:
        source = 'Supplementary Table S2' if key=='omega0' else 'Fish_CPG-2.pdf Table 3'
        if key in ('coupling_sensitivity_bound','conditional_common_bound','conditional_differential_bound'):
            step = 10**(-dec)
            delta = old-h[key] if key=='coupling_sensitivity_bound' else h[key]-old
            assert np.isfinite(delta)
            add('Table 3',key,old,h[key],step,unit,source,rel(TABLES/'Table_3.csv'),
                'Conservative directed rounding: sensitivity coefficient up; sufficient tolerated drift down.',
                status=EXACT if delta==0 else CLOSE if 0<delta<=step+2e-15 else DIFF)
        elif dec is None:
            add('Table 3',key,old,h[key],units=unit,source=source,evidence=rel(TABLES/'Table_3.csv'))
        else:
            published('Table 3',key,old,h[key],dec,unit,source,rel(TABLES/'Table_3.csv'))
    for metric in ['full physical control-cycle latency','device-voltage programming energy','crossbar operation']:
        add('Table 3 hardware validation',metric,'physical measurement','unavailable',source='Fish_CPG-2.pdf Table 3',
            note='Design requirement; no corresponding physical measurement.',status=MISSING)


def main():
    ROWS.clear()
    for directory in (TABLES,VALIDATION):
        directory.mkdir(parents=True,exist_ok=True)
    # Figure 2 pooled errors use equal weight per held-out record.
    folder = WORK/'figure2/data'
    r = read(folder/'eq3_recursive_errors.csv')
    pool = r.groupby(['branch','fold','event'],as_index=False).agg(
        record_mean_mse_mS2=('cumulative_mse_mS2','mean'),heldout_records=('record','nunique'))
    pool['rmse_mS'] = np.sqrt(pool.record_mean_mse_mS2)
    pool.to_csv(folder/'pooled_rmse.csv',index=False,float_format='%.17g')
    input_integrity()
    measurements_and_calibration()
    phase()
    controller_inputs()
    tracking()
    stochastic()
    sensitivity_and_hardware()
    result = pd.DataFrame(ROWS)
    result.to_csv(VALIDATION/'comparison.csv',index=False,float_format='%.17g')
    print(result.status.value_counts().to_string())
    print(result[result.status == DIFF][['item','metric','original','reproduced']].to_string(index=False))


if __name__ == '__main__':
    main()
