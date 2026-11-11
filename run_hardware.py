"""Recompute finite-state reachability and Table 3 design arithmetic."""
import json
from math import hypot, pi
import numpy as np
import pandas as pd
from scripts.runtime import CONFIGS, TABLES, WORK, command, sensitivity


def main():
    runtime = WORK / 'sensitivity'
    sensitivity(runtime)
    command('programming_reachability', runtime / 'source/pot_dep_analysis.py')
    v = pd.read_csv(runtime / 'original_frequency_graph_vertices.csv', float_precision='round_trip')
    e = pd.read_csv(runtime / 'original_frequency_graph_edges.csv', float_precision='round_trip')
    v = v[v.forward_reachable_with_budget10]
    e = e[e.allowed]
    f = v.set_index(['plus_n', 'minus_n']).frequency_Hz
    changes = np.array([r.frequency_Hz-f.loc[r.from_plus_n, r.from_minus_n] for r in e.itertuples()])
    assert (changes > 0).all()
    cal = pd.read_csv(WORK / 'calibration/data/records.csv')
    extent = cal.groupby('branch').agg(low=('G_min_mS', 'min'), high=('G_max_mS', 'max'))
    p = json.loads((CONFIGS / 'sensitivity.json').read_text())
    interface = json.loads((WORK / 'phase/results/interface.json').read_text())
    delay_count = cal.delay_us.nunique()
    n = p['base_parameters']['N_network']
    decision = p['base_parameters']['decision_interval']
    gain = interface['gamma_omega_rad_s_per_mS']/(2*pi)
    initial = v[(v.plus_n == 0) & (v.minus_n == 0)].iloc[0]
    gp, gm = initial.plus_G_mS, initial.minus_G_mS
    tau = p['base_parameters']['tau_omega']
    motion_limit = p['success']['motion_accuracy']['Eq_RMSE_over_Qscale_max']
    bounds = []
    for t in (8., 20.):
        factor = t*t/2-tau*t+tau*tau*(1-np.exp(-t/tau))
        for pattern, coeff in [('frequency_common', gain*(gp-gm)), ('frequency_differential', gain*(gp+gm))]:
            derivative = 2*pi*coeff*factor/20
            bounds.append(dict(pattern=pattern, end_s=t, prepared_Gplus_mS=gp, prepared_Gminus_mS=gm,
                               df_c_d_delta_Hz=coeff, endpoint_phase_derivative_rad_per_delta20=derivative,
                               sufficient_delta20_abs_for_pointwise_phase_bound=motion_limit/abs(derivative),
                               phase_budget_rad=motion_limit))
    limits = pd.DataFrame(bounds)
    TABLES.mkdir(parents=True, exist_ok=True)
    limits.to_csv(TABLES/'conditional_hardware_bounds.csv', index=False, float_format='%.17g')
    metrics = [
        ('virtual_devices', 8*(n-1)+2, 'devices', 'Eq. (13)'),
        ('frequency_devices', 2, 'devices', 'Differential pair'),
        ('coupling_devices', 8*(n-1), 'devices', 'Four per directed link'),
        ('POT_G_min', extent.loc['POT', 'low'], 'mS', 'All measured POT reads'),
        ('POT_G_max', extent.loc['POT', 'high'], 'mS', 'All measured POT reads'),
        ('DEP_G_min', extent.loc['DEP', 'low'], 'mS', 'All measured DEP reads'),
        ('DEP_G_max', extent.loc['DEP', 'high'], 'mS', 'All measured DEP reads'),
        ('delay_count', delay_count, 'conditions per polarity', 'Archive inventory'),
        ('events_per_device', 10, 'writes', 'Measured block length'),
        ('frequency_conversion', gain, 'Hz/mS', 'Eq. (7)'),
        ('reachable_states', len(v), 'states', 'Supported forward graph, 10 online writes, 1.9–3.4 Hz'),
        ('reachable_frequency_min', v.frequency_Hz.min(), 'Hz', 'Minimum reachable vertex'),
        ('reachable_frequency_max', v.frequency_Hz.max(), 'Hz', 'Maximum reachable vertex'),
        ('reachable_increment_min', changes.min(), 'Hz', 'Minimum allowed forward increment'),
        ('reachable_increment_max', changes.max(), 'Hz', 'Maximum allowed forward increment'),
        ('gamma_c', interface['gamma_c_rad_s_per_mS'], 'rad/s/mS', 'Selected coupling mapping'),
        ('gamma_d', interface['gamma_d_rad_s_per_mS'], 'rad/s/mS', 'Selected coupling mapping'),
        ('coupling_sensitivity_bound', 2*hypot(interface['gamma_c_rad_s_per_mS'],
                                             interface['gamma_d_rad_s_per_mS']), 'rad/s/mS', 'Triangle inequality'),
        ('write_waveform', (1000+500)/1000, 'ms', 'Two overlapping 1000 µs pulses, delay 500 µs'),
        ('decision_interval', 1000*decision, 'ms', 'Controller configuration, not measured total latency'),
        ('read_source_voltage', .20, 'V', 'Specified in Methods §2.1'),
        ('read_pulse_duration', 50, 'µs', 'Specified in Methods §2.1'),
        ('conditional_common_bound', 100*limits.iloc[2].sufficient_delta20_abs_for_pointwise_phase_bound,
         '%', '20 s isolated hold, sufficient endpoint phase bound'),
        ('conditional_differential_bound', 100*limits.iloc[3].sufficient_delta20_abs_for_pointwise_phase_bound,
         '%', '20 s isolated hold, sufficient endpoint phase bound'),
        ('output_channels', 1, 'channel', 'Eq. (12), one modeled actuator'),
        ('omega0', interface['omega0_rad_s'], 'rad/s', '2π·2.5 minus gain × initial conductance difference'),
    ]
    pd.DataFrame(metrics, columns=['metric', 'value', 'unit', 'derivation']).to_csv(
        TABLES / 'Table_3.csv', index=False, float_format='%.17g')
    print(f'{len(v)} reachable states and {len(e)} allowed edges, offset {interface["omega0_rad_s"]:.14g} rad/s')


if __name__ == '__main__':
    main()
