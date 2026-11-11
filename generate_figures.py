"""Publication and audit plots from measured and newly simulated data."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.runtime import DATA, FIGURES, TABLES, WORK

BLUE = '#2166a6'
ORANGE = '#c75814'
GREEN = '#00836a'
PURPLE = '#79549a'
GREY = '#626975'
STYLES = {'Fixed':(':', GREY), 'Ideal':('--', BLUE), 'Device':('-', ORANGE)}
plt.rcParams.update({'font.size':10, 'axes.labelsize':10, 'axes.titlesize':11,
                     'axes.spines.top':False, 'axes.spines.right':False, 'legend.frameon':False,
                     'legend.fontsize':8, 'pdf.fonttype':42, 'savefig.dpi':300})


def csv(path):
    return pd.read_csv(path, float_precision='round_trip', low_memory=False)


def save(fig, name):
    FIGURES.mkdir(parents=True,exist_ok=True)
    for suffix in ('png','pdf'):
        fig.savefig(FIGURES/f'{name}.{suffix}')
    plt.close(fig)


def panel(ax, letter, title):
    ax.set_title(title,loc='left')
    ax.set_title(letter,loc='right',fontweight='bold',fontsize=12)
    ax.grid(alpha=.17)


def figure1b():
    d = csv(WORK/'measurements/endurance_plotted.csv')
    fig,ax = plt.subplots(figsize=(5.8,3.45),layout='constrained')
    for state, color in [('HRS',BLUE),('LRS',ORANGE)]:
        ax.plot(d.protocol_cycle,d[f'I_{state}_mean_uA'],color=color,label=state)
    ax.set(xlabel='Switching cycle',ylabel='Read current (µA)')
    ax.legend()
    panel(ax,'B','Measured endurance across 77 records')
    save(fig,'Figure_1B')


def calibration_comparison():
    p = WORK/'figure2/data'
    a = csv(DATA/'reference/figure2/record_characteristics.csv')
    b = csv(p/'record_characteristics.csv')
    pair = a.merge(b,on=['record','branch'],suffixes=('_arch','_new'),validate='one_to_one')
    x = csv(DATA/'reference/figure2/plotted_rmse.csv')
    y = csv(p/'pooled_rmse.csv')
    diff = x.merge(y,on=['branch','fold','event'],suffixes=('_arch','_new'),validate='one_to_one')
    fig,axes = plt.subplots(1,2,figsize=(9.2,3.5),layout='constrained')
    limits = pair[['first_mean_mS_arch','first_mean_mS_new']].to_numpy()
    axes[0].plot([limits.min(),limits.max()],[limits.min(),limits.max()],':',color=GREY)
    for branch, color in [('POT',BLUE),('DEP',ORANGE)]:
        d = pair[pair.branch==branch]
        axes[0].scatter(d.first_mean_mS_arch,d.first_mean_mS_new,label=branch,color=color,s=18)
        for fold,g in diff[diff.branch==branch].groupby('fold'):
            axes[1].plot(g.event,(g.rmse_mS_new-g.rmse_mS_arch)*1e15,marker='.',label=f'{branch} {fold}',
                         color=color,alpha=.48 if '1->2' in fold else .95)
    axes[0].set(xlabel='Archived first-event increment (mS)',ylabel='Reproduced increment (mS)')
    axes[1].set(xlabel='Event',ylabel='Reproduced − archived RMSE (10⁻¹⁵ mS)',xticks=np.arange(1,11))
    axes[0].legend()
    axes[1].legend(loc='best')
    panel(axes[0],'A','Measured conductance changes')
    panel(axes[1],'B','Held-out cumulative prediction errors')
    save(fig,'Compare_Figure_2')


def phase_comparison():
    base = WORK/'phase/results'
    ref = DATA/'reference/phase'
    a, b = csv(ref/'fixed_gain_state_scan.csv'), csv(base/'fixed_gain_state_scan.csv')
    key = ['actual_K_rad_s','mismatch_endpoint_Hz']
    n, m = (x.groupby(key).phase_locked.sum().rename('locked').reset_index() for x in (a,b))
    fig,axes = plt.subplots(1,3,figsize=(12,3.55),layout='constrained')
    for case,color in [('nominal',BLUE),('weak',ORANGE),('detuned_strong',PURPLE),('detuned_weak',GREEN)]:
        old, new = (csv(p/f'trace_{case}.csv') for p in (ref,base))
        assert np.array_equal(old.t_s,new.t_s)
        d = new.t_s.between(179,205)
        x = new.t_s[d]-180
        axes[0].plot(x,new.Ephi_rad[d],color=color,label=case.replace('_',' '))
        axes[0].plot(x,old.Ephi_rad[d],':',color='black',alpha=.35,lw=1)
        axes[1].plot(x,new.Ephi_rad[d]-old.Ephi_rad[d],color=color,alpha=.7)
    axes[0].set(xlabel='Time from pulse onset (s)',ylabel='Relative-phase error (rad)')
    axes[0].legend()
    axes[1].set(xlabel='Time from pulse onset (s)',ylabel='Reproduced − archived (rad)',ylim=(-1e-8,1e-8))
    ax = axes[2]
    f = np.linspace(0,.10,101)
    ax.plot(f,3*np.pi*f,color=GREY,label=r'$K_{\mathrm{crit}}=3\pi\Delta f$')
    ax.scatter(n.mismatch_endpoint_Hz,n.actual_K_rad_s,marker='o',facecolors='none',
               edgecolors=GREY,s=55,label='Archived points')
    for value,color in [(3,BLUE),(0,ORANGE)]:
        group = m[m.locked==value]
        ax.scatter(group.mismatch_endpoint_Hz,group.actual_K_rad_s,marker='x',color=color,s=22,
                   label=f'Rerun: {value}/3 locked')
    ax.set(xlabel='Endpoint detuning (Hz)',ylabel=r'$K$ (rad/s)',ylim=(0,2.03))
    ax.legend(loc='upper left')
    for ax,letter,title in zip(axes,'ABC',['Original and reproduced','Pointwise difference','Locking boundary'],strict=True):
        panel(ax,letter,title)
    save(fig,'Compare_Figure_3')


def chosen_predictive():
    directory = WORK/'predictive'
    m = csv(directory/'metrics.csv')
    tracks = {}
    for row in m[(m.budget_cap==10)|m.controller.str.endswith(('Fixed','Ideal'))].itertuples():
        tracks[row.group,row.controller] = csv(directory/'runs'/row.run_id/'trace.csv.gz')
    return m, tracks


def predictive_figures():
    m, traces = chosen_predictive()
    old = csv(WORK/'prescribed/trajectories.csv.gz')
    historic = csv(WORK/'prescribed/budget_metrics.csv')
    for group, target, stem in [('main',2.968299459705117,'2p968'),('manuscript_2p9',2.9,'2p9')]:
        fig,axes = plt.subplots(2,2,figsize=(9.8,6.25),layout='constrained')
        for i, architecture in enumerate(['S','N']):
            ax = axes[0,i]
            for kind,(style,color) in STYLES.items():
                d = traces[group,f'{architecture}-{kind}']
                ax.plot(d.time_s,d.motor_true_Hz,style,color=color,label=kind)
            ax.step([0,2,8],[2.5,target,target],where='post',color='black',alpha=.55,label='Target')
            ax.axvspan(4,4.2,color='.75',alpha=.30)
            ax.set(xlabel='Time (s)',ylabel='Actuator frequency (Hz)')
            ax.legend(loc='best')
            panel(ax,'AB'[i],f'{architecture} architecture · frequency')
        ax = axes[1,0]
        for architecture,color in [('S',ORANGE),('N',BLUE)]:
            d = traces[group,f'{architecture}-Device']
            w = d.time_s.between(4.2,5.2)
            ax.plot(d.time_s[w],d.q_m[w],color=color,label=f'{architecture}-Device')
        ax.plot(d.time_s[w],d.q_target[w],':',color='black',label='Target')
        ax.set(xlabel='Time (s)',ylabel='Normalized displacement')
        ax.legend(loc='upper right')
        panel(ax,'C','One actuator per simulation')
        ax = axes[1,1]
        for arch,color in [('S',ORANGE),('N',BLUE)]:
            for kind,(style,_) in STYLES.items():
                r = m[(m.group==group)&(m.controller==f'{arch}-{kind}')].sort_values('budget_cap')
                if kind=='Device':
                    ax.plot(r.budget_cap,r.E_q_RMSE_over_Qscale,style,marker='.',color=color,label=f'{arch}-Device')
                else:
                    ax.axhline(r.E_q_RMSE_over_Qscale.iloc[0],ls=style,color=color,alpha=.65,label=f'{arch}-{kind}')
        ax.set(xlabel='Programming budget',ylabel='Normalized displacement RMSE')
        ax.legend(ncols=2)
        panel(ax,'D','Common 2–8 s evaluation window')
        save(fig,f'Figure_4_predictive_{stem}')

    fig,axes = plt.subplots(2,2,figsize=(10.2,6.3),layout='constrained')
    diffs = []
    for i,(arch,color) in enumerate([('S',ORANGE),('N',BLUE)]):
        ref = old[(old.controller==f'{arch}-Device')&(old.programming_budget==10)&(old.time_s<=8)].copy()
        ax = axes[0,i]
        ax.plot(ref.time_s,ref.motor_frequency_Hz,'-',color=GREY,label='Prescribed 2.968 Hz')
        for group,style,label in [('main','--','Predictive 2.968 Hz'),('manuscript_2p9',':','Predictive 2.9 Hz')]:
            sim = traces[group,f'{arch}-Device']
            ax.plot(sim.time_s,sim.motor_true_Hz,style,color=color,label=label)
            assert sim.time_s.is_monotonic_increasing
            q = np.interp(ref.time_s,sim.time_s,sim.q_m)
            omega = np.interp(ref.time_s,sim.time_s,sim.motor_true_Hz)
            diffs.append(pd.DataFrame(dict(architecture=arch,protocol=group,time_s=ref.time_s,
                                           original_q=ref.q_m,reproduced_q=q,difference_q=q-ref.q_m,
                                           original_frequency_Hz=ref.motor_frequency_Hz,
                                           reproduced_frequency_Hz=omega,difference_frequency_Hz=omega-ref.motor_frequency_Hz)))
            if group=='main':
                axes[1,0].plot(ref.time_s,q-ref.q_m,color=color,label=arch)
        ax.set(xlabel='Time (s)',ylabel='Actuator frequency (Hz)')
        ax.legend()
        panel(ax,'AB'[i],f'{arch} architecture · 10-write cap')
    pd.concat(diffs).to_csv(TABLES/'tracking_protocol_differences.csv.gz',index=False,float_format='%.17g')
    ax = axes[1,0]
    ax.set(xlabel='Time (s)',ylabel='Predictive − prescribed displacement')
    ax.legend()
    panel(ax,'C','Matched 2.968 Hz target')
    ax = axes[1,1]
    for arch,color in [('S',ORANGE),('N',BLUE)]:
        for d,label,style,budget in [(historic,'prescribed',':','programming_budget'),
                                     (m[m.group=='main'],'predictive','-','budget_cap')]:
            g = d[d.controller==f'{arch}-Device'].sort_values(budget)
            ax.plot(g[budget],g.E_q_RMSE_over_Qscale,style,color=color,marker='.',label=f'{arch} {label}')
    ax.set(xlabel='Programming budget',ylabel='Normalized displacement RMSE')
    ax.legend()
    panel(ax,'D','RMSE on the same 2–8 s window')
    save(fig,'Compare_Figure_4')


def supplementary():
    root = WORK/'prescribed'
    hist = csv(root/'figure_S1_selection.csv.gz')
    writes = csv(root/'programming_events.csv')
    target = writes.frequency_after_write_Hz.iloc[-1]
    fig,axes = plt.subplots(2,2,figsize=(9.5,6.4),layout='constrained')
    t = np.r_[1.99,writes.completion_s,2.46]
    freq = np.r_[2.5,writes.frequency_after_write_Hz,target]
    axes[0,0].step(t,freq,where='post',color=GREY)
    axes[0,0].plot(writes.completion_s,writes.frequency_after_write_Hz,'o',color=BLUE,ms=3)
    axes[0,1].step([1.8,2,5.2],[2.5,target,target],where='post',color=GREY,ls=':',label='Target')
    for name,color in [('S-Device',ORANGE),('N-Device',BLUE)]:
        d = hist[(hist.controller==name)&(hist.programming_budget==10)]
        w = d.time_s.between(1.8,5.2)
        axes[0,1].plot(d.time_s[w],d.motor_frequency_Hz[w],color=color,label=name)
        for budget,style in [(1,':'),(10,'-')]:
            w = hist[(hist.controller==name)&(hist.programming_budget==budget)&(hist.time_s<=8)]
            theta = 2*np.pi*(2.5*np.minimum(w.time_s,2)+target*np.maximum(w.time_s-2,0))
            assert np.max(np.abs(np.cos(theta)-w.q_target))<1e-10
            axes[1,0].plot(w.time_s,w.motor_phase_rad-theta,style,color=color,label=f'{name[0]}, budget {budget}')
        zoom = d.time_s.between(4.2,5.2)
        axes[1,1].plot(d.time_s[zoom],d.q_m[zoom],color=color,label=name)
    axes[1,1].plot(d.time_s[zoom],d.q_target[zoom],':',color=GREY,label='Target')
    for ax in (axes[0,1],axes[1,0]):
        ax.axvspan(4,4.2,color='.75',alpha=.3)
    for ax,letter,label,ylabel in zip(axes.flat,'ABCD',
            ['Prescribed DEP writes','Historical actuator frequency','Accumulated phase error','Displacement after the pulse'],
            ['Common frequency (Hz)','True frequency (Hz)','Motor − target phase (rad)','Normalized displacement'],
            strict=True):
        ax.set(xlabel='Time (s)',ylabel=ylabel)
        if letter!='A':
            ax.legend()
        panel(ax,letter,label)
    save(fig,'Figure_S1')

    d = csv(TABLES/'conductance_drift.csv')
    fig,axes = plt.subplots(2,2,figsize=(9,6.2),layout='constrained')
    for name,color in [('frequency_common',BLUE),('frequency_differential',ORANGE),('coupling_heterogeneous',GREEN)]:
        r = d[(d.scenario=='hold')&(d.end_s==20)&(d.pattern==name)&np.isclose(d.drift_at20_fraction,.02)].iloc[0]
        tr = csv(WORK/'sensitivity/runs'/r.run_id/'trace.csv.gz')
        for ax, series in zip(axes.flat,[tr.motor_true_Hz-tr.target_Hz,tr.theta_m_rad-tr.theta_target_rad,tr.Ephi_rad,tr.R],strict=True):
            ax.plot(tr.time_s,series,color=color,label=name.replace('_',' '),lw=1)
    for ax,letter,label,ylabel in zip(axes.flat,'ABCD',['Actuator frequency bias','Accumulated phase error','Phase-pattern error','Collective coherence'],
                                     ['True frequency − target (Hz)','Motor − target phase (rad)','Error (rad)',r'$R$'],strict=True):
        ax.set(xlabel='Time (s)',ylabel=ylabel)
        panel(ax,letter,label)
    axes[0,0].legend()
    axes[1,0].axhline(.05,ls=':',color=GREY)
    axes[1,1].set_ylim(.9978,1.0002)
    save(fig,'Figure_S2')


def reachability():
    d = WORK/'sensitivity'
    ref = DATA/'reference/sensitivity'
    a, b = [csv(p/'original_frequency_graph_vertices.csv') for p in (ref,d)]
    f = csv(TABLES/'programming_direction.csv')
    fig,axes = plt.subplots(1,2,figsize=(9.6,3.6),layout='constrained')
    for x,style in [(a,dict(facecolors='none',edgecolors=GREY,s=48,label='Archived')),
                    (b,dict(c=BLUE,marker='x',s=22,label='Reproduced'))]:
        g = x[x.forward_reachable_with_budget10]
        axes[0].scatter(g.total_online_writes,g.frequency_Hz,**style)
    axes[0].set(xlabel='Accepted online writes',ylabel='Reachable frequency (Hz)')
    axes[0].legend()
    for case,color,marker in [('up',ORANGE,'o'),('down',BLUE,'^')]:
        dcase = f[(f.end_s==8)&(f.controller=='J-frequency-both')&(f.scenario==case)]
        axes[1].plot(dcase.budget,dcase.final_common_Hz,marker=marker,color=color,label=f'{case} · controller')
        axes[1].axhline(dcase.frequency_target_Hz.iloc[0],ls=':',color=color,alpha=.65,label=f'{case} · target')
    axes[1].set(xlabel='Programming budget',ylabel='End frequency (Hz)')
    axes[1].legend()
    panel(axes[0],'A','Forward reachability from registered starts')
    panel(axes[1],'B','Matched POT/DEP direction tests')
    save(fig,'Compare_Reachability')


def cohort_plot():
    m = csv(TABLES/'Table_2.csv').sort_values(['evaluation_end_s','controller','mode']).reset_index(drop=True)
    ref = csv(DATA/'reference/supplement/Table_S4_cohort_summaries_full.csv')
    fig,axes = plt.subplots(1,2,figsize=(10.4,4.65),layout='constrained')
    labels = []
    for i,r in enumerate(m.itertuples()):
        old = ref[(ref.scenario==r.scenario)&(ref.controller==r.controller)&(ref['mode']==r.mode)].iloc[0]
        labels.append(f'{r.evaluation_end_s:g} s · {r.controller.removeprefix("Reassigned-")} · '
                      + ('I' if r.mode.startswith('independent') else 'C'))
        estimate = r.overall_success_n/r.n_runs
        interval = [[estimate-r.overall_success_CI95_low],[r.overall_success_CI95_high-estimate]]
        axes[0].errorbar(100*estimate,i,xerr=np.array(interval)*100,fmt='o',color=BLUE,capsize=3,
                         label='Rerun and Wilson 95% CI' if i==0 else None)
        axes[0].scatter(100*old.overall_n/old.n_runs,i,marker='o',facecolors='none',
                        edgecolors=ORANGE,s=68,label='Archived' if i==0 else None)
        axes[1].scatter(r.E_q_RMSE_over_Qscale_median,i,color=BLUE)
        axes[1].scatter(old.E_q_RMSE_over_Qscale_median,i,marker='o',facecolors='none',edgecolors=ORANGE,s=68)
    axes[0].set(yticks=range(len(m)),yticklabels=labels,xlabel='Overall success (%)')
    axes[0].invert_yaxis()
    axes[0].legend(loc='upper center',bbox_to_anchor=(.5,-.15),ncols=2)
    axes[1].set(yticks=range(len(m)),yticklabels=[],xlabel='Median normalized displacement RMSE')
    axes[1].invert_yaxis()
    panel(axes[0],'A','Probability estimates for each cohort')
    panel(axes[1],'B','Errors among initialized, completed runs')
    save(fig,'Compare_Stochastic')


def main():
    figure1b()
    calibration_comparison()
    phase_comparison()
    predictive_figures()
    supplementary()
    reachability()
    cohort_plot()
    print('Comparison, predictive, experimental and supplementary figures generated.')


if __name__=='__main__':
    main()
