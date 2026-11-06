"""Exploratory final-exit survival and chronological next-episode hazard model."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import duckdb
from lifelines import KaplanMeierFitter
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score, average_precision_score

FEATURES=['episode','tenure','log_comments','roberta_mean']
VERSION='final_exit_logistic_v1'

def run_modeling(root):
    root=Path(root); out=root/'tables/survival_dashboard';out.mkdir(parents=True,exist_ok=True)
    cast=pd.read_csv(root/'data/metadata/islander_metadata.csv').rename(columns={'islander':'target_islander'})
    assert cast.target_islander.is_unique
    cast['event_observed']=cast.final_placement.fillna('').str.strip().eq('').astype(int)
    cast['duration_episodes']=cast.exit_episode-cast.entrance_episode+1
    assert cast.duration_episodes.gt(0).all()
    with duckdb.connect(str(root/'data/love_island_reddit.duckdb'),read_only=True) as con:
        schedule=con.execute('SELECT episode, MIN(created_at) AS discussion_start_proxy FROM comments WHERE episode IS NOT NULL GROUP BY episode ORDER BY episode').df()
        scores=con.execute('''SELECT w.window_id,w.comment_id,w.target_islander,w.episode,w.created_at,s.roberta_sentiment
          FROM islander_sentiment_windows w LEFT JOIN roberta_window_scores s
          ON w.window_id=s.window_id AND w.model_text=s.model_text
          WHERE w.keep_analysis='Y' ''').df()
    assert scores.window_id.is_unique
    assert scores.roberta_sentiment.notna().all()
    assert schedule.episode.tolist()==list(range(1,36))
    schedule['prediction_cutoff']=schedule.discussion_start_proxy.shift(-1)
    schedule['cutoff_method']='Earliest observed comment in next episode discussion; retrospective proxy, not verified airtime'
    assert schedule.discussion_start_proxy.is_monotonic_increasing
    timed=scores.merge(schedule[['episode','prediction_cutoff']],on='episode',validate='many_to_one')
    timed['available']=timed.created_at.lt(timed.prediction_cutoff)
    audit=timed.groupby('episode',as_index=False).agg(source_windows=('window_id','size'),available_windows=('available','sum'))
    audit['excluded_windows']=audit.source_windows-audit.available_windows
    # Average repeated target mentions within each independent comment first.
    eligible=timed[timed.available]
    comments=eligible.groupby(['target_islander','episode','comment_id'],as_index=False).agg(
        sentiment=('roberta_sentiment','mean'),latest_comment=('created_at','max'))
    features=comments.groupby(['target_islander','episode'],as_index=False).agg(
        num_comments=('comment_id','size'),roberta_mean=('sentiment','mean'),latest_feature_timestamp=('latest_comment','max'))
    rows=[]
    for r in cast.itertuples():
        # Conditional on no FINAL exit by end of t; temporary absences not modeled.
        for ep in range(int(r.entrance_episode),min(int(r.exit_episode),35)):
            rows.append(dict(target_islander=r.target_islander,episode=ep,target_episode=ep+1,
                tenure=ep-int(r.entrance_episode)+1,
                final_exit_next_episode=int(r.event_observed and int(r.exit_episode)==ep+1)))
    panel=pd.DataFrame(rows).merge(features,on=['target_islander','episode'],how='left',validate='one_to_one')
    panel=panel.merge(schedule[['episode','prediction_cutoff']],on='episode',validate='many_to_one')
    panel.num_comments=panel.num_comments.fillna(0).astype(int)
    panel['log_comments']=np.log1p(panel.num_comments)
    panel['model_version']=VERSION
    assert panel.final_exit_next_episode.sum()==cast.event_observed.sum()
    assert (panel.latest_feature_timestamp.dropna()<panel.loc[panel.latest_feature_timestamp.notna(),'prediction_cutoff']).all()
    assert not panel.duplicated(['target_islander','episode']).any()
    def model():
        return make_pipeline(SimpleImputer(strategy='constant',fill_value=0,add_indicator=True),
                             StandardScaler(),LogisticRegression(C=0.25,max_iter=2000))
    predictions=[]; folds=[]
    # Outcomes through episode t are known before predicting t+1.
    for ep,test in panel.groupby('episode',sort=True):
        train=panel[panel.target_episode.le(ep)]
        events=int(train.final_exit_next_episode.sum())
        if events<3 or train.final_exit_next_episode.nunique()<2:
            folds.append(dict(prediction_episode=ep,training_rows=len(train),training_events=events,status='warmup_no_prediction'))
            continue
        clf=model().fit(train[FEATURES],train.final_exit_next_episode)
        pred=test.copy();pred['exit_probability']=clf.predict_proba(test[FEATURES])[:,1]
        pred['survive_next_episode_probability']=1-pred.exit_probability
        pred['baseline_exit_probability']=(events+1)/(len(train)+2)
        pred['training_last_outcome_episode']=int(train.target_episode.max())
        pred['training_rows']=len(train);pred['training_events']=events
        pred['prediction_type']='chronological_out_of_time'
        predictions.append(pred)
        folds.append(dict(prediction_episode=ep,training_rows=len(train),training_events=events,status='predicted'))
    predictions=pd.concat(predictions,ignore_index=True)
    assert predictions.training_last_outcome_episode.le(predictions.episode).all()
    metrics=[]
    for name,col in [('logistic','exit_probability'),('training_rate_baseline','baseline_exit_probability')]:
        y=predictions.final_exit_next_episode;p=predictions[col]
        metrics.append(dict(model=name,n=len(y),events=int(y.sum()),brier=brier_score_loss(y,p),
            log_loss=log_loss(y,p,labels=[0,1]),roc_auc=roc_auc_score(y,p),average_precision=average_precision_score(y,p)))
    predictions['calibration_bin']=pd.cut(predictions.exit_probability,[0,.05,.10,.20,.40,.60,1],include_lowest=True).astype(str)
    calibration=predictions.groupby('calibration_bin',as_index=False).agg(
        n=('exit_probability','size'),predicted=('exit_probability','mean'),observed=('final_exit_next_episode','mean'))
    # Descriptive tenure survival: administrative right censoring of finalists.
    km=KaplanMeierFitter().fit(cast.duration_episodes,event_observed=cast.event_observed,label='survival')
    survival=km.survival_function_.join(km.confidence_interval_).join(km.event_table).reset_index().rename(columns={'index':'tenure_episodes'})
    survival=survival.rename(columns={survival.columns[0]:'tenure_episodes'})
    survival['estimate_type']='descriptive_full_season_not_forecast'
    # Full-data coefficients are descriptive; never substituted for held-out predictions.
    full=model().fit(panel[FEATURES],panel.final_exit_next_episode)
    names=full.named_steps['simpleimputer'].get_feature_names_out(FEATURES)
    coefficients=pd.DataFrame({'feature':names,'standardized_log_odds_coefficient':full.named_steps['logisticregression'].coef_[0]})
    coefficients['fit_type']='full_season_descriptive';coefficients['C']=0.25
    tables={'final_exit_subjects':cast,'final_exit_risk_panel':panel,'final_exit_predictions':predictions,
            'final_exit_model_metrics':pd.DataFrame(metrics),'final_exit_calibration':calibration,
            'final_exit_survival_curve':survival,'final_exit_feature_audit':audit,
            'final_exit_prediction_cutoffs':schedule,'final_exit_folds':pd.DataFrame(folds),
            'final_exit_coefficients':coefficients}
    with duckdb.connect(str(root/'data/love_island_reddit.duckdb')) as con:
        con.execute('BEGIN')
        try:
            for name,table in tables.items():
                con.register('model_export',table)
                con.execute(f'CREATE OR REPLACE TABLE dashboard.{name} AS SELECT * FROM model_export')
                assert con.execute(f'SELECT count(*) FROM dashboard.{name}').fetchone()[0]==len(table)
                con.unregister('model_export')
            con.execute('COMMIT')
        except Exception:con.execute('ROLLBACK');raise
    for name,table in tables.items():table.to_csv(out/f'{name}.csv',index=False)
    config=dict(version=VERSION,features=FEATURES,C=.25,event='final exit excluding censored finalists',
        cutoff='earliest observed next-episode comment, unverified retrospective proxy',
        risk_set='between recorded entrance and final exit; temporary absence spells unavailable',
        limitations=['one season, 34 subjects, clustered group/couple exits',
          'historical expanding-window evaluation conditional on recorded final-exit risk set',
          'episode index includes existing specials; not a verified dumping-opportunity calendar',
          'no future multi-episode individual survival forecast; only one-step survival and descriptive KM',
          'KM independent-censoring assumption may not hold for strategic mid-season entry; descriptive only'])
    (out/'model_config.json').write_text(json.dumps(config,indent=2))
    print(pd.DataFrame(metrics).to_string(index=False))
    print(f'{len(panel)} risk rows, {len(predictions)} held-out predictions, {int(audit.loc[audit.episode.lt(35),"excluded_windows"].sum())} late windows excluded before finale.')
    return tables
