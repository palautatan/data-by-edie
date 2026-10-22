"""Validate imported ChatGPT coding, describe standouts, persist dashboard tables."""
import hashlib
from pathlib import Path
import pandas as pd
import duckdb

METRICS=['approval','winner_support','genuine','trustworthy','entertaining','winner_prediction']

def build_and_save(root):
    root=Path(root); pieces=[]
    for path in sorted((root/'data/openai').glob('*__community_perceptions.csv')):
        df=pd.read_csv(path,dtype={'episode':'string'})
        required={'islander','episode','evidence_count','confidence',*METRICS}
        if not required.issubset(df): raise ValueError(f'Missing columns: {path}')
        df['source_file']=path.name
        df['source_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
        pieces.append(df)
    if not pieces: raise ValueError('No community perception files found')
    raw=pd.concat(pieces,ignore_index=True)
    if raw[['islander','episode']].isna().any().any(): raise ValueError('Missing source identity')
    if raw.duplicated(['islander','episode']).any(): raise ValueError('Duplicate islander/episode assessments')
    for col in METRICS:
        raw[col]=pd.to_numeric(raw[col],errors='raise').astype('Int64')
        if not raw[col].dropna().isin([0,1]).all(): raise ValueError(f'Invalid binary values: {col}')
    raw['evidence_count']=pd.to_numeric(raw.evidence_count,errors='raise').astype('Int64')
    if raw.evidence_count.isna().any() or raw.evidence_count.lt(0).any(): raise ValueError('Invalid evidence count')
    raw['scope']=raw.episode.eq('overall').map({True:'overall',False:'episode'})
    raw['episode']=pd.to_numeric(raw.episode.mask(raw.scope.eq('overall')),errors='raise').astype('Int64')
    raw=raw.rename(columns={'islander':'target_islander'})
    raw['assessment_source']='ChatGPT manual upload; sampled comments'
    raw['coding_note']='1=supports perception; 0=opposes perception; NULL=unclear/not discussed. User-confirmed definitions. Not audience percentages.'
    with duckdb.connect(str(root/'data/love_island_reddit.duckdb'),read_only=True) as con:
        expected=con.execute('SELECT target_islander,episode,num_comments FROM dashboard.episode_relative_metrics').df()
        overall_metrics=con.execute('SELECT target_islander,num_comments,roberta_mean,toxicity_mean FROM dashboard.overall_relative_metrics').df()
    if not set(raw.target_islander).issubset(set(overall_metrics.target_islander)): raise ValueError('Unknown islander names')
    sample_dir=root/'tables/sentiment_dashboard/qualitative_samples'
    samples=pd.read_csv(sample_dir/'episode_coverage.csv').rename(columns={'islander':'target_islander'})
    manifest=pd.read_csv(sample_dir/'manifest.csv').rename(columns={'islander':'target_islander','sha256':'sample_sha256','filename':'sample_file'})
    overall=raw[raw.scope.eq('overall')].merge(overall_metrics,on='target_islander',validate='one_to_one')
    overall=overall.merge(manifest[['target_islander','sampled_comments','sampled_characters','sample_sha256','sample_file','seed']],on='target_islander',validate='one_to_one')
    episode=raw[raw.scope.eq('episode')].merge(expected,on=['target_islander','episode'],how='left',validate='one_to_one')
    episode=episode.merge(samples[['target_islander','episode','sampled_comments','sampled_characters']],on=['target_islander','episode'],how='left',validate='one_to_one')
    overall['evidence_count_matches_sample']=overall.evidence_count.eq(overall.sampled_comments)
    episode['evidence_count_matches_sample']=episode.evidence_count.eq(episode.sampled_comments)
    # Audit expected and returned scope/episode keys, including unexpected rows.
    keys=['target_islander','scope','episode']
    expected_all=pd.concat([expected[['target_islander','episode']].assign(scope='episode'),
                           overall_metrics[['target_islander']].assign(scope='overall',episode=pd.NA)],ignore_index=True)
    expected_all.episode=expected_all.episode.astype('Int64')
    coverage=expected_all.merge(raw[keys],on=keys,how='outer',indicator=True,validate='one_to_one')
    coverage['status']=coverage.pop('_merge').map({'left_only':'missing_output','right_only':'unexpected_output','both':'present'}).astype(str)
    # Retain missing values; each episode counts once, never evidence-count weighted.
    records=[]
    for name, group in episode.groupby('target_islander'):
        for metric in METRICS:
            codes=group[metric]; assessed=int(codes.notna().sum()); ones=int(codes.eq(1).sum())
            records.append(dict(target_islander=name,metric=metric,episodes_returned=len(group),
                episodes_coded_1=ones,episodes_coded_0=int(codes.eq(0).sum()),episodes_unknown=int(codes.isna().sum()),
                episodes_assessed=assessed,share_assessed_episodes_coded_1=ones/assessed if assessed else float('nan')))
    consistency=pd.DataFrame(records)
    consistency['rank_within_metric']=consistency.groupby('metric').share_assessed_episodes_coded_1.rank(method='min',ascending=False).astype('Int64')
    highlights=[]
    for metric in METRICS:
        for row in overall.loc[overall[metric].eq(1)].itertuples():
            highlights.append(dict(metric=metric,target_islander=row.target_islander,overall_code=1,
                confidence=row.confidence,evidence_count=row.evidence_count,sampled_comments=row.sampled_comments))
    standouts=pd.DataFrame(highlights)
    overall['overall_columns_coded_1']=overall[METRICS].eq(1).sum(axis=1)
    overall['overall_columns_assessed']=overall[METRICS].notna().sum(axis=1)
    tables={'overall_community_perceptions':overall,'episode_community_perceptions':episode,
            'community_perception_standouts':standouts,'community_perception_episode_consistency':consistency,
            'community_perception_coverage':coverage}
    out=root/'tables/sentiment_dashboard'
    with duckdb.connect(str(root/'data/love_island_reddit.duckdb')) as con:
        con.execute('CREATE SCHEMA IF NOT EXISTS dashboard'); con.execute('BEGIN')
        try:
            for name,table in tables.items():
                con.register('export_frame',table)
                con.execute(f'CREATE OR REPLACE TABLE dashboard.{name} AS SELECT * FROM export_frame')
                saved=con.execute(f'SELECT * FROM dashboard.{name}').df()
                pd.testing.assert_frame_equal(saved,table.reset_index(drop=True),check_dtype=False)
                con.unregister('export_frame')
            con.execute('COMMIT')
        except Exception:
            con.execute('ROLLBACK'); raise
    for name,table in tables.items(): table.to_csv(out/f'{name}.csv',index=False,encoding='utf-8')
    return tables
