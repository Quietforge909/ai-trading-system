"""Recalculate the published numbers from saved simulation records.

No model inference or new backtest is run. Drawdowns come from saved episode
summaries; full equity paths and sampled coin identities are not in these files.
"""
import csv
from datetime import datetime,timedelta
import json
import math
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[1]


def near(a,b):
    assert math.isclose(a,b,rel_tol=1e-9,abs_tol=1e-8),(a,b)


def verify(path):
    d=json.loads(path.read_text());rows=d['per_episode'];reported=d['aggregate'];pnls=[];actions={}
    assert len(rows)==reported['episodes']==200 and d['fee_rate_per_side']==.005
    for e in rows:
        assert datetime.fromisoformat(e['end'])-datetime.fromisoformat(e['start'])+timedelta(hours=1)==timedelta(days=30)
        assert e['coins']==20 and len(e['trade_pnls'])==len(e['trade_forced'])==e['manual_sells']+e['forced_sells']
        near(sum(e['trade_pnls']),e['return']*d['starting_cash_per_episode_usd'])
        assert sum(e['trade_forced'])==e['forced_sells']
        pnls.extend(e['trade_pnls'])
        for k,v in e['actions'].items():actions[k]=actions.get(k,0)+v
    profit=sum(x for x in pnls if x>0);loss=-sum(x for x in pnls if x<0)
    calculated={'mean_return':statistics.mean(e['return'] for e in rows),'median_return':statistics.median(e['return'] for e in rows),
        'profitable_episode_rate':sum(e['return']>0 for e in rows)/len(rows),'worst_max_drawdown':min(e['max_drawdown'] for e in rows),
        'profit_factor':profit/loss,'net_pnl_dollars':sum(pnls),'gross_profit_dollars':profit,'gross_loss_dollars':loss,
        'expectancy_dollars':statistics.mean(pnls),'win_rate':sum(x>0 for x in pnls)/len(pnls),'trades':len(pnls),
        'manual_sells':sum(e['manual_sells'] for e in rows),'forced_sells':sum(e['forced_sells'] for e in rows)}
    for k,v in calculated.items():near(v,reported[k])
    assert actions==reported['actions'] and sum(actions.values())==200*720
    assert d['policy']['stochastic'] is False and d['policy']['seed']==42
    return d


a=verify(ROOT/'results/v344-200episodes.json');b=verify(ROOT/'results/v345-200episodes.json')
assert a['policy']==b['policy'] and a['requested_period']==b['requested_period']
assert [(e['start'],e['end'],e['coins']) for e in a['per_episode']]==[(e['start'],e['end'],e['coins']) for e in b['per_episode']]
recent=json.loads((ROOT/'results/v344-later-30days.json').read_text());s=recent['summary']
assert len(recent['trades'])==s['trades']==4
near(sum(t['pnl'] for t in recent['trades']),s['pnl']);near(s['pnl']/recent['starting_cash_usd'],s['return'])
near(sum(max(t['pnl'],0) for t in recent['trades']),s['gross_profit'])
near(-sum(min(t['pnl'],0) for t in recent['trades']),s['gross_loss'])
coverage=json.loads((ROOT/'data-sample/coverage.json').read_text());symbols=coverage['per_symbol']
assert len(symbols)==coverage['symbols']==265 and len({s['symbol'] for s in symbols})==265
assert sum(s['rows'] for s in symbols)==coverage['rows']==8880723
assert min(s['first_timestamp'] for s in symbols if s['first_timestamp'])==coverage['first_timestamp']
assert max(s['last_timestamp'] for s in symbols if s['last_timestamp'])==coverage['last_timestamp']
with (ROOT/'data-sample/BTCUSDT-first24h.csv').open() as f:candles=list(csv.DictReader(f))
assert len(candles)==24
for i,c in enumerate(candles):
    assert float(c['low'])<=min(float(c['open']),float(c['close']))<=max(float(c['open']),float(c['close']))<=float(c['high'])
    if i:assert datetime.fromisoformat(c['timestamp'])-datetime.fromisoformat(candles[i-1]['timestamp'])==timedelta(hours=1)
print(json.dumps({'saved_backtests_checked':3,'sampled_episodes_checked':400,'same_saved_episode_dates_and_coin_counts':True,
                  'data_coverage_rows':coverage['rows'],'data_sample_candles':24,'new_training_or_backtest_run':False},indent=2))
