"""Scripted accounting examples using the unchanged original Portfolio class.

No policy inference, exchange client, network, private data or credentials.
"""
import json
from rl.env.portfolio import Portfolio


def scenario(exit_price):
    portfolio=Portfolio(starting_cash=1000.,fee_rate=.005,fee_round_decimals=1)
    portfolio.execute_buy(price=100.,step=0,allocation=.5)
    after_buy={'cash':portfolio.state.cash,'units':portfolio.state.position_size,'cost_basis':portfolio.state.entry_cost_basis}
    pnl=portfolio.execute_sell(price=exit_price)
    assert not portfolio.state.has_position
    assert abs(portfolio.state.cash-(1000.+pnl))<1e-8
    return {'scripted_buy_price':100.,'scripted_sell_price':exit_price,'allocation':.5,
            'after_buy':after_buy,'realized_pnl_after_fees':round(pnl,8),'ending_cash':round(portfolio.state.cash,8)}


def main():
    result={'kind':'illustrative_original_accounting_not_AI_performance','network_calls':0,'trained_policy_loaded':False,
            'rising_price_example':scenario(110.),'falling_price_example':scenario(90.)}
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
