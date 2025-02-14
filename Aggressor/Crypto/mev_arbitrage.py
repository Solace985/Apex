from flashbots import FlashbotsAPI

def avoid_mev(tx):
    fb = FlashbotsAPI()
    simulated_tx = fb.simulate(tx)
    if simulated_tx.profit > 0:
        return fb.send_private_transaction(tx)
    else:
        cancel_transaction(tx)