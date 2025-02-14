#include <Solarflare/OpenOnload.h>

void execute_order(const HFTEvent& event) {
    IcebergOrder order;
    order.symbol = event.symbol;
    order.quantity = event.qty;
    order.mask_liquidity = true;
    order.slippage_tolerance = 0.005; // 0.5%
    
    // Colocated NY4 server
    send_to_exchange(order, "ny4");
}