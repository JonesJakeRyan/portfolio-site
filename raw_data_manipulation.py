import pandas as pd
from datetime import datetime
from collections import deque

# Load the CSV file
file_path = 'Webull_Orders_Records.csv'
df = pd.read_csv(file_path)
pd.set_option('display.max_columns', None)

print("=== LOADING AND CLEANING DATA ===")
print(f"Original dataset shape: {df.shape}")

# Remove any rows with missing Filled Time
df = df.dropna(subset=['Filled Time'])
print(f"After removing missing Filled Time: {df.shape}")

# Convert to datetime, handling BOTH EDT and EST timezones
df['Filled Time'] = pd.to_datetime(
    df['Filled Time'].str.replace(' EDT', '').str.replace(' EST', ''),
    format='%m/%d/%Y %H:%M:%S',
    errors='coerce'
)

# Remove any rows where datetime conversion failed
df = df.dropna(subset=['Filled Time'])
print(f"Final dataset shape after datetime conversion: {df.shape}")

# Sort by Symbol and Filled Time
df = df.sort_values(by=['Symbol', 'Filled Time'])


def calculate_fifo_pnl_with_daily_tracking(trades):
    """
    Calculate realized P&L using FIFO method and track daily realizations.
    Returns: (realized_pnl, current_position, unrealized_cost_basis, daily_realizations)
    """
    # Queue to track purchases (quantity, price, date)
    long_positions = deque()  # For regular purchases
    short_positions = deque()  # For short sales

    realized_pnl = 0
    current_net_position = 0
    daily_realizations = []  # List of (date, symbol, realized_pnl_amount)

    for trade in trades:
        side = trade['side']
        quantity = trade['quantity']
        price = trade['price']
        trade_date = trade['date']
        symbol = trade['symbol']

        if side == 'Buy':
            current_net_position += quantity

            # First, close out any short positions using FIFO
            temp_quantity = quantity
            while temp_quantity > 0 and short_positions:
                short_qty, short_price, short_date = short_positions.popleft()

                if temp_quantity >= short_qty:
                    # Close entire short position
                    pnl_amount = short_qty * (short_price - price)
                    realized_pnl += pnl_amount
                    daily_realizations.append((trade_date, symbol, pnl_amount))
                    temp_quantity -= short_qty
                else:
                    # Partially close short position
                    pnl_amount = temp_quantity * (short_price - price)
                    realized_pnl += pnl_amount
                    daily_realizations.append((trade_date, symbol, pnl_amount))
                    short_positions.appendleft((short_qty - temp_quantity, short_price, short_date))
                    temp_quantity = 0

            # Add remaining quantity as long position
            if temp_quantity > 0:
                long_positions.append((temp_quantity, price, trade_date))

        elif side == 'Sell':
            current_net_position -= quantity

            # First, close out long positions using FIFO
            temp_quantity = quantity
            while temp_quantity > 0 and long_positions:
                long_qty, long_price, long_date = long_positions.popleft()

                if temp_quantity >= long_qty:
                    # Close entire long position
                    pnl_amount = long_qty * (price - long_price)
                    realized_pnl += pnl_amount
                    daily_realizations.append((trade_date, symbol, pnl_amount))
                    temp_quantity -= long_qty
                else:
                    # Partially close long position
                    pnl_amount = temp_quantity * (price - long_price)
                    realized_pnl += pnl_amount
                    daily_realizations.append((trade_date, symbol, pnl_amount))
                    long_positions.appendleft((long_qty - temp_quantity, long_price, long_date))
                    temp_quantity = 0

            # Add remaining quantity as short position
            if temp_quantity > 0:
                short_positions.append((temp_quantity, price, trade_date))

        elif side == 'Short':
            current_net_position -= quantity
            short_positions.append((quantity, price, trade_date))

    # Calculate unrealized cost basis
    unrealized_cost_basis = 0
    for qty, price, date in long_positions:
        unrealized_cost_basis += qty * price
    for qty, price, date in short_positions:
        unrealized_cost_basis -= qty * price  # Short positions have negative cost basis

    return realized_pnl, current_net_position, unrealized_cost_basis, daily_realizations


# Process each symbol
print("\n=== CALCULATING FIFO-BASED REALIZED P&L ===")
print("=" * 60)

symbol_results = {}
total_realized_pnl = 0
all_daily_realizations = []  # Collect all daily realizations across symbols

for symbol, group in df.groupby('Symbol'):
    trades = []

    # Convert to our trade format
    for _, trade in group.iterrows():
        trades.append({
            'side': trade['Side'],
            'quantity': trade['Filled'],
            'price': trade['Avg Price'],
            'date': trade['Filled Time'],
            'symbol': symbol
        })

    # Calculate FIFO P&L with daily tracking
    realized_pnl, current_position, unrealized_cost, daily_realizations = calculate_fifo_pnl_with_daily_tracking(trades)

    # Add this symbol's daily realizations to the master list
    all_daily_realizations.extend(daily_realizations)

    symbol_results[symbol] = {
        'realized_pnl': realized_pnl,
        'current_position': current_position,
        'unrealized_cost': unrealized_cost,
        'trades': trades,
        'total_trades': len(trades),
        'daily_realizations': daily_realizations
    }

    total_realized_pnl += realized_pnl

print(f"🎯 TOTAL REALIZED P&L (FIFO Method): ${total_realized_pnl:.2f}")
print()

# Sort by realized P&L
sorted_symbols = sorted(symbol_results.items(), key=lambda x: x[1]['realized_pnl'], reverse=True)

print("REALIZED P&L BY SYMBOL (includes closed portions of open positions):")
print("-" * 70)

total_profits = 0
total_losses = 0
profit_count = 0
loss_count = 0

for symbol, data in sorted_symbols:
    realized_pnl = data['realized_pnl']
    current_position = data['current_position']

    if realized_pnl > 0:
        status = "📈 PROFIT"
        total_profits += realized_pnl
        profit_count += 1
    elif realized_pnl < 0:
        status = "📉 LOSS"
        total_losses += realized_pnl
        loss_count += 1
    else:
        status = "➖ BREAKEVEN"

    position_status = f"({current_position:+} shares)" if current_position != 0 else "(CLOSED)"

    print(f"{symbol:8} ${realized_pnl:+8.2f} {status:10} {position_status}")

print(f"\n💰 Total Realized Profits: ${total_profits:.2f} ({profit_count} symbols)")
print(f"💸 Total Realized Losses: ${total_losses:.2f} ({loss_count} symbols)")
if profit_count + loss_count > 0:
    print(f"🏆 Win Rate: {profit_count / (profit_count + loss_count) * 100:.1f}%")

# Detailed breakdown for significant P&L
print(f"\n=== DETAILED BREAKDOWN (P&L > $50 or < -$50) ===")

for symbol, data in sorted_symbols:
    if abs(data['realized_pnl']) > 50:
        print(f"\n{symbol} - REALIZED P&L: ${data['realized_pnl']:.2f}")
        print(f"  📊 Current position: {data['current_position']} shares")
        print(f"  📝 Total trades: {data['total_trades']}")

        if data['unrealized_cost'] != 0:
            print(f"  💼 Unrealized cost basis: ${data['unrealized_cost']:.2f}")

        # Show some trade examples
        trades = data['trades']
        if len(trades) <= 8:
            print("  📈 All trades:")
            for trade in trades:
                print(
                    f"    {trade['date'].strftime('%m/%d')} {trade['side']:5} {trade['quantity']:3} @ ${trade['price']:6.2f}")
        else:
            print("  📈 First 4 trades:")
            for trade in trades[:4]:
                print(
                    f"    {trade['date'].strftime('%m/%d')} {trade['side']:5} {trade['quantity']:3} @ ${trade['price']:6.2f}")
            print("  📈 Last 4 trades:")
            for trade in trades[-4:]:
                print(
                    f"    {trade['date'].strftime('%m/%d')} {trade['side']:5} {trade['quantity']:3} @ ${trade['price']:6.2f}")

# Summary of open vs closed positions
print(f"\n=== POSITION STATUS SUMMARY ===")
fully_closed = sum(1 for data in symbol_results.values() if data['current_position'] == 0)
open_positions = sum(1 for data in symbol_results.values() if data['current_position'] != 0)

print(f"📋 Fully closed positions: {fully_closed}")
print(f"📈 Positions with open shares: {open_positions}")
print(f"💡 Note: Realized P&L includes profits/losses from closed portions of open positions")

print(f"\n" + "=" * 60)
print(f"🎯 FINAL REALIZED P&L: ${total_realized_pnl:.2f}")
print(f"📊 This includes ALL completed buy/sell cycles")
print(f"📋 Using FIFO (First-In-First-Out) matching method")
print("=" * 60)

# Show current open positions
print(f"\n=== CURRENT OPEN POSITIONS ===")
print("(These have unrealized P&L not included above)")
total_unrealized_cost = 0

for symbol, data in symbol_results.items():
    if data['current_position'] != 0:
        position = data['current_position']
        cost_basis = data['unrealized_cost']
        total_unrealized_cost += cost_basis

        position_type = "LONG" if position > 0 else "SHORT"
        avg_price = abs(cost_basis / position) if position != 0 else 0

        print(f"{symbol:8} {position:+4} shares {position_type:5} (avg cost: ${avg_price:.2f})")

print(f"\n💼 Total unrealized cost basis: ${total_unrealized_cost:.2f}")

# =================================================================
# CREATE CSV EXPORTS
# =================================================================
print(f"\n=== CREATING CSV EXPORTS ===")

# 1. CREATE DAILY ACCOUNT SUMMARY CSV
from collections import defaultdict

# Group daily realizations by date
daily_totals = defaultdict(float)
for date, symbol, pnl_amount in all_daily_realizations:
    date_str = date.strftime('%Y-%m-%d')
    daily_totals[date_str] += pnl_amount

# Create daily summary DataFrame
daily_summary_data = []
cumulative_pnl = 0

for date_str in sorted(daily_totals.keys()):
    daily_pnl = daily_totals[date_str]
    cumulative_pnl += daily_pnl
    daily_summary_data.append({
        'Date': date_str,
        'Daily_Realized_PnL': round(daily_pnl, 2),
        'Cumulative_Realized_PnL': round(cumulative_pnl, 2)
    })

daily_summary_df = pd.DataFrame(daily_summary_data)

# Export daily summary
daily_summary_df.to_csv('account_summary.csv', index=False)
print(f"✅ Exported daily P&L to 'account_summary.csv' ({len(daily_summary_df)} days)")

# 2. CREATE CURRENT POSITIONS CSV
current_positions_data = []

for symbol, data in symbol_results.items():
    if data['current_position'] != 0:
        position = data['current_position']
        cost_basis = data['unrealized_cost']
        avg_price = abs(cost_basis / position) if position != 0 else 0
        position_type = "LONG" if position > 0 else "SHORT"

        # Get first and last trade dates for this symbol
        trade_dates = [trade['date'] for trade in data['trades']]
        first_trade = min(trade_dates).strftime('%Y-%m-%d')
        last_trade = max(trade_dates).strftime('%Y-%m-%d')

        current_positions_data.append({
            'Symbol': symbol,
            'Current_Position': position,
            'Position_Type': position_type,
            'Unrealized_Cost_Basis': round(cost_basis, 2),
            'Average_Price': round(avg_price, 2),
            'Total_Trades': data['total_trades'],
            'Realized_PnL': round(data['realized_pnl'], 2),
            'First_Trade_Date': first_trade,
            'Last_Trade_Date': last_trade
        })

# Sort by absolute cost basis (largest positions first)
current_positions_data.sort(key=lambda x: abs(x['Unrealized_Cost_Basis']), reverse=True)

current_positions_df = pd.DataFrame(current_positions_data)

# 3. CREATE CLOSED POSITIONS (BY SYMBOL SUMMARY) CSV
#    Contains one row per symbol with total trades and realized P&L.
#    Note: includes ALL symbols traded, regardless of whether a position is currently open.
closed_positions_data = []
for symbol, data in symbol_results.items():
    closed_positions_data.append({
        'Symbol': symbol,
        'Total_Trades': data['total_trades'],
        'Realized_PnL': round(data['realized_pnl'], 2),
        'Current_Position': data['current_position'],
        'Position_Status': 'CLOSED' if data['current_position'] == 0 else 'OPEN'
    })

# Sort by realized P&L (descending)
closed_positions_data.sort(key=lambda x: x['Realized_PnL'], reverse=True)

closed_positions_df = pd.DataFrame(closed_positions_data)
closed_positions_df.to_csv('closed_positions.csv', index=False)
print(f"✅ Exported symbol summary to 'closed_positions.csv' ({len(closed_positions_df)} symbols)")


# Export current positions
current_positions_df.to_csv('current_positions.csv', index=False)
print(f"✅ Exported open positions to 'current_positions.csv' ({len(current_positions_df)} positions)")

print(f"\n📁 CSV Files Created:")
print(f"   • account_summary.csv - Daily realized P&L summary")
print(f"   • current_positions.csv - Current open positions with details")
print(f"   • closed_positions.csv - Per-symbol summary (trades, realized P&L, open/closed)")
