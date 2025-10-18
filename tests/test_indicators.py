import pytest
from decimal import Decimal
from scripts.pipeline.generate_btc_ready import calc_max_future_price, calc_min_future_price, STEP_BACK_PERC, LOSS_PERC

# Tests para calc_max_future_price

def test_calc_max_future_price_simple_uptrend():
    """Caso simple: precio solo sube"""
    prices = [100, 110, 120]
    timestamps = [1, 2, 3]
    max_price, max_perc, max_date = calc_max_future_price(prices, timestamps)
    assert max_price == [120, 120, 120]
    assert pytest.approx(max_perc[0], 0.01) == 0.2  # (120/100 - 1)
    assert pytest.approx(max_perc[1], 0.01) == 0.090909  # (120/110 - 1)
    assert pytest.approx(max_perc[2], 0.01) == 0.0  # último trade
    assert max_date == [3, 3, 3]

def test_calc_max_future_price_with_step_back():
    """Caso STEP_BACK: precio sube y luego baja más del 1%"""
    prices = [100, 110, 120, 118]  # Baja de 120 a 118 = -1.67% > STEP_BACK_PERC
    timestamps = [1, 2, 3, 4]
    max_price, max_perc, max_date = calc_max_future_price(prices, timestamps)
    # Para el primer trade: sube hasta 120, luego baja a 118 (-1.67% desde 120)
    assert max_price[0] == 120
    assert pytest.approx(max_perc[0], 0.01) == 0.2  # (120/100 - 1)
    assert max_date[0] == 3

def test_calc_max_future_price_with_loss():
    """Caso LOSS: precio baja más del 0.5% desde el precio inicial"""
    prices = [100, 99, 98, 110]  # Baja de 100 a 99 = -1% > LOSS_PERC
    timestamps = [1, 2, 3, 4]
    max_price, max_perc, max_date = calc_max_future_price(prices, timestamps)
    # Para el primer trade: baja a 99 (-1% desde 100) que es > LOSS_PERC
    assert max_price[0] == 100  # Retorna el precio inicial
    assert max_perc[0] == 0.0
    assert max_date[0] == 1

def test_calc_max_future_price_no_future():
    """Caso último trade: no hay trades futuros"""
    prices = [100]
    timestamps = [1]
    max_price, max_perc, max_date = calc_max_future_price(prices, timestamps)
    assert max_price == [100]
    assert max_perc == [0.0]
    assert max_date == [1]

# Tests para calc_min_future_price

def test_calc_min_future_price_simple_downtrend():
    """Caso simple: precio solo baja"""
    prices = [100, 90, 80]
    timestamps = [1, 2, 3]
    min_price, min_perc, min_date = calc_min_future_price(prices, timestamps)
    assert min_price == [80, 80, 80]
    assert pytest.approx(min_perc[0], 0.01) == 0.25  # (100/80 - 1)
    assert pytest.approx(min_perc[1], 0.01) == 0.125  # (90/80 - 1)
    assert pytest.approx(min_perc[2], 0.01) == 0.0  # último trade
    assert min_date == [3, 3, 3]

def test_calc_min_future_price_with_step_back():
    """Caso STEP_BACK: precio baja y luego sube más del 1%"""
    prices = [100, 90, 80, 81]  # Sube de 80 a 81 = +1.25% > STEP_BACK_PERC
    timestamps = [1, 2, 3, 4]
    min_price, min_perc, min_date = calc_min_future_price(prices, timestamps)
    # Para el primer trade: baja hasta 80, luego sube a 81 (+1.25% desde 80)
    assert min_price[0] == 80
    assert pytest.approx(min_perc[0], 0.01) == 0.25  # (100/80 - 1)
    assert min_date[0] == 3

def test_calc_min_future_price_with_loss():
    """Caso LOSS: precio sube más del 0.5% desde el precio inicial"""
    prices = [100, 101, 102, 90]  # Sube de 100 a 101 = +1% > LOSS_PERC
    timestamps = [1, 2, 3, 4]
    min_price, min_perc, min_date = calc_min_future_price(prices, timestamps)
    # Para el primer trade: sube a 101 (+1% desde 100) que es > LOSS_PERC
    assert min_price[0] == 0.0
    assert min_perc[0] == 0.0
    assert min_date[0] == 1

def test_calc_min_future_price_real_case():
    """Caso real del usuario: 7195 -> 7180 -> 7200"""
    prices = [7195, 7180, 7200]
    timestamps = [1, 2, 3]
    min_price, min_perc, min_date = calc_min_future_price(prices, timestamps)
    # El mínimo futuro para el primer trade es 7180
    assert min_price[0] == 7180
    assert pytest.approx(min_perc[0], 0.0001) == (7195/7180 - 1)
    # El segundo trade: sube a 7200 (+0.28% desde 7180) que no alcanza STEP_BACK
    assert min_price[1] == 7180
    assert pytest.approx(min_perc[1], 0.0001) == 0.0
    # El último trade solo se compara consigo mismo
    assert min_price[2] == 7200
    assert pytest.approx(min_perc[2], 0.0001) == 0.0

def test_calc_min_future_price_no_future():
    """Caso último trade: no hay trades futuros"""
    prices = [100]
    timestamps = [1]
    min_price, min_perc, min_date = calc_min_future_price(prices, timestamps)
    assert min_price == [100]
    assert min_perc == [0.0]
    assert min_date == [1]

# Tests de edge cases

def test_max_future_price_exact_step_back():
    """Caso borde: bajada exacta de STEP_BACK_PERC"""
    initial = 100
    future_max = 110
    step_back = future_max * (1 - STEP_BACK_PERC)  # 110 * 0.99 = 108.9
    prices = [initial, future_max, step_back - 0.01]  # Justo por debajo del límite
    timestamps = [1, 2, 3]
    max_price, max_perc, max_date = calc_max_future_price(prices, timestamps)
    assert max_price[0] == future_max
    assert pytest.approx(max_perc[0], 0.01) == 0.1

def test_min_future_price_exact_step_back():
    """Caso borde: subida exacta de STEP_BACK_PERC"""
    initial = 100
    future_min = 90
    step_back = future_min * (1 + STEP_BACK_PERC)  # 90 * 1.01 = 90.9
    prices = [initial, future_min, step_back + 0.01]  # Justo por encima del límite
    timestamps = [1, 2, 3]
    min_price, min_perc, min_date = calc_min_future_price(prices, timestamps)
    assert min_price[0] == future_min
    assert pytest.approx(min_perc[0], 0.01) == (initial / future_min - 1)

