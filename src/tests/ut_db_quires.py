from src.db_quiries_functions import *

def test_check_history():
    url = 'http://example.com/dataset'
    data_type = 'Дискретные'
    preference = 'Лучше, если будут отсутствовать некоторые существующие связи'

    # Test case 1: No new methods or alphas
    expected_new_params = None
    expected_status = 0
    expected_prev_launch_info = (None, None, None, None, 0.05, None, None, None)
    assert check_history(url, data_type, preference) == (expected_new_params, expected_status, expected_prev_launch_info)

    # Test case 2: New methods and alphas
    url = 'https://drive.google.com/file/d/1ONv7otvx6c0VzFE_JFiUOKRxONRLe-UY/view?usp=sharing'
    data_type = 'Категориальные'
    preference = 'Нет предпочтений'
    expected_new_params = (['gsq'], [0.2, 0.25, 0.3])
    expected_status = 1
    expected_prev_launch_info = (None, None, None, None, None, None, None, None)
    assert check_history(url, data_type, preference) == (expected_new_params, expected_status, expected_prev_launch_info)

    # Test case 3: Error retrieving dataset
    url = 'http://example.com/nonexistent'
    data_type = 'Вещественные'
    preference = 'Лучше, если будет присутствовать связи, которых на самом деле нет'
    expected_new_params = None
    expected_status = 2
    expected_prev_launch_info = None
    assert check_history(url, data_type, preference) == (expected_new_params, expected_status, expected_prev_launch_info)

test_check_history()
