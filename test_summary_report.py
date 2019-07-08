import helpers
from create_summary import create_summary
import numpy as np
import numpy.testing as npt
import pandas.util.testing as pdt
import pandas as pd
import pytest


@pytest.fixture
def orders():
    params = {"size": 5,
              "order_id_range": 70,
              "customer_id_range": 20000,
              "datetime_range": 1095,
              "start_date": "2017-01-01"}

    return helpers.orders_mock_generator(**params)


@pytest.fixture
def order_lines():
    params = {"size": 5,
              "product_id_range": 10000,
              "order_id_range": 70,
              "price_range": [6.0, 5000.0]}

    return helpers.order_lines_mock_generator(**params)


def test_random_date_generator():
    output_date = helpers.random_date_generator(5, "2017-01-01", 1095)
    expected_date = np.array(['2019-05-11', '2019-11-11', '2017-05-02', '2018-04-12',
                              '2017-11-27'], dtype='datetime64[D]')

    npt.assert_array_equal(output_date, expected_date)


def test_orders_mock_generator(orders):
    expected_dataframe = pd.DataFrame({'OrderId': np.array([51, 14, 60, 20, 23]),
                                       'CustomerId': np.array([8322, 1685, 769, 2433, 5311]),
                                       'DateTime': np.array(['2019-05-11', '2019-11-11', '2017-05-02', '2018-04-12',
                                                             '2017-11-27'], dtype='datetime64[D]')})

    pdt.assert_frame_equal(orders, expected_dataframe)


def test_order_lines_mock_generator(order_lines):
    expected_dataframe = pd.DataFrame({'ProductId': np.array([7270, 860, 5390, 5191, 5734]),
                                       'OrderId': np.array([23, 2, 21, 52, 1]),
                                       'Price': np.array(
                                           [3611.661869, 4693.132229, 9.889157, 4961.104527, 3089.702659])})

    pdt.assert_frame_equal(order_lines, expected_dataframe)


def test_check_orders_schema(orders):
    expected_dtypes = {'OrderId': np.dtype('int64'),
                       'CustomerId': np.dtype('int64'),
                       'DateTime': np.dtype('<M8[ns]')}

    assert dict(orders.dtypes) == expected_dtypes


def test_check_order_lines_schema(order_lines):
    expected_dtypes = {'ProductId': np.dtype('int64'),
                       'OrderId': np.dtype('int64'),
                       'Price': np.dtype('float64')}

    assert dict(order_lines.dtypes) == expected_dtypes


def test_select_last(orders):
    output_dataframe = helpers.select_last(orders, 'DateTime', '1M')

    expected_dataframe = pd.DataFrame(data=[[14, 1685]],
                                      columns=['OrderId', 'CustomerId'],
                                      index=pd.to_datetime(["2019-11-11"]))
    expected_dataframe.index.name = 'DateTime'

    pdt.assert_frame_equal(output_dataframe, expected_dataframe)


def test_combine_dataframes():
    order_params = {"size": 70,
                    "order_id_range": 70,
                    "customer_id_range": 20000,
                    "datetime_range": 1095,
                    "start_date": "2017-01-01"}

    orders = helpers.orders_mock_generator(**order_params)

    order_line_params = {"size": 70,
                         "product_id_range": 10000,
                         "order_id_range": 70,
                         "price_range": [6.0, 5000.0]}

    order_lines = helpers.order_lines_mock_generator(**order_line_params)

    output_dataframe = helpers.combine_dataframes(orders, order_lines, 'DateTime', '1M', 'OrderId')

    expected_dataframe = pd.DataFrame({'OrderId': np.array([52, 41, 41]),
                                       'CustomerId': np.array([15707, 15265, 15265]),
                                       'ProductId': np.array([5390, 8433, 3073]),
                                       'Price': np.array([4304.488533, 937.473441, 4623.919930])})

    expected_dataframe.index = pd.RangeIndex(start=0, stop=3, step=1)

    pdt.assert_frame_equal(output_dataframe, expected_dataframe)


def test_most_popular():
    order_params = {"size": 1000,
                    "order_id_range": 70,
                    "customer_id_range": 20000,
                    "datetime_range": 1095,
                    "start_date": "2017-01-01"}

    orders = helpers.orders_mock_generator(**order_params)

    order_line_params = {"size": 1000,
                         "product_id_range": 10000,
                         "order_id_range": 70,
                         "price_range": [6.0, 5000.0]}

    order_lines = helpers.order_lines_mock_generator(**order_line_params)

    full_df = helpers.combine_dataframes(orders, order_lines, 'DateTime', '1M', 'OrderId')

    output_array = helpers.most_popular(full_df, 5)
    expected_array = np.array([9, 7400, 6544, 7830, 7560])

    npt.assert_array_equal(output_array, expected_array)


def test_sum_price():
    order_params = {"size": 1000,
                    "order_id_range": 70,
                    "customer_id_range": 20000,
                    "datetime_range": 1095,
                    "start_date": "2017-01-01"}

    orders = helpers.orders_mock_generator(**order_params)

    order_line_params = {"size": 1000,
                         "product_id_range": 10000,
                         "order_id_range": 70,
                         "price_range": [6.0, 5000.0]}

    order_lines = helpers.order_lines_mock_generator(**order_line_params)

    full_df = helpers.combine_dataframes(orders, order_lines, 'DateTime', '1M', 'OrderId')

    topn = helpers.most_popular(full_df, 5)

    output_list = [helpers.sum_price(full_df, 'ProductId', x) for x in topn]
    expected_list = [9884.248749578293, 11618.180478452077, 5120.931623720911, 6616.692310380738, 11896.544905964982]

    npt.assert_array_equal(output_list, expected_list)


def test_unique_orderid():
    order_params = {"size": 1000,
                    "order_id_range": 70,
                    "customer_id_range": 20000,
                    "datetime_range": 1095,
                    "start_date": "2017-01-01"}

    orders = helpers.orders_mock_generator(**order_params)

    order_line_params = {"size": 1000,
                         "product_id_range": 10000,
                         "order_id_range": 70,
                         "price_range": [6.0, 5000.0]}

    order_lines = helpers.order_lines_mock_generator(**order_line_params)

    full_df = helpers.combine_dataframes(orders, order_lines, 'DateTime', '1M', 'OrderId')

    topn = helpers.most_popular(full_df, 5)

    output_uniques = helpers.unique_orderid(full_df, topn)
    expected_uniques = np.array([2, 2, 1, 1, 1])

    npt.assert_array_equal(output_uniques, expected_uniques)


def test_create_summary():
    order_params = {"size": 1000,
                    "order_id_range": 70,
                    "customer_id_range": 20000,
                    "datetime_range": 1095,
                    "start_date": "2017-01-01"}

    orders = helpers.orders_mock_generator(**order_params)

    order_line_params = {"size": 10000,
                         "product_id_range": 10000,
                         "order_id_range": 70,
                         "price_range": [6.0, 5000.0]}

    order_lines = helpers.order_lines_mock_generator(**order_line_params)

    output_dataframe = create_summary(orders, order_lines)
    expected_dataframe = pd.DataFrame({'ProductId': np.array([8652, 3083, 615, 526, 4076]),
                                       'TotalRevenue': np.array(
                                           [21212.549591, 20723.342060, 16072.759135, 8999.519248, 14749.578349]),
                                       'AverageCheck': np.array(
                                           [7070.849864, 6907.780687, 5357.586378, 4499.759624, 7374.789174]),
                                       })

    pdt.assert_frame_equal(output_dataframe, expected_dataframe)