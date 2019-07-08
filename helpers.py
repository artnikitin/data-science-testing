import pandas as pd
import numpy as np


def random_date_generator(size, start_date, range_in_days, random_state=42):
    """
    Генерирует массив случайных дат.

    :param size: размер массива
    :param start_date: начальная дата, к которой буду прибавляться случайные дельты дней
    :param range_in_days: максимальный размер дельты дней
    :param random_state: seed
    :return: массив случайных дат
    """
    np.random.seed(random_state)

    days_to_add = np.random.randint(range_in_days, size=size)
    start_date_array = np.array([np.datetime64(start_date) for i in range(size)])

    return start_date_array + days_to_add


def orders_mock_generator(size, order_id_range, customer_id_range,
                          datetime_range, start_date, random_state=42):
    """
    Генерирует фрейм orders из случайных чисел.

    :param size: размер фрейма
    :param order_id_range: максимальный размер order_id
    :param customer_id_range: максимальный размер customer_id
    :param datetime_range: максимальный размер дельты дней
    :param start_date: начальная дата
    :param random_state: seed
    :return: фрейм orders
    """
    np.random.seed(random_state)
    return pd.DataFrame({'OrderId': np.random.randint(order_id_range, size=size),
                         'CustomerId': np.random.randint(customer_id_range, size=size),
                         'DateTime': np.array(
                             random_date_generator(size, start_date, datetime_range))})


def order_lines_mock_generator(size, product_id_range, order_id_range,
                               price_range, random_state=42):
    """
    Генерирует фрейм order_lines из случайных чисел.

    :param size: размер фрейма
    :param product_id_range: максимальный размер product_id
    :param order_id_range: максимальный размер order_id
    :param price_range: максимальный размер price
    :param random_state: seed
    :return: фрейм order_lines
    """
    np.random.seed(random_state)
    return pd.DataFrame({'ProductId': np.random.randint(product_id_range, size=size),
                         'OrderId': np.random.randint(order_id_range, size=size),
                         'Price': np.random.uniform(price_range[0], price_range[1], size=size)})


def check_orders_schema(orders):
    """
    Проверяет правильность фрейма orders (название колонок и типы).

    :param orders: фрейм orders
    :return: True, если нет ошибки, либо TypeError
    """
    expected_dtypes = {'OrderId': np.dtype('int64'),
                       'CustomerId': np.dtype('int64'),
                       'DateTime': np.dtype('<M8[ns]')}

    if dict(orders.dtypes) != expected_dtypes:
        print(expected_dtypes)
        raise TypeError("orders DataFrame has incorrect schema (columns or dtypes)")

    else:
        return True

def check_order_lines_schema(order_lines):
    """
    Проверяет правильность фрейма order_lines (название колонок и типы).

    :param order_lines: фрейм order_lines
    :return: True, если нет ошибки, либо TypeError
    """
    expected_dtypes = {'ProductId': np.dtype('int64'),
                         'OrderId': np.dtype('int64'),
                         'Price': np.dtype('float64')}

    if dict(order_lines.dtypes) != expected_dtypes:
        print(expected_dtypes)
        raise TypeError("order_lines DataFrame has incorrect schema (columns or dtypes)")

    else:
        return True

def select_last(df, sort_column, period):
    """
    Выбирает все продукты за последний месяц.

    :param df: фрейм orders
    :param sort_column: колонка, по которой производится сортировка ('DateTime')
    :param period: период, в данном случае один месяц
    :return: отфильтрованный фрейм
    """
    return df.sort_values(by=sort_column, ascending=True).set_index(sort_column).last(period)

def combine_dataframes(orders, order_lines, sort_column, time_range, join_column):
    """
    Объединяет (join) два датафрейма orders и order_lines по колонке 'OrderId'.

    :param orders: фрейм orders
    :param order_lines: фрейм order_lines
    :param sort_column: колонка, по которой производится сортировка ('DateTime')
    :param time_range: период, в данном случае один месяц
    :param join_column: колонка, по которой производится join ('OrderId')
    :return: объединенный фрейм
    """
    return pd.merge(select_last(orders, sort_column, time_range), order_lines, on=[join_column])


def most_popular(df, topn):
    """
    Отбирает topn самых популярных продуктов.

    :param df: объединенный фрейм
    :param topn: число самых популярных продуктов для вывода
    :return: массив topn самых популярных продуктов
    """
    return np.array(df['ProductId'].value_counts().head(topn).index)


def sum_price(df, column, product_id):
    """
    Считает суммарную выручку по каждому такому продукту.

    :param df: объединенный фрейм
    :param column: колонка с продуктами ('ProductId')
    :param product_id: id продукта из topn
    :return: суммарную выручку по выбранному продукту
    """
    return df[df[column] == product_id]['Price'].sum()


def unique_orderid(df, topn):
    """
    Считает количество уникальных заказов для каждого продукта из topn.

    :param df: объединенный фрейм
    :param topn: число самых популярных продуктов для вывода
    :return: массив количества уникальных заказов для каждого продукта из topn
    """
    return np.array([df[df['ProductId'] == item]['OrderId'].nunique() for item in topn])
