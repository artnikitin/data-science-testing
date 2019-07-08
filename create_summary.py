import pandas as pd
import numpy as np
import helpers


def create_summary(orders, order_lines):
    """
    Создает отчет по популярным продуктами из orders и order_lines:
    – Самые популярные за последний месяц продукты
    – Суммарная выручка по каждому такому продукту
    – Средний чек заказов, в которых есть такие продукты

    :param orders: фрейм orders
    :param order_lines: фрейм order_lines
    :return: отчет
    """

    # проверяет соответствие фреймов шаблону
    helpers.check_orders_schema(orders)
    helpers.check_order_lines_schema(order_lines)

    # отбирает данные за последний месяц и объединяет фреймы
    full_df = helpers.combine_dataframes(orders, order_lines, 'DateTime', '1M', 'OrderId')

    # находит топ-5 самых популярных продуктов
    topn = helpers.most_popular(full_df, 5)

    # cчитает совокупную выручку по каждому такому продукту
    sums = np.array([helpers.sum_price(full_df, 'ProductId', x) for x in topn])

    # считает средний чек заказов, в которых есть такие продукты
    avg_check = sums / helpers.unique_orderid(full_df, topn)

    return pd.DataFrame({'ProductId': topn, 'TotalRevenue': sums, 'AverageCheck': avg_check})