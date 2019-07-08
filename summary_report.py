import helpers
from create_summary import create_summary

orders_mock_params = {"size": 1000,
                      "order_id_range": 70,
                      "customer_id_range": 20000,
                      "datetime_range": 1095,
                      "start_date": "2017-01-01"}

order_lines_mock_params = {"size": 10000,
                           "product_id_range": 10000,
                           "order_id_range": 70,
                           "price_range": [6.0, 5000.0]}

orders_mock = helpers.orders_mock_generator(**orders_mock_params)
order_lines_mock = helpers.order_lines_mock_generator(**order_lines_mock_params)

def main():
    summary = create_summary(orders_mock, order_lines_mock)
    print(summary)

if __name__ == "__main__":
    main()