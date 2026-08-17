"""Builds the CSV export of a customer's orders."""

from .orders_api import list_orders
from .csv_writer import rows_to_csv


def export_orders_csv(customer_id: str) -> str:
    orders = list_orders(customer_id, page_size=25)
    return rows_to_csv(orders)
