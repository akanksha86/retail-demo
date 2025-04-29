import pandas as pd
import numpy as np
import random
import uuid
import os
from datetime import timedelta, datetime
from faker import Faker
from google.cloud import bigquery
from google.oauth2 import service_account

PROJECT_ID = os.environ.get("PROJECT_ID")
DATASET_ID = os.environ.get("DATASET_ID")
GCP_LOCATION = os.environ.get("GCP_LOCATION")


class SyntheticDataGenerator:
    """
    A class to generate synthetic data for a Nordics-based retailer.
    """

    def __init__(self, seed=42, locale="en_US"):
        """
        Initialize the data generator with a seed for reproducibility.

        Args:
            seed (int): Random seed for reproducibility
            locale (str): Locale for Faker
        """
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        self.faker = Faker(locale)
        self.faker.seed_instance(seed)

        # Nordic countries and cities
        self.countries = ["Sweden", "Norway", "Denmark", "Finland", "Iceland"]
        self.cities = {
            "Sweden": ["Stockholm", "Gothenburg", "Malmö", "Uppsala", "Västerås"],
            "Norway": ["Oslo", "Bergen", "Trondheim", "Stavanger", "Drammen"],
            "Denmark": ["Copenhagen", "Aarhus", "Odense", "Aalborg", "Esbjerg"],
            "Finland": ["Helsinki", "Espoo", "Tampere", "Vantaa", "Oulu"],
            "Iceland": ["Reykjavik", "Kópavogur", "Hafnarfjörður", "Akureyri", "Reykjanesbær"]
        }

        # Product categories and subcategories
        self.product_categories = {
            "Clothing": [
                "Spring Dresses",
                "Light Jackets",
                "Casual Tops",
                "Spring Pants",
                "Swimwear",
                "Accessories"
            ],
            "Home & Living": [
                "Spring Decor",
                "Garden & Outdoor",
                "Kitchen & Dining",
                "Bedding & Bath",
                "Storage & Organization",
                "Lighting"
            ],
            "Beauty & Personal Care": [
                "Skincare",
                "Makeup",
                "Haircare",
                "Fragrances",
                "Body Care",
                "Sun Care"
            ]
        }

        # Product brands by category
        self.brands = {
            "Clothing": [
                "H&M",
                "Zara",
                "COS",
                "Arket",
                "& Other Stories",
                "Weekday",
                "Monki",
                "Gina Tricot",
                "Lager 157",
                "Bik Bok"
            ],
            "Home & Living": [
                "IKEA",
                "Søstrene Grene",
                "Hem",
                "Kvadrat",
                "Flying Tiger",
                "Jysk",
                "Rusta",
                "Clas Ohlson",
                "Tiger",
                "Åhlens"
            ],
            "Beauty & Personal Care": [
                "Kicks",
                "Sephora",
                "Lyko",
                "KICKS",
                "Apoteket",
                "Kruidvat",
                "Normal",
                "Beauty Bay",
                "Lookfantastic",
                "Notino"
            ]
        }

        # Spring-specific product attributes
        self.spring_colors = [
            "Pastel Pink",
            "Mint Green",
            "Sky Blue",
            "Lavender",
            "Coral",
            "Butter Yellow",
            "Sage Green",
            "Peach",
            "Light Blue",
            "Cream"
        ]

        self.spring_materials = [
            "Cotton",
            "Linen",
            "Silk",
            "Light Wool",
            "Bamboo",
            "Recycled Polyester",
            "Organic Cotton",
            "Tencel",
            "Hemp",
            "Viscose"
        ]

        self.spring_patterns = [
            "Floral",
            "Striped",
            "Polka Dot",
            "Gingham",
            "Paisley",
            "Abstract",
            "Geometric",
            "Tropical",
            "Botanical",
            "Watercolor"
        ]

        # Payment methods
        self.payment_methods = ["Credit Card", "Debit Card", "Swish", "Klarna", "Vipps", "MobilePay"]

        # Shipping methods
        self.shipping_methods = ["Standard", "Express", "Pickup Point", "Home Delivery"]

        # Currency information
        self.currencies = {
            "SEK": {"name": "Swedish Krona", "symbol": "kr", "exchange_rate": 1.0},
            "NOK": {"name": "Norwegian Krone", "symbol": "kr", "exchange_rate": 0.98},
            "DKK": {"name": "Danish Krone", "symbol": "kr", "exchange_rate": 0.68},
            "EUR": {"name": "Euro", "symbol": "€", "exchange_rate": 0.087},
            "ISK": {"name": "Icelandic Króna", "symbol": "kr", "exchange_rate": 0.0072}
        }

        # Currency by country
        self.country_currency = {
            "Sweden": "SEK",
            "Norway": "NOK",
            "Denmark": "DKK",
            "Finland": "EUR",
            "Iceland": "ISK"
        }

        # Initialize dataframes
        self.customer_df = None
        self.product_df = None
        self.order_df = None
        self.order_items_df = None
        self.exchange_rates_df = None

    def convert_currency(self, amount, from_currency, to_currency):
        """
        Convert amount between currencies using exchange rates.

        Args:
            amount (float): Amount to convert
            from_currency (str): Source currency code
            to_currency (str): Target currency code

        Returns:
            float: Converted amount
        """
        if from_currency == to_currency:
            return amount
        
        # Convert to SEK first (base currency)
        amount_in_sek = amount / self.currencies[from_currency]["exchange_rate"]
        # Then convert to target currency
        return amount_in_sek * self.currencies[to_currency]["exchange_rate"]

    def format_currency(self, amount, currency):
        """
        Format amount according to currency rules.

        Args:
            amount (float): Amount to format
            currency (str): Currency code

        Returns:
            str: Formatted amount with currency symbol
        """
        if currency in ["SEK", "NOK", "DKK"]:
            return f"{amount:.2f} {self.currencies[currency]['symbol']}"
        elif currency == "EUR":
            return f"€{amount:.2f}"
        elif currency == "ISK":
            return f"{amount:.0f} {self.currencies[currency]['symbol']}"
        return f"{amount:.2f} {currency}"

    def generate_customer_data(self, num_customers=1000):
        """
        Generate synthetic customer profile data.

        Args:
            num_customers (int): Number of customer profiles to generate

        Returns:
            pandas.DataFrame: DataFrame containing customer data
        """
        data = []

        for _ in range(num_customers):
            country = random.choice(self.countries)
            city = random.choice(self.cities[country])
            
            first_name = self.faker.first_name()
            last_name = self.faker.last_name()
            
            # Generate email based on name
            email_domain = random.choice(["gmail.com", "hotmail.com", "outlook.com", "yahoo.com"])
            email = f"{first_name.lower()}.{last_name.lower()}@{email_domain}"
            
            customer = {
                "customer_id": str(uuid.uuid4()),
                "first_name": first_name,
                "last_name": last_name,
                "email": email,
                "phone_number": self.faker.phone_number(),
                "street_address": self.faker.street_address(),
                "city": city,
                "country": country,
                "postal_code": self.faker.postcode(),
                "registration_date": self.faker.date_time_between(
                    start_date="-3y", end_date="now"
                ),
                "loyalty_points": random.randint(0, 5000),
                "preferred_payment_method": random.choice(self.payment_methods),
                "preferred_shipping_method": random.choice(self.shipping_methods),
                "is_active": random.random() < 0.9  # 90% of customers are active
            }
            data.append(customer)

        self.customer_df = pd.DataFrame(data)
        return self.customer_df

    def generate_product_data(self, num_products=500):
        """
        Generate synthetic product data with spring theme.

        Args:
            num_products (int): Number of products to generate

        Returns:
            pandas.DataFrame: DataFrame containing product data
        """
        data = []

        for _ in range(num_products):
            category = random.choice(list(self.product_categories.keys()))
            subcategory = random.choice(self.product_categories[category])
            brand = random.choice(self.brands[category])
            
            # Generate realistic price ranges based on category and subcategory (in SEK)
            if category == "Clothing":
                if subcategory in ["Spring Dresses", "Light Jackets"]:
                    base_price = random.uniform(2000, 8000)  # 200-800 SEK
                elif subcategory == "Accessories":
                    base_price = random.uniform(500, 3000)   # 50-300 SEK
                else:
                    base_price = random.uniform(1000, 5000)  # 100-500 SEK
            elif category == "Home & Living":
                if subcategory in ["Garden & Outdoor", "Kitchen & Dining"]:
                    base_price = random.uniform(1500, 10000)  # 150-1000 SEK
                elif subcategory == "Lighting":
                    base_price = random.uniform(2000, 8000)   # 200-800 SEK
                else:
                    base_price = random.uniform(500, 5000)    # 50-500 SEK
            else:  # Beauty & Personal Care
                if subcategory in ["Fragrances", "Skincare"]:
                    base_price = random.uniform(1000, 4000)   # 100-400 SEK
                else:
                    base_price = random.uniform(200, 2000)    # 20-200 SEK

            # Generate spring-themed product name
            color = random.choice(self.spring_colors)
            material = random.choice(self.spring_materials)
            pattern = random.choice(self.spring_patterns)
            
            if category == "Clothing":
                name = f"{brand} {color} {pattern} {subcategory}"
            elif category == "Home & Living":
                name = f"{brand} {material} {pattern} {subcategory}"
            else:  # Beauty & Personal Care
                name = f"{brand} {color} {subcategory} Collection"

            product = {
                "product_id": str(uuid.uuid4()),
                "name": name,
                "category": category,
                "subcategory": subcategory,
                "brand": brand,
                "color": color,
                "material": material if category != "Beauty & Personal Care" else None,
                "pattern": pattern if category != "Beauty & Personal Care" else None,
                "price": round(base_price, 2),
                "price_unit": "SEK",
                "cost": round(base_price * random.uniform(0.4, 0.7), 2),  # 40-70% of price
                "cost_unit": "SEK",
                "stock_quantity": random.randint(0, 100),
                "is_active": random.random() < 0.95,  # 95% of products are active
                "created_date": self.faker.date_time_between(
                    start_date="-2y", end_date="now"
                ),
                "last_restock_date": self.faker.date_time_between(
                    start_date="-6m", end_date="now"
                ),
                "is_spring_collection": True,
                "is_sustainable": random.random() < 0.3  # 30% of products are sustainable
            }
            data.append(product)

        self.product_df = pd.DataFrame(data)
        return self.product_df

    def generate_order_data(self, customer_df=None, product_df=None, num_orders=2000):
        """
        Generate synthetic order data with multiple currencies.

        Args:
            customer_df (pandas.DataFrame, optional): Customer data
            product_df (pandas.DataFrame, optional): Product data
            num_orders (int): Number of orders to generate

        Returns:
            tuple: (order_df, order_items_df) containing order and order items data
        """
        if customer_df is None:
            if self.customer_df is None:
                raise ValueError("No customer data available. Generate customer data first.")
            customer_df = self.customer_df

        if product_df is None:
            if self.product_df is None:
                raise ValueError("No product data available. Generate product data first.")
            product_df = self.product_df

        orders_data = []
        order_items_data = []

        # Set date range for orders
        start_date = datetime(2025, 1, 1)
        end_date = datetime.now()

        for _ in range(num_orders):
            customer = customer_df.sample(n=1).iloc[0]
            order_date = self.faker.date_time_between(
                start_date=start_date,
                end_date=end_date
            )
            
            # Determine order currency based on customer's country
            order_currency = self.country_currency[customer["country"]]
            
            # Generate 1-5 items per order
            num_items = random.randint(1, 5)
            order_items = product_df.sample(n=num_items)
            
            # Calculate order total in SEK first
            order_total_sek = order_items["price"].sum()
            
            # Convert to order currency
            order_total = self.convert_currency(order_total_sek, "SEK", order_currency)
            
            # Shipping cost in order currency
            shipping_cost = random.uniform(0, 500) if order_total < 5000 else 0
            tax_rate = 0.25  # Standard Nordic VAT rate
            tax_amount = (order_total + shipping_cost) * tax_rate
            
            order = {
                "order_id": str(uuid.uuid4()),
                "customer_id": customer["customer_id"],
                "order_date": order_date,
                "status": random.choice(["Completed", "Processing", "Shipped", "Delivered"]),
                "payment_method": customer["preferred_payment_method"],
                "shipping_method": customer["preferred_shipping_method"],
                "subtotal": round(order_total, 2),
                "shipping_cost": round(shipping_cost, 2),
                "tax_amount": round(tax_amount, 2),
                "total_amount": round(order_total + shipping_cost + tax_amount, 2),
                "currency": order_currency,
                "currency_symbol": self.currencies[order_currency]["symbol"],
                "formatted_total": self.format_currency(
                    round(order_total + shipping_cost + tax_amount, 2),
                    order_currency
                ),
                "is_returned": random.random() < 0.05  # 5% return rate
            }
            orders_data.append(order)

            # Generate order items
            for _, product in order_items.iterrows():
                quantity = random.randint(1, 3)
                # Convert product price to order currency
                unit_price = self.convert_currency(product["price"], "SEK", order_currency)
                total_price = unit_price * quantity
                
                item = {
                    "order_item_id": str(uuid.uuid4()),
                    "order_id": order["order_id"],
                    "product_id": product["product_id"],
                    "quantity": quantity,
                    "unit_price": round(unit_price, 2),
                    "total_price": round(total_price, 2),
                    "currency": order_currency,
                    "currency_symbol": self.currencies[order_currency]["symbol"],
                    "formatted_price": self.format_currency(round(total_price, 2), order_currency)
                }
                order_items_data.append(item)

        self.order_df = pd.DataFrame(orders_data)
        self.order_items_df = pd.DataFrame(order_items_data)
        return self.order_df, self.order_items_df

    def generate_currency_exchange_rates(self, start_date=None, end_date=None):
        """
        Generate monthly currency exchange rates relative to SEK.

        Args:
            start_date (datetime, optional): Start date for exchange rates
            end_date (datetime, optional): End date for exchange rates

        Returns:
            pandas.DataFrame: DataFrame containing monthly exchange rates
        """
        if start_date is None:
            start_date = datetime(2025, 1, 1)
        if end_date is None:
            end_date = datetime.now()

        # Generate monthly dates
        dates = pd.date_range(start=start_date, end=end_date, freq='MS')
        
        data = []
        for date in dates:
            # Base rates with some random variation
            base_rates = {
                "SEK": 1.0,  # Base currency
                "NOK": 0.98 + random.uniform(-0.02, 0.02),  # ~0.98 NOK per SEK
                "DKK": 0.68 + random.uniform(-0.01, 0.01),  # ~0.68 DKK per SEK
                "EUR": 0.087 + random.uniform(-0.002, 0.002),  # ~0.087 EUR per SEK
                "ISK": 0.0072 + random.uniform(-0.0002, 0.0002)  # ~0.0072 ISK per SEK
            }
            
            for currency, rate in base_rates.items():
                data.append({
                    "date": date,
                    "currency": currency,
                    "exchange_rate": round(rate, 4),
                    "base_currency": "SEK"
                })

        self.exchange_rates_df = pd.DataFrame(data)
        return self.exchange_rates_df

    def save_to_parquet(self, output_dir="data"):
        """
        Save all generated data to parquet files.

        Args:
            output_dir (str): Directory to save the parquet files

        Returns:
            dict: Paths to the saved parquet files
        """
        os.makedirs(output_dir, exist_ok=True)

        paths = {}

        if self.customer_df is not None:
            customer_path = os.path.join(output_dir, "customers.parquet")
            self.customer_df.to_parquet(customer_path, index=False)
            paths["customers"] = customer_path

        if self.product_df is not None:
            product_path = os.path.join(output_dir, "products.parquet")
            self.product_df.to_parquet(product_path, index=False)
            paths["products"] = product_path

        if self.order_df is not None:
            order_path = os.path.join(output_dir, "orders.parquet")
            self.order_df.to_parquet(order_path, index=False)
            paths["orders"] = order_path

        if self.order_items_df is not None:
            order_items_path = os.path.join(output_dir, "order_items.parquet")
            self.order_items_df.to_parquet(order_items_path, index=False)
            paths["order_items"] = order_items_path

        if self.exchange_rates_df is not None:
            exchange_rates_path = os.path.join(output_dir, "exchange_rates.parquet")
            self.exchange_rates_df.to_parquet(exchange_rates_path, index=False)
            paths["exchange_rates"] = exchange_rates_path

        return paths

    def load_to_bigquery(self, project_id, dataset_id, credentials_path=None):
        """
        Load all generated data to BigQuery.

        Args:
            project_id (str): Google Cloud project ID
            dataset_id (str): BigQuery dataset ID
            credentials_path (str, optional): Path to service account credentials JSON file

        Returns:
            dict: BigQuery table references
        """
        if credentials_path:
            credentials = service_account.Credentials.from_service_account_file(
                credentials_path, scopes=["https://www.googleapis.com/auth/bigquery"]
            )
            client = bigquery.Client(credentials=credentials, project=project_id)
        else:
            client = bigquery.Client(project=project_id)

        dataset_ref = f"{project_id}.{dataset_id}"
        try:
            client.get_dataset(dataset_ref)
        except Exception:
            dataset = bigquery.Dataset(dataset_ref)
            dataset.location = GCP_LOCATION
            client.create_dataset(dataset, exists_ok=True)

        table_refs = {}

        if self.customer_df is not None:
            table_id = f"{dataset_ref}.customers"
            job_config = bigquery.LoadJobConfig(
                write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
            )
            job = client.load_table_from_dataframe(
                self.customer_df, table_id, job_config=job_config
            )
            job.result()
            table_refs["customers"] = table_id

        if self.product_df is not None:
            table_id = f"{dataset_ref}.products"
            job_config = bigquery.LoadJobConfig(
                write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
            )
            job = client.load_table_from_dataframe(
                self.product_df, table_id, job_config=job_config
            )
            job.result()
            table_refs["products"] = table_id

        if self.order_df is not None:
            table_id = f"{dataset_ref}.orders"
            job_config = bigquery.LoadJobConfig(
                write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
            )
            job = client.load_table_from_dataframe(
                self.order_df, table_id, job_config=job_config
            )
            job.result()
            table_refs["orders"] = table_id

        if self.order_items_df is not None:
            table_id = f"{dataset_ref}.order_items"
            job_config = bigquery.LoadJobConfig(
                write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
            )
            job = client.load_table_from_dataframe(
                self.order_items_df, table_id, job_config=job_config
            )
            job.result()
            table_refs["order_items"] = table_id

        if self.exchange_rates_df is not None:
            table_id = f"{dataset_ref}.exchange_rates"
            job_config = bigquery.LoadJobConfig(
                write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
            )
            job = client.load_table_from_dataframe(
                self.exchange_rates_df, table_id, job_config=job_config
            )
            job.result()
            table_refs["exchange_rates"] = table_id

        return table_refs

    def generate_all_data(
        self,
        num_customers=2000,
        num_products=500,
        num_orders=10000
    ):
        """
        Generate all types of data at once.

        Args:
            num_customers (int): Number of customer profiles to generate
            num_products (int): Number of products to generate
            num_orders (int): Number of orders to generate

        Returns:
            tuple: (customer_df, product_df, order_df, order_items_df, exchange_rates_df)
        """
        self.generate_customer_data(num_customers)
        self.generate_product_data(num_products)
        self.generate_order_data(num_orders=num_orders)
        self.generate_currency_exchange_rates()

        return self.customer_df, self.product_df, self.order_df, self.order_items_df, self.exchange_rates_df

    def generate_sales_forecast(self, project_id, dataset_id, credentials_path=None, forecast_horizon=12):
        """
        Generate sales forecasts using BigQuery's TIMES_FM model.

        Args:
            project_id (str): Google Cloud project ID
            dataset_id (str): BigQuery dataset ID
            credentials_path (str, optional): Path to service account credentials JSON file
            forecast_horizon (int): Number of periods to forecast (default: 12 months)

        Returns:
            pandas.DataFrame: DataFrame containing the forecast results
        """
        if self.order_df is None:
            raise ValueError("No order data available. Generate order data first.")

        if credentials_path:
            credentials = service_account.Credentials.from_service_account_file(
                credentials_path, scopes=["https://www.googleapis.com/auth/bigquery"]
            )
            client = bigquery.Client(credentials=credentials, project=project_id)
        else:
            client = bigquery.Client(project=project_id)

        # Prepare the time series data
        # Aggregate daily sales to monthly
        monthly_sales = self.order_df.groupby(
            pd.Grouper(key='order_date', freq='MS')
        )['total_amount'].sum().reset_index()
        
        monthly_sales.columns = ['ds', 'y']  # Rename columns for TIMES_FM format
        
        # Create the model
        model = bigquery.Model(f"{project_id}.{dataset_id}.sales_forecast_model")
        
        # Configure the model
        model.model_type = bigquery.Model.ModelType.ARIMA_PLUS
        model.time_series_timestamp_column = "ds"
        model.time_series_data_column = "y"
        model.data_frequency = bigquery.Model.DataFrequency.MONTHLY
        model.holiday_region = bigquery.Model.HolidayRegion.EUROPE  # For Nordic countries
        
        # Create the model
        model = client.create_model(model)
        
        # Train the model
        job_config = bigquery.QueryJobConfig()
        job_config.query = f"""
        CREATE OR REPLACE MODEL `{project_id}.{dataset_id}.sales_forecast_model`
        OPTIONS(
            model_type='ARIMA_PLUS',
            time_series_timestamp_column='ds',
            time_series_data_column='y',
            data_frequency='MONTHLY',
            holiday_region='EUROPE'
        )
        AS
        SELECT * FROM `{project_id}.{dataset_id}.monthly_sales`
        """
        
        query_job = client.query(job_config.query)
        query_job.result()
        
        # Generate forecast
        forecast_query = f"""
        SELECT
            forecast_timestamp,
            forecast_value,
            prediction_interval_lower_bound,
            prediction_interval_upper_bound
        FROM
            ML.FORECAST(
                MODEL `{project_id}.{dataset_id}.sales_forecast_model`,
                STRUCT({forecast_horizon} AS horizon)
            )
        ORDER BY
            forecast_timestamp
        """
        
        forecast_job = client.query(forecast_query)
        forecast_results = forecast_job.result().to_dataframe()
        
        return forecast_results


if __name__ == "__main__":
    generator = SyntheticDataGenerator(seed=42)

    customers, products, orders, order_items, exchange_rates = generator.generate_all_data(
        num_customers=2000,
        num_products=500,
        num_orders=10000
    )

    paths = generator.save_to_parquet(output_dir="data")
    print(f"Data saved to: {paths}")

    table_refs = generator.load_to_bigquery(
        project_id=PROJECT_ID, dataset_id=DATASET_ID
    )
    print(f"Data loaded to BigQuery tables: {table_refs}")
