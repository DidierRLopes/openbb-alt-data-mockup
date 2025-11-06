# Import required libraries
import json
import random
import requests
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from functools import wraps
import asyncio
from typing import List, Optional, Dict, Any

# Initialize FastAPI application
app = FastAPI(
    title="Carbon Arc Backend for OpenBB Workspace",
    description="Comprehensive mock data backend for all 5 Carbon Arc apps",
    version="2.0.0"
)

# CORS configuration
origins = [
    "https://pro.openbb.co",
    "https://pro.openbb.dev",
    "http://localhost:1420"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize empty dictionary for widgets
WIDGETS = {}

def register_widget(widget_config):
    """Decorator that registers a widget configuration in the WIDGETS dictionary."""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            return await func(*args, **kwargs)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        
        # Extract the endpoint from the widget_config
        endpoint = widget_config.get("endpoint")
        if endpoint:
            # Add an id field to the widget_config if not already present
            if "widgetId" not in widget_config:
                widget_config["widgetId"] = endpoint

            # Use id as the key to allow multiple widgets per endpoint
            widget_id = widget_config["widgetId"]
            WIDGETS[widget_id] = widget_config

        # Return the appropriate wrapper based on whether the function is async
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    return decorator

@app.get("/")
def read_root():
    """Root endpoint"""
    return {"Info": "Carbon Arc Backend for OpenBB Workspace - All 5 Apps"}

@app.get("/widgets.json")
def get_widgets():
    """Returns the configuration of all registered widgets"""
    return WIDGETS

@app.get("/apps.json")
def get_apps():
    """Returns the Carbon Arc apps configuration for OpenBB Workspace"""
    try:
        with open("carbon_arc_apps.json", "r") as f:
            apps_config = json.load(f)
        return apps_config
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Apps configuration file not found")
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Invalid JSON in apps configuration file")

# =============================================================================
# UTILITY FUNCTIONS FOR MOCK DATA GENERATION
# =============================================================================

def generate_time_series(start_date: str, end_date: str, freq: str = 'D', base_value: float = 100, volatility: float = 0.1, trend: float = 0.001):
    """Generate realistic time series data with trend and seasonality"""
    np.random.seed(42)
    dates = pd.date_range(start=start_date, end=end_date, freq=freq)
    
    # Generate random walk with trend
    returns = np.random.normal(trend, volatility, len(dates))
    # Add seasonality (weekly/monthly patterns)
    seasonality = 0.05 * np.sin(2 * np.pi * np.arange(len(dates)) / 7)  # Weekly pattern
    returns += seasonality
    
    # Calculate cumulative prices
    prices = base_value * np.exp(np.cumsum(returns))
    
    return dates, prices

def generate_brands_data(brands: List[str], metric_name: str, base_values: Dict[str, float] = None):
    """Generate correlated data for multiple brands"""
    if base_values is None:
        base_values = {brand: 50000 + random.uniform(-10000, 20000) for brand in brands}
    
    data = []
    for brand in brands:
        base = base_values.get(brand, 50000)
        growth = random.uniform(-0.15, 0.25)  # -15% to +25% YoY
        current_value = base * (1 + growth)
        
        data.append({
            "brand": brand,
            metric_name: round(current_value, 2),
            f"{metric_name}_yoy_change": round(growth * 100, 1)
        })
    
    return data

# =============================================================================
# 1. POS & TRANSACTIONS MONITOR APP WIDGETS
# =============================================================================

@register_widget({
    "name": "Sales Trend",
    "description": "Time series of revenue and units with YoY/3m/6m deltas",
    "category": "POS Analysis",
    "subcategory": "Revenue Analytics",
    "widgetId": "sales_trend",
    "gridData": {"x": 0, "y": 0, "w": 30, "h": 12},
    "endpoint": "sales_trend",
    "type": "table",
    "data": {
        "table": {
            "enableCharts": True,
            "chartView": {
                "enabled": True,
                "chartType": "line"
            },
            "columnsDefs": [
                {"field": "date", "headerName": "Date", "chartDataType": "time", "width": 120},
                {"field": "revenue", "headerName": "Revenue ($)", "chartDataType": "series", "formatterFn": "int", "width": 120},
                {"field": "units", "headerName": "Units Sold", "chartDataType": "series", "formatterFn": "int", "width": 120},
                {"field": "yoy_change", "headerName": "YoY Change (%)", "formatterFn": "percent", "renderFn": "greenRed", "width": 120}
            ]
        }
    },
    "params": [
        {
            "paramName": "brands",
            "type": "text",
            "multiSelect": True,
            "label": "Brands",
            "value": "Brand A",
            "options": [
                {"label": "Brand A", "value": "Brand A"},
                {"label": "Brand B", "value": "Brand B"},
                {"label": "Brand C", "value": "Brand C"}
            ]
        },
        {
            "paramName": "channel",
            "type": "text",
            "label": "Channel",
            "value": "all",
            "options": [
                {"label": "All Channels", "value": "all"},
                {"label": "Retail", "value": "retail"},
                {"label": "E-commerce", "value": "ecommerce"},
                {"label": "Mobile", "value": "mobile"}
            ]
        }
    ]
})
@app.get("/sales_trend")
def get_sales_trend(
    brands: str = "Brand A",
    channel: str = "all",
    date_range_start: str = "2024-01-01",
    date_range_end: str = "2024-12-31"
):
    """Get sales trend data as tabular format"""
    dates, revenues = generate_time_series(date_range_start, date_range_end, 'W', base_value=50000, volatility=0.08)
    
    data = []
    for i, (date, revenue) in enumerate(zip(dates, revenues)):
        units = int(revenue / (45 + random.uniform(-5, 10)))  # Variable price per unit
        yoy_change = random.uniform(-20, 30)  # Mock YoY change
        
        data.append({
            "date": date.strftime("%Y-%m-%d"),
            "revenue": round(revenue, 0),
            "units": units,
            "yoy_change": round(yoy_change, 1),
            "brand": brands.split(",")[0] if brands else "Brand A",
            "channel": channel
        })
    
    return data

@register_widget({
    "name": "Average Ticket & Basket",
    "description": "Line chart of avg ticket; table for basket size metrics",
    "category": "POS Analysis",
    "subcategory": "Revenue Analytics", 
    "widgetId": "avg_ticket_basket",
    "gridData": {"x": 30, "y": 0, "w": 20, "h": 12},
    "endpoint": "avg_ticket_basket",
    "type": "table",
    "data": {
        "table": {
            "enableCharts": True,
            "chartView": {
                "enabled": True,
                "chartType": "line"
            },
            "columnsDefs": [
                {"field": "date", "headerName": "Date", "chartDataType": "time", "width": 120},
                {"field": "avg_ticket", "headerName": "Avg Ticket ($)", "chartDataType": "series", "formatterFn": "none", "width": 130},
                {"field": "basket_size", "headerName": "Items/Transaction", "chartDataType": "series", "formatterFn": "none", "width": 130},
                {"field": "basket_value", "headerName": "Basket Value ($)", "chartDataType": "series", "formatterFn": "int", "width": 130}
            ]
        }
    },
    "params": [
        {
            "paramName": "brands",
            "type": "text",
            "multiSelect": True,
            "label": "Brands",
            "value": "Brand A",
            "options": [
                {"label": "Brand A", "value": "Brand A"},
                {"label": "Brand B", "value": "Brand B"},
                {"label": "Brand C", "value": "Brand C"}
            ]
        }
    ]
})
@app.get("/avg_ticket_basket")
def get_avg_ticket_basket(
    brands: str = "Brand A",
    date_range_start: str = "2024-01-01",
    date_range_end: str = "2024-12-31"
):
    """Get average ticket and basket metrics"""
    dates, base_values = generate_time_series(date_range_start, date_range_end, 'W', base_value=45, volatility=0.05)
    
    data = []
    for date, avg_ticket in zip(dates, base_values):
        basket_size = 2.1 + random.uniform(-0.5, 0.8)  # 1.6 to 2.9 items
        basket_value = avg_ticket * basket_size
        
        data.append({
            "date": date.strftime("%Y-%m-%d"),
            "avg_ticket": round(avg_ticket, 2),
            "basket_size": round(basket_size, 1),
            "basket_value": round(basket_value, 2),
            "brand": brands.split(",")[0] if brands else "Brand A"
        })
    
    return data

@register_widget({
    "name": "Channel Mix",
    "description": "Stacked bar showing in-store vs e-commerce mix with share % and changes",
    "category": "POS Analysis",
    "subcategory": "Channel Analytics",
    "widgetId": "channel_mix",
    "gridData": {"x": 0, "y": 12, "w": 25, "h": 10},
    "endpoint": "channel_mix",
    "type": "table",
    "data": {
        "table": {
            "enableCharts": True,
            "chartView": {
                "enabled": True,
                "chartType": "stackedColumn"
            },
            "columnsDefs": [
                {"field": "month", "headerName": "Month", "chartDataType": "category", "width": 100},
                {"field": "retail_revenue", "headerName": "Retail ($)", "chartDataType": "series", "formatterFn": "int", "width": 120},
                {"field": "ecommerce_revenue", "headerName": "E-commerce ($)", "chartDataType": "series", "formatterFn": "int", "width": 120},
                {"field": "mobile_revenue", "headerName": "Mobile ($)", "chartDataType": "series", "formatterFn": "int", "width": 120},
                {"field": "retail_share", "headerName": "Retail %", "formatterFn": "percent", "width": 90},
                {"field": "ecommerce_share", "headerName": "E-comm %", "formatterFn": "percent", "width": 90}
            ]
        }
    },
    "params": [
        {
            "paramName": "brands",
            "type": "text",
            "multiSelect": True,
            "label": "Brands",
            "value": "Brand A",
            "options": [
                {"label": "Brand A", "value": "Brand A"},
                {"label": "Brand B", "value": "Brand B"},
                {"label": "Brand C", "value": "Brand C"}
            ]
        }
    ]
})
@app.get("/channel_mix")
def get_channel_mix(
    brands: str = "Brand A",
    date_range_start: str = "2024-01-01",
    date_range_end: str = "2024-12-31"
):
    """Get channel mix analysis"""
    months = pd.date_range(start=date_range_start, end=date_range_end, freq='M')
    
    data = []
    for month in months:
        # Generate channel split with some trends (e-commerce growing)
        total_revenue = 180000 + random.uniform(-20000, 30000)
        ecommerce_share = 0.35 + 0.01 * len(data) + random.uniform(-0.02, 0.02)  # Growing trend
        mobile_share = 0.15 + 0.005 * len(data) + random.uniform(-0.01, 0.01)   # Growing trend
        retail_share = 1 - ecommerce_share - mobile_share
        
        retail_revenue = total_revenue * retail_share
        ecommerce_revenue = total_revenue * ecommerce_share  
        mobile_revenue = total_revenue * mobile_share
        
        data.append({
            "month": month.strftime("%Y-%m"),
            "retail_revenue": round(retail_revenue, 0),
            "ecommerce_revenue": round(ecommerce_revenue, 0),
            "mobile_revenue": round(mobile_revenue, 0),
            "retail_share": round(retail_share, 3),
            "ecommerce_share": round(ecommerce_share, 3),
            "mobile_share": round(mobile_share, 3),
            "brand": brands.split(",")[0] if brands else "Brand A"
        })
    
    return data

@register_widget({
    "name": "Geo Heatmap",
    "description": "Geographic sales distribution with YoY growth indicators",
    "category": "POS Analysis",
    "subcategory": "Geographic Analytics",
    "widgetId": "geo_heatmap",
    "gridData": {"x": 25, "y": 12, "w": 25, "h": 10},
    "endpoint": "geo_heatmap",
    "type": "table",
    "data": {
        "table": {
            "enableCharts": True,
            "chartView": {
                "enabled": True,
                "chartType": "column"
            },
            "columnsDefs": [
                {"field": "location", "headerName": "Location", "chartDataType": "category", "width": 120},
                {"field": "sales", "headerName": "Sales ($)", "chartDataType": "series", "formatterFn": "int", "width": 120},
                {"field": "yoy_growth", "headerName": "YoY Growth (%)", "formatterFn": "percent", "renderFn": "greenRed", "width": 120},
                {"field": "market_share", "headerName": "Market Share (%)", "formatterFn": "percent", "width": 120}
            ]
        }
    },
    "params": [
        {
            "paramName": "geo_level",
            "type": "text",
            "label": "Geographic Level",
            "value": "state",
            "options": [
                {"label": "State", "value": "state"},
                {"label": "City", "value": "city"},
                {"label": "Region", "value": "region"}
            ]
        }
    ]
})
@app.get("/geo_heatmap")
def get_geo_heatmap(
    geo_level: str = "state",
    brands: str = "Brand A"
):
    """Get geographic heatmap data"""
    if geo_level == "state":
        locations = ["CA", "TX", "NY", "FL", "IL", "PA", "OH", "GA", "NC", "MI"]
    elif geo_level == "city":
        locations = ["Los Angeles", "New York", "Chicago", "Houston", "Phoenix", "Philadelphia", "San Antonio", "San Diego", "Dallas", "San Jose"]
    else:
        locations = ["West", "East", "Central", "South", "Northeast"]
    
    data = []
    for location in locations:
        base_sales = 40000 + random.uniform(-15000, 25000)
        yoy_growth = random.uniform(-15, 25) / 100
        market_share = random.uniform(0.05, 0.25)
        
        data.append({
            "location": location,
            "sales": round(base_sales, 0),
            "yoy_growth": round(yoy_growth, 3),
            "market_share": round(market_share, 3),
            "brand": brands.split(",")[0] if brands else "Brand A"
        })
    
    return data

@register_widget({
    "name": "Sales vs Spend Surprise",
    "description": "Bar chart showing actual vs expected sales with surprise % and z-score",
    "category": "POS Analysis", 
    "subcategory": "Performance Analytics",
    "widgetId": "sales_surprise",
    "gridData": {"x": 0, "y": 22, "w": 25, "h": 8},
    "endpoint": "sales_surprise",
    "type": "table",
    "data": {
        "table": {
            "enableCharts": True,
            "chartView": {
                "enabled": True,
                "chartType": "groupedColumn"
            },
            "columnsDefs": [
                {"field": "brand", "headerName": "Brand", "chartDataType": "category", "width": 100},
                {"field": "actual_sales", "headerName": "Actual Sales ($)", "chartDataType": "series", "formatterFn": "int", "width": 140},
                {"field": "expected_sales", "headerName": "Expected Sales ($)", "chartDataType": "series", "formatterFn": "int", "width": 140},
                {"field": "surprise_pct", "headerName": "Surprise (%)", "formatterFn": "percent", "renderFn": "greenRed", "width": 120},
                {"field": "z_score", "headerName": "Z-Score", "formatterFn": "none", "width": 100}
            ]
        }
    },
    "params": [
        {
            "paramName": "compare_set",
            "type": "text",
            "label": "Compare Against",
            "value": "consensus",
            "options": [
                {"label": "Consensus", "value": "consensus"},
                {"label": "Last Year", "value": "yoy"},
                {"label": "Budget", "value": "budget"}
            ]
        }
    ]
})
@app.get("/sales_surprise") 
def get_sales_surprise(
    compare_set: str = "consensus",
    brands: str = "Brand A,Brand B,Brand C"
):
    """Get sales surprise analysis"""
    brand_list = brands.split(",") if brands else ["Brand A", "Brand B", "Brand C"]
    
    data = []
    for brand in brand_list:
        expected = 95000 + random.uniform(-15000, 20000)
        surprise_pct = random.uniform(-0.25, 0.35)  # -25% to +35%
        actual = expected * (1 + surprise_pct)
        z_score = surprise_pct / 0.12  # Assuming 12% standard deviation
        
        data.append({
            "brand": brand.strip(),
            "actual_sales": round(actual, 0),
            "expected_sales": round(expected, 0),
            "surprise_pct": round(surprise_pct, 3),
            "z_score": round(z_score, 2),
            "compare_method": compare_set
        })
    
    return data

# =============================================================================
# 2. AD SPEND & EFFICIENCY APP WIDGETS
# =============================================================================

@register_widget({
    "name": "Ad Spend Trend",
    "description": "Time series ad spend by platform/channel",
    "category": "Advertising",
    "subcategory": "Spend Analytics",
    "widgetId": "ad_spend_trend",
    "gridData": {"x": 0, "y": 0, "w": 25, "h": 12},
    "endpoint": "ad_spend_trend",
    "type": "table",
    "data": {
        "table": {
            "enableCharts": True,
            "chartView": {
                "enabled": True,
                "chartType": "line"
            },
            "columnsDefs": [
                {"field": "date", "headerName": "Date", "chartDataType": "time", "width": 120},
                {"field": "facebook_spend", "headerName": "Facebook ($)", "chartDataType": "series", "formatterFn": "int", "width": 120},
                {"field": "google_spend", "headerName": "Google ($)", "chartDataType": "series", "formatterFn": "int", "width": 120},
                {"field": "tiktok_spend", "headerName": "TikTok ($)", "chartDataType": "series", "formatterFn": "int", "width": 120},
                {"field": "total_spend", "headerName": "Total Spend ($)", "chartDataType": "series", "formatterFn": "int", "width": 120}
            ]
        }
    },
    "params": [
        {
            "paramName": "platform",
            "type": "text",
            "multiSelect": True,
            "label": "Platform",
            "value": "facebook,google",
            "options": [
                {"label": "Facebook", "value": "facebook"},
                {"label": "Google", "value": "google"},
                {"label": "TikTok", "value": "tiktok"},
                {"label": "YouTube", "value": "youtube"}
            ]
        }
    ]
})
@app.get("/ad_spend_trend")
def get_ad_spend_trend(
    platform: str = "facebook,google",
    date_range_start: str = "2024-01-01",
    date_range_end: str = "2024-12-31"
):
    """Get ad spend trends by platform"""
    dates, _ = generate_time_series(date_range_start, date_range_end, 'D', base_value=1000)
    
    data = []
    for date in dates:
        facebook_spend = 8000 + random.uniform(-1000, 2000)
        google_spend = 12000 + random.uniform(-2000, 3000)
        tiktok_spend = 3000 + random.uniform(-500, 1500)
        youtube_spend = 2000 + random.uniform(-300, 800)
        
        data.append({
            "date": date.strftime("%Y-%m-%d"),
            "facebook_spend": round(facebook_spend, 0),
            "google_spend": round(google_spend, 0),
            "tiktok_spend": round(tiktok_spend, 0),
            "youtube_spend": round(youtube_spend, 0),
            "total_spend": round(facebook_spend + google_spend + tiktok_spend + youtube_spend, 0)
        })
    
    return data

@register_widget({
    "name": "ROAS Panel",
    "description": "ROAS metrics with spend vs sales correlation analysis",
    "category": "Advertising",
    "subcategory": "Performance Analytics",
    "widgetId": "roas_panel",
    "gridData": {"x": 25, "y": 0, "w": 25, "h": 12},
    "endpoint": "roas_panel", 
    "type": "table",
    "data": {
        "table": {
            "columnsDefs": [
                {"field": "platform", "headerName": "Platform", "width": 120},
                {"field": "spend", "headerName": "Spend ($)", "formatterFn": "int", "width": 120},
                {"field": "revenue", "headerName": "Revenue ($)", "formatterFn": "int", "width": 120},
                {"field": "roas", "headerName": "ROAS", "formatterFn": "none", "renderFn": "greenRed", "width": 100},
                {"field": "roas_median", "headerName": "ROAS Median", "formatterFn": "none", "width": 120},
                {"field": "roas_p95", "headerName": "ROAS P95", "formatterFn": "none", "width": 120},
                {"field": "best_lag_days", "headerName": "Best Lag (days)", "formatterFn": "int", "width": 120}
            ]
        }
    },
    "params": [
        {
            "paramName": "lag_window",
            "type": "text",
            "label": "Lag Window (weeks)",
            "value": "4",
            "options": [
                {"label": "2 weeks", "value": "2"},
                {"label": "4 weeks", "value": "4"},
                {"label": "6 weeks", "value": "6"},
                {"label": "8 weeks", "value": "8"}
            ]
        }
    ]
})
@app.get("/roas_panel")
def get_roas_panel(
    lag_window: str = "4",
    brands: str = "Brand A"
):
    """Get ROAS analysis panel"""
    platforms = ["Facebook", "Google", "TikTok", "YouTube"]
    
    data = []
    for platform in platforms:
        spend = 15000 + random.uniform(-5000, 8000)
        roas = 2.5 + random.uniform(-0.8, 1.5)  # 1.7 to 4.0 ROAS
        revenue = spend * roas
        
        roas_median = roas * 0.9  # Slightly lower median
        roas_p95 = roas * 1.4    # Higher p95
        best_lag_days = random.randint(7, int(lag_window) * 7)
        
        data.append({
            "platform": platform,
            "spend": round(spend, 0),
            "revenue": round(revenue, 0),
            "roas": round(roas, 2),
            "roas_median": round(roas_median, 2),
            "roas_p95": round(roas_p95, 2),
            "best_lag_days": best_lag_days,
            "brand": brands.split(",")[0] if brands else "Brand A"
        })
    
    return data

@register_widget({
    "name": "Share of Voice vs Share of Sales",
    "description": "Bubble chart showing SOV (spend %) vs SOS (sales %) efficiency",
    "category": "Advertising",
    "subcategory": "Efficiency Analytics",
    "widgetId": "share_voice_sales",
    "gridData": {"x": 0, "y": 12, "w": 25, "h": 10},
    "endpoint": "share_voice_sales",
    "type": "table",
    "data": {
        "table": {
            "enableCharts": True,
            "chartView": {
                "enabled": True,
                "chartType": "bubble"
            },
            "columnsDefs": [
                {"field": "brand", "headerName": "Brand", "chartDataType": "category", "width": 100},
                {"field": "share_of_voice", "headerName": "Share of Voice (%)", "chartDataType": "series", "formatterFn": "percent", "width": 140},
                {"field": "share_of_sales", "headerName": "Share of Sales (%)", "chartDataType": "series", "formatterFn": "percent", "width": 140},
                {"field": "efficiency_ratio", "headerName": "Efficiency Ratio", "formatterFn": "none", "renderFn": "greenRed", "width": 120},
                {"field": "total_spend", "headerName": "Total Spend ($)", "formatterFn": "int", "width": 120}
            ]
        }
    },
    "params": [
        {
            "paramName": "brands",
            "type": "text",
            "multiSelect": True,
            "label": "Brands",
            "value": "Brand A",
            "options": [
                {"label": "Brand A", "value": "Brand A"},
                {"label": "Brand B", "value": "Brand B"},
                {"label": "Brand C", "value": "Brand C"},
                {"label": "Competitor X", "value": "Competitor X"},
                {"label": "Competitor Y", "value": "Competitor Y"}
            ]
        }
    ]
})
@app.get("/share_voice_sales")
def get_share_voice_sales(
    brands: str = "Brand A,Brand B,Brand C,Competitor X",
    date_range_start: str = "2024-01-01",
    date_range_end: str = "2024-12-31"
):
    """Get share of voice vs share of sales analysis"""
    brand_list = brands.split(",") if brands else ["Brand A", "Brand B", "Brand C", "Competitor X"]
    
    # Ensure shares add up to 100%
    sov_shares = np.random.dirichlet([3, 2, 2, 1.5])  # Weighted random shares
    sos_shares = np.random.dirichlet([2.5, 2.2, 1.8, 1.2])  # Different distribution
    
    data = []
    for i, brand in enumerate(brand_list):
        sov = sov_shares[i] if i < len(sov_shares) else 0.1
        sos = sos_shares[i] if i < len(sos_shares) else 0.1
        efficiency_ratio = sos / sov if sov > 0 else 0
        total_spend = 50000 * sov
        
        data.append({
            "brand": brand.strip(),
            "share_of_voice": round(sov, 3),
            "share_of_sales": round(sos, 3),
            "efficiency_ratio": round(efficiency_ratio, 2),
            "total_spend": round(total_spend, 0)
        })
    
    return data

@register_widget({
    "name": "Creative Saturation Proxy",
    "description": "Frequency/impressions per user vs marginal ROAS curve (diminishing returns)",
    "category": "Advertising",
    "subcategory": "Creative Analytics", 
    "widgetId": "creative_saturation",
    "gridData": {"x": 25, "y": 12, "w": 25, "h": 10},
    "endpoint": "creative_saturation",
    "type": "table",
    "data": {
        "table": {
            "enableCharts": True,
            "chartView": {
                "enabled": True,
                "chartType": "scatter"
            },
            "columnsDefs": [
                {"field": "frequency", "headerName": "Frequency", "chartDataType": "series", "formatterFn": "none", "width": 100},
                {"field": "impressions_per_user", "headerName": "Impressions/User", "chartDataType": "category", "formatterFn": "none", "width": 130},
                {"field": "marginal_roas", "headerName": "Marginal ROAS", "chartDataType": "series", "formatterFn": "none", "width": 120},
                {"field": "total_impressions", "headerName": "Total Impressions", "formatterFn": "int", "width": 130},
                {"field": "spend", "headerName": "Spend ($)", "formatterFn": "int", "width": 100}
            ]
        }
    },
    "params": [
        {
            "paramName": "platform",
            "type": "text",
            "label": "Platform",
            "value": "facebook",
            "options": [
                {"label": "Facebook", "value": "facebook"},
                {"label": "Google", "value": "google"},
                {"label": "TikTok", "value": "tiktok"},
                {"label": "YouTube", "value": "youtube"}
            ]
        }
    ]
})
@app.get("/creative_saturation")
def get_creative_saturation(
    platform: str = "facebook",
    brands: str = "Brand A"
):
    """Get creative saturation analysis showing diminishing returns"""
    # Generate frequency buckets showing diminishing returns
    frequencies = np.arange(1.0, 8.1, 0.5)
    
    data = []
    for freq in frequencies:
        impressions_per_user = freq * (5 + random.uniform(-1, 2))
        # Diminishing returns curve: high initial ROAS, declining with frequency
        marginal_roas = max(0.2, 4.5 * np.exp(-0.3 * freq) + random.uniform(-0.2, 0.3))
        total_impressions = int(impressions_per_user * (50000 + random.uniform(-10000, 15000)))
        spend = total_impressions * 0.02  # CPM-based spend
        
        data.append({
            "frequency": round(freq, 1),
            "impressions_per_user": round(impressions_per_user, 1),
            "marginal_roas": round(marginal_roas, 2),
            "total_impressions": total_impressions,
            "spend": round(spend, 0),
            "platform": platform,
            "brand": brands.split(",")[0] if brands else "Brand A"
        })
    
    return data

# =============================================================================
# 3. DIGITAL ENGAGEMENT APP WIDGETS
# =============================================================================

@register_widget({
    "name": "DAU/MAU & Stickiness",
    "description": "DAU, MAU, DAU/MAU ratio and retention cohorts",
    "category": "Digital Engagement",
    "subcategory": "User Analytics",
    "widgetId": "dau_mau_stickiness",
    "gridData": {"x": 0, "y": 0, "w": 25, "h": 12},
    "endpoint": "dau_mau_stickiness",
    "type": "table",
    "data": {
        "table": {
            "enableCharts": True,
            "chartView": {
                "enabled": True,
                "chartType": "line"
            },
            "columnsDefs": [
                {"field": "date", "headerName": "Date", "chartDataType": "time", "width": 120},
                {"field": "dau", "headerName": "DAU", "chartDataType": "series", "formatterFn": "int", "width": 100},
                {"field": "mau", "headerName": "MAU", "chartDataType": "series", "formatterFn": "int", "width": 100},
                {"field": "stickiness", "headerName": "Stickiness (DAU/MAU)", "chartDataType": "series", "formatterFn": "percent", "width": 150},
                {"field": "retention_7d", "headerName": "7-Day Retention (%)", "formatterFn": "percent", "width": 150}
            ]
        }
    },
    "params": [
        {
            "paramName": "platform",
            "type": "text",
            "label": "Platform",
            "value": "all",
            "options": [
                {"label": "All Platforms", "value": "all"},
                {"label": "iOS", "value": "ios"},
                {"label": "Android", "value": "android"},
                {"label": "Web", "value": "web"}
            ]
        }
    ]
})
@app.get("/dau_mau_stickiness")
def get_dau_mau_stickiness(
    platform: str = "all",
    brands: str = "Brand A",
    date_range_start: str = "2024-01-01",
    date_range_end: str = "2024-12-31"
):
    """Get DAU/MAU and stickiness metrics"""
    dates, _ = generate_time_series(date_range_start, date_range_end, 'D', base_value=100000)
    
    data = []
    for i, date in enumerate(dates):
        base_mau = 150000 + random.uniform(-20000, 30000)
        seasonal_factor = 1 + 0.1 * np.sin(2 * np.pi * i / 30)  # Monthly seasonality
        mau = int(base_mau * seasonal_factor)
        
        stickiness = 0.15 + random.uniform(-0.05, 0.08)  # 10% - 23% stickiness
        dau = int(mau * stickiness)
        
        retention_7d = 0.25 + random.uniform(-0.08, 0.12)  # 17% - 37% retention
        
        data.append({
            "date": date.strftime("%Y-%m-%d"),
            "dau": dau,
            "mau": mau,
            "stickiness": round(stickiness, 3),
            "retention_7d": round(retention_7d, 3),
            "platform": platform,
            "brand": brands.split(",")[0] if brands else "Brand A"
        })
    
    return data

@register_widget({
    "name": "Web Traffic Share",
    "description": "Stacked area share with gains/losses table",
    "category": "Digital Engagement",
    "subcategory": "Traffic Analytics",
    "widgetId": "web_traffic_share",
    "gridData": {"x": 25, "y": 0, "w": 25, "h": 12},
    "endpoint": "web_traffic_share",
    "type": "table",
    "data": {
        "table": {
            "enableCharts": True,
            "chartView": {
                "enabled": True,
                "chartType": "stackedArea"
            },
            "columnsDefs": [
                {"field": "date", "headerName": "Date", "chartDataType": "time", "width": 120},
                {"field": "organic", "headerName": "Organic Traffic", "chartDataType": "series", "formatterFn": "int", "width": 130},
                {"field": "paid", "headerName": "Paid Traffic", "chartDataType": "series", "formatterFn": "int", "width": 130},
                {"field": "social", "headerName": "Social Traffic", "chartDataType": "series", "formatterFn": "int", "width": 130},
                {"field": "direct", "headerName": "Direct Traffic", "chartDataType": "series", "formatterFn": "int", "width": 130}
            ]
        }
    },
    "params": [
        {
            "paramName": "brands",
            "type": "text",
            "multiSelect": True,
            "label": "Brands",
            "value": "Brand A",
            "options": [
                {"label": "Brand A", "value": "Brand A"},
                {"label": "Brand B", "value": "Brand B"},
                {"label": "Brand C", "value": "Brand C"}
            ]
        }
    ]
})
@app.get("/web_traffic_share")
def get_web_traffic_share(
    brands: str = "Brand A",
    date_range_start: str = "2024-01-01",
    date_range_end: str = "2024-12-31"
):
    """Get web traffic share analysis"""
    dates = pd.date_range(start=date_range_start, end=date_range_end, freq='W')
    
    data = []
    for i, date in enumerate(dates):
        total_traffic = 25000 + random.uniform(-3000, 5000)
        
        # Evolving shares over time
        organic_share = 0.45 - 0.001 * i + random.uniform(-0.02, 0.02)  # Declining
        paid_share = 0.25 + 0.0008 * i + random.uniform(-0.015, 0.015)    # Growing
        social_share = 0.15 + 0.0005 * i + random.uniform(-0.01, 0.01)    # Growing
        direct_share = 1 - organic_share - paid_share - social_share
        
        data.append({
            "date": date.strftime("%Y-%m-%d"),
            "organic": round(total_traffic * organic_share, 0),
            "paid": round(total_traffic * paid_share, 0),
            "social": round(total_traffic * social_share, 0),
            "direct": round(total_traffic * direct_share, 0),
            "brand": brands.split(",")[0] if brands else "Brand A"
        })
    
    return data

@register_widget({
    "name": "Engagement to Sales Funnel",
    "description": "Multi-panel funnel showing visits → adds-to-cart → conversions with POS correlation",
    "category": "Digital Engagement",
    "subcategory": "Conversion Analytics",
    "widgetId": "engagement_funnel",
    "gridData": {"x": 0, "y": 12, "w": 25, "h": 10},
    "endpoint": "engagement_funnel",
    "type": "table",
    "data": {
        "table": {
            "enableCharts": True,
            "chartView": {
                "enabled": True,
                "chartType": "column"
            },
            "columnsDefs": [
                {"field": "stage", "headerName": "Funnel Stage", "chartDataType": "category", "width": 130},
                {"field": "users", "headerName": "Users", "chartDataType": "series", "formatterFn": "int", "width": 100},
                {"field": "conversion_rate", "headerName": "Conversion Rate (%)", "formatterFn": "percent", "renderFn": "greenRed", "width": 150},
                {"field": "pos_correlation", "headerName": "POS Correlation", "formatterFn": "none", "width": 130},
                {"field": "avg_time_to_convert", "headerName": "Avg Time to Convert (hrs)", "formatterFn": "none", "width": 180}
            ]
        }
    },
    "params": [
        {
            "paramName": "lag_window",
            "type": "text",
            "label": "Lag Window (days)",
            "value": "7",
            "options": [
                {"label": "3 days", "value": "3"},
                {"label": "7 days", "value": "7"},
                {"label": "14 days", "value": "14"},
                {"label": "30 days", "value": "30"}
            ]
        }
    ]
})
@app.get("/engagement_funnel")
def get_engagement_funnel(
    lag_window: str = "7",
    brands: str = "Brand A"
):
    """Get engagement to sales funnel analysis"""
    stages = [
        {"stage": "Visits", "base_users": 100000},
        {"stage": "Product Views", "base_users": 45000},
        {"stage": "Add to Cart", "base_users": 8500},
        {"stage": "Checkout Started", "base_users": 6200},
        {"stage": "Purchase Completed", "base_users": 4800},
        {"stage": "POS Purchase (within lag)", "base_users": 3200}
    ]
    
    data = []
    prev_users = None
    
    for i, stage_info in enumerate(stages):
        users = stage_info["base_users"] + random.randint(-500, 800)
        
        if prev_users:
            conversion_rate = users / prev_users
        else:
            conversion_rate = 1.0  # First stage is 100%
        
        # POS correlation decreases as we move down funnel
        pos_correlation = 0.85 - (i * 0.12) + random.uniform(-0.05, 0.05)
        pos_correlation = max(0.1, min(0.95, pos_correlation))
        
        # Time to convert increases down funnel
        avg_time_to_convert = (i + 1) * 2.5 + random.uniform(-0.5, 1.0)
        
        data.append({
            "stage": stage_info["stage"],
            "users": users,
            "conversion_rate": round(conversion_rate, 3),
            "pos_correlation": round(pos_correlation, 2),
            "avg_time_to_convert": round(avg_time_to_convert, 1),
            "brand": brands.split(",")[0] if brands else "Brand A"
        })
        
        prev_users = users
    
    return data

@register_widget({
    "name": "Review/Ratings Pulse",
    "description": "Ratings trend and sentiment analysis (mock)",
    "category": "Digital Engagement",
    "subcategory": "Sentiment Analytics",
    "widgetId": "review_ratings_pulse",
    "gridData": {"x": 25, "y": 12, "w": 25, "h": 10},
    "endpoint": "review_ratings_pulse",
    "type": "table",
    "data": {
        "table": {
            "enableCharts": True,
            "chartView": {
                "enabled": True,
                "chartType": "line"
            },
            "columnsDefs": [
                {"field": "date", "headerName": "Date", "chartDataType": "time", "width": 120},
                {"field": "avg_rating", "headerName": "Avg Rating", "chartDataType": "series", "formatterFn": "none", "width": 110},
                {"field": "review_count", "headerName": "Review Count", "chartDataType": "series", "formatterFn": "int", "width": 120},
                {"field": "sentiment_score", "headerName": "Sentiment Score", "formatterFn": "none", "renderFn": "greenRed", "width": 130},
                {"field": "response_rate", "headerName": "Response Rate (%)", "formatterFn": "percent", "width": 130}
            ]
        }
    },
    "params": [
        {
            "paramName": "platform",
            "type": "text",
            "multiSelect": True,
            "label": "Platform",
            "value": "google,yelp",
            "options": [
                {"label": "Google", "value": "google"},
                {"label": "Yelp", "value": "yelp"},
                {"label": "Facebook", "value": "facebook"},
                {"label": "TripAdvisor", "value": "tripadvisor"},
                {"label": "Amazon", "value": "amazon"}
            ]
        }
    ]
})
@app.get("/review_ratings_pulse")
def get_review_ratings_pulse(
    platform: str = "google,yelp",
    brands: str = "Brand A"
):
    """Get review and ratings pulse analysis"""
    dates = pd.date_range(start="2024-01-01", end="2024-12-31", freq='W')
    
    data = []
    for i, date in enumerate(dates):
        # Simulate ratings with some trend and seasonality
        base_rating = 4.2 + 0.001 * i  # Slight improvement over time
        seasonal_effect = 0.1 * np.sin(2 * np.pi * i / 26)  # Bi-annual cycle
        avg_rating = base_rating + seasonal_effect + random.uniform(-0.15, 0.15)
        avg_rating = max(1.0, min(5.0, avg_rating))
        
        # Review count with seasonality (more reviews during holiday seasons)
        base_reviews = 150 + random.randint(-30, 50)
        holiday_boost = 1.3 if i % 26 in [22, 23, 24, 25, 0, 1] else 1.0  # Holiday weeks
        review_count = int(base_reviews * holiday_boost)
        
        # Sentiment score correlated with rating
        sentiment_score = (avg_rating - 3.0) / 2.0 + random.uniform(-0.1, 0.1)  # -1 to 1 scale
        sentiment_score = max(-1.0, min(1.0, sentiment_score))
        
        # Response rate
        response_rate = 0.75 + random.uniform(-0.15, 0.20)
        response_rate = max(0.0, min(1.0, response_rate))
        
        data.append({
            "date": date.strftime("%Y-%m-%d"),
            "avg_rating": round(avg_rating, 2),
            "review_count": review_count,
            "sentiment_score": round(sentiment_score, 3),
            "response_rate": round(response_rate, 3),
            "platform": platform.split(",")[0] if platform else "google",
            "brand": brands.split(",")[0] if brands else "Brand A"
        })
    
    return data

# =============================================================================
# 4. MEDIA PERFORMANCE APP WIDGETS  
# =============================================================================

@register_widget({
    "name": "OTT Viewership Trends",
    "description": "Hours viewed per title/platform with completion rates",
    "category": "Media Performance",
    "subcategory": "Content Analytics",
    "widgetId": "ott_viewership",
    "gridData": {"x": 0, "y": 0, "w": 25, "h": 12},
    "endpoint": "ott_viewership",
    "type": "table",
    "data": {
        "table": {
            "enableCharts": True,
            "chartView": {
                "enabled": True,
                "chartType": "groupedColumn"
            },
            "columnsDefs": [
                {"field": "title", "headerName": "Title", "chartDataType": "category", "width": 150},
                {"field": "platform", "headerName": "Platform", "width": 100},
                {"field": "hours_viewed", "headerName": "Hours Viewed", "chartDataType": "series", "formatterFn": "int", "width": 120},
                {"field": "completion_rate", "headerName": "Completion Rate (%)", "formatterFn": "percent", "renderFn": "greenRed", "width": 150},
                {"field": "avg_rating", "headerName": "Avg Rating", "formatterFn": "none", "width": 110}
            ]
        }
    },
    "params": [
        {
            "paramName": "platform",
            "type": "text",
            "multiSelect": True,
            "label": "Platform",
            "value": "netflix",
            "options": [
                {"label": "Netflix", "value": "netflix"},
                {"label": "Hulu", "value": "hulu"},
                {"label": "Disney+", "value": "disney"},
                {"label": "HBO Max", "value": "hbo"}
            ]
        }
    ]
})
@app.get("/ott_viewership")
def get_ott_viewership(
    platform: str = "netflix,hulu",
    title_set: str = "all"
):
    """Get OTT viewership trends"""
    titles = [
        "Action Series Alpha", "Comedy Show Beta", "Drama Series Gamma", 
        "Documentary Delta", "Reality Show Epsilon", "Sci-Fi Series Zeta",
        "Romance Movie Eta", "Thriller Series Theta"
    ]
    platforms = platform.split(",") if platform else ["netflix", "hulu", "disney"]
    
    data = []
    for title in titles[:6]:  # Limit to 6 for display
        for plat in platforms[:2]:  # Limit platforms
            hours_viewed = random.randint(50000, 500000)
            completion_rate = random.uniform(0.35, 0.85)  # 35-85% completion
            avg_rating = random.uniform(3.2, 4.8)
            
            data.append({
                "title": title,
                "platform": plat.capitalize(),
                "hours_viewed": hours_viewed,
                "completion_rate": round(completion_rate, 3),
                "avg_rating": round(avg_rating, 1)
            })
    
    return data

@register_widget({
    "name": "Box Office Momentum", 
    "description": "Weekly box office performance with ranking trends",
    "category": "Media Performance",
    "subcategory": "Box Office Analytics",
    "widgetId": "box_office_momentum",
    "gridData": {"x": 25, "y": 0, "w": 25, "h": 12},
    "endpoint": "box_office_momentum",
    "type": "table",
    "data": {
        "table": {
            "enableCharts": True,
            "chartView": {
                "enabled": True,
                "chartType": "line"
            },
            "columnsDefs": [
                {"field": "week", "headerName": "Week", "chartDataType": "time", "width": 120},
                {"field": "box_office", "headerName": "Box Office ($M)", "chartDataType": "series", "formatterFn": "none", "width": 130},
                {"field": "rank", "headerName": "Rank", "formatterFn": "int", "width": 80},
                {"field": "theaters", "headerName": "Theaters", "formatterFn": "int", "width": 100},
                {"field": "per_theater_avg", "headerName": "Per Theater Avg ($)", "formatterFn": "int", "width": 140}
            ]
        }
    },
    "params": [
        {
            "paramName": "title_set",
            "type": "text",
            "label": "Movie Title",
            "value": "blockbuster_a",
            "options": [
                {"label": "Blockbuster A", "value": "blockbuster_a"},
                {"label": "Comedy Hit B", "value": "comedy_b"},
                {"label": "Drama C", "value": "drama_c"},
                {"label": "Action Film D", "value": "action_d"}
            ]
        }
    ]
})
@app.get("/box_office_momentum")
def get_box_office_momentum(
    title_set: str = "blockbuster_a"
):
    """Get box office momentum data"""
    # Generate 12 weeks of box office data
    weeks = pd.date_range(start="2024-01-07", periods=12, freq='W')
    
    data = []
    initial_bo = 45.5  # $45.5M opening weekend
    
    for i, week in enumerate(weeks):
        # Typical box office decline curve
        if i == 0:
            box_office = initial_bo
            rank = 1
            theaters = 4200
        else:
            # Natural decline with some volatility
            decline_factor = 0.65 ** i  # Exponential decline
            box_office = initial_bo * decline_factor * random.uniform(0.8, 1.2)
            rank = min(1 + i + random.randint(-1, 2), 10)
            theaters = max(1000, 4200 - i * 300 + random.randint(-200, 100))
        
        per_theater_avg = (box_office * 1000000) / theaters if theaters > 0 else 0
        
        data.append({
            "week": week.strftime("%Y-%m-%d"),
            "box_office": round(box_office, 1),
            "rank": int(rank),
            "theaters": int(theaters),
            "per_theater_avg": round(per_theater_avg, 0),
            "title": title_set
        })
    
    return data

@register_widget({
    "name": "Music Streams vs Ticket Sales",
    "description": "Streams vs secondary ticket price/volume with lead/lag correlation",
    "category": "Media Performance",
    "subcategory": "Music Analytics",
    "widgetId": "music_streams_tickets",
    "gridData": {"x": 0, "y": 12, "w": 25, "h": 10},
    "endpoint": "music_streams_tickets",
    "type": "table",
    "data": {
        "table": {
            "enableCharts": True,
            "chartView": {
                "enabled": True,
                "chartType": "scatter"
            },
            "columnsDefs": [
                {"field": "artist", "headerName": "Artist", "chartDataType": "category", "width": 130},
                {"field": "weekly_streams", "headerName": "Weekly Streams (M)", "chartDataType": "series", "formatterFn": "none", "width": 150},
                {"field": "avg_ticket_price", "headerName": "Avg Ticket Price ($)", "chartDataType": "series", "formatterFn": "int", "width": 150},
                {"field": "tickets_sold", "headerName": "Tickets Sold", "formatterFn": "int", "width": 120},
                {"field": "correlation_score", "headerName": "Stream-Ticket Correlation", "formatterFn": "none", "width": 180}
            ]
        }
    },
    "params": [
        {
            "paramName": "artist_set",
            "type": "text",
            "multiSelect": True,
            "label": "Artist Set",
            "value": "pop_stars,rock_bands",
            "options": [
                {"label": "Pop Stars", "value": "pop_stars"},
                {"label": "Rock Bands", "value": "rock_bands"},
                {"label": "Hip Hop Artists", "value": "hip_hop"},
                {"label": "Country Artists", "value": "country"},
                {"label": "Electronic DJs", "value": "electronic"}
            ]
        }
    ]
})
@app.get("/music_streams_tickets")
def get_music_streams_tickets(
    artist_set: str = "pop_stars,rock_bands",
    geo: str = "national"
):
    """Get music streams vs ticket sales correlation analysis"""
    
    # Different artist categories with different characteristics
    artist_categories = {
        "pop_stars": [
            {"name": "Pop Star Alpha", "base_streams": 45.2, "base_price": 185},
            {"name": "Pop Icon Beta", "base_streams": 38.7, "base_price": 220},
            {"name": "Teen Sensation Gamma", "base_streams": 52.1, "base_price": 125}
        ],
        "rock_bands": [
            {"name": "Rock Legends Delta", "base_streams": 28.4, "base_price": 165},
            {"name": "Alt Rock Epsilon", "base_streams": 19.8, "base_price": 95},
            {"name": "Metal Force Zeta", "base_streams": 15.3, "base_price": 85}
        ],
        "hip_hop": [
            {"name": "Hip Hop King Eta", "base_streams": 67.2, "base_price": 275},
            {"name": "Rap Collective Theta", "base_streams": 41.8, "base_price": 155}
        ]
    }
    
    data = []
    selected_categories = artist_set.split(",") if artist_set else ["pop_stars"]
    
    for category in selected_categories:
        if category in artist_categories:
            for artist_info in artist_categories[category]:
                # Add some randomness to base values
                weekly_streams = artist_info["base_streams"] + random.uniform(-5, 8)
                avg_ticket_price = artist_info["base_price"] + random.randint(-25, 40)
                
                # Ticket sales inversely correlated with price, positively with streams
                price_factor = max(0.3, 1.2 - (avg_ticket_price / 200))
                stream_factor = min(2.0, weekly_streams / 25)
                tickets_sold = int(5000 * price_factor * stream_factor + random.uniform(-800, 1200))
                tickets_sold = max(500, tickets_sold)
                
                # Correlation between streams and ticket demand (varies by artist type)
                if category == "pop_stars":
                    correlation_score = 0.75 + random.uniform(-0.1, 0.15)
                elif category == "rock_bands":
                    correlation_score = 0.62 + random.uniform(-0.12, 0.18)
                else:
                    correlation_score = 0.68 + random.uniform(-0.08, 0.12)
                
                correlation_score = max(0.1, min(0.95, correlation_score))
                
                data.append({
                    "artist": artist_info["name"],
                    "weekly_streams": round(weekly_streams, 1),
                    "avg_ticket_price": int(avg_ticket_price),
                    "tickets_sold": tickets_sold,
                    "correlation_score": round(correlation_score, 3),
                    "category": category,
                    "geography": geo
                })
    
    return data

@register_widget({
    "name": "Platform Share",
    "description": "OTT platform share by hours/MAU with market dynamics",
    "category": "Media Performance",
    "subcategory": "Platform Analytics",
    "widgetId": "platform_share",
    "gridData": {"x": 25, "y": 12, "w": 25, "h": 10},
    "endpoint": "platform_share",
    "type": "table",
    "data": {
        "table": {
            "enableCharts": True,
            "chartView": {
                "enabled": True,
                "chartType": "pie"
            },
            "columnsDefs": [
                {"field": "platform", "headerName": "Platform", "chartDataType": "category", "width": 120},
                {"field": "hours_share", "headerName": "Hours Share (%)", "chartDataType": "series", "formatterFn": "percent", "width": 130},
                {"field": "mau_share", "headerName": "MAU Share (%)", "formatterFn": "percent", "width": 120},
                {"field": "engagement_per_user", "headerName": "Hours/MAU", "formatterFn": "none", "width": 120},
                {"field": "yoy_change", "headerName": "YoY Change (pp)", "formatterFn": "none", "renderFn": "greenRed", "width": 140}
            ]
        }
    },
    "params": [
        {
            "paramName": "platform",
            "type": "text",
            "multiSelect": True,
            "label": "Platform",
            "value": "netflix,hulu,disney,hbo,amazon",
            "options": [
                {"label": "Netflix", "value": "netflix"},
                {"label": "Hulu", "value": "hulu"},
                {"label": "Disney+", "value": "disney"},
                {"label": "HBO Max", "value": "hbo"},
                {"label": "Amazon Prime", "value": "amazon"},
                {"label": "Apple TV+", "value": "apple"},
                {"label": "Paramount+", "value": "paramount"}
            ]
        }
    ]
})
@app.get("/platform_share")
def get_platform_share(
    platform: str = "netflix,hulu,disney,hbo,amazon"
):
    """Get OTT platform share analysis"""
    
    # Platform market data with realistic shares
    platform_data = {
        "netflix": {"base_hours_share": 0.32, "base_mau_share": 0.28, "yoy_trend": -0.02},
        "hulu": {"base_hours_share": 0.18, "base_mau_share": 0.22, "yoy_trend": 0.01},
        "disney": {"base_hours_share": 0.15, "base_mau_share": 0.18, "yoy_trend": 0.03},
        "hbo": {"base_hours_share": 0.12, "base_mau_share": 0.10, "yoy_trend": 0.015},
        "amazon": {"base_hours_share": 0.11, "base_mau_share": 0.12, "yoy_trend": 0.005},
        "apple": {"base_hours_share": 0.06, "base_mau_share": 0.05, "yoy_trend": 0.02},
        "paramount": {"base_hours_share": 0.06, "base_mau_share": 0.05, "yoy_trend": 0.01}
    }
    
    selected_platforms = platform.split(",") if platform else ["netflix", "hulu", "disney"]
    
    # Normalize shares to selected platforms
    total_hours_share = sum(platform_data[p]["base_hours_share"] for p in selected_platforms if p in platform_data)
    total_mau_share = sum(platform_data[p]["base_mau_share"] for p in selected_platforms if p in platform_data)
    
    data = []
    for plat in selected_platforms:
        if plat in platform_data:
            info = platform_data[plat]
            
            # Normalize to selected platforms and add some randomness
            hours_share = (info["base_hours_share"] / total_hours_share) + random.uniform(-0.02, 0.02)
            mau_share = (info["base_mau_share"] / total_mau_share) + random.uniform(-0.015, 0.015)
            
            # Engagement per user (hours/MAU ratio)
            engagement_per_user = hours_share / mau_share if mau_share > 0 else 1.0
            
            # YoY change with some randomness
            yoy_change = info["yoy_trend"] + random.uniform(-0.01, 0.01)
            
            data.append({
                "platform": plat.title(),
                "hours_share": round(max(0.01, hours_share), 3),
                "mau_share": round(max(0.01, mau_share), 3),
                "engagement_per_user": round(engagement_per_user, 2),
                "yoy_change": round(yoy_change, 3)
            })
    
    return data

@register_widget({
    "name": "Building Permits Tracker",
    "description": "Building permits time series with YoY comparisons (construction proxy)",
    "category": "Mobility & Macro",
    "subcategory": "Construction Analytics",
    "widgetId": "building_permits",
    "gridData": {"x": 0, "y": 12, "w": 25, "h": 10},
    "endpoint": "building_permits",
    "type": "table",
    "data": {
        "table": {
            "enableCharts": True,
            "chartView": {
                "enabled": True,
                "chartType": "line"
            },
            "columnsDefs": [
                {"field": "date", "headerName": "Date", "chartDataType": "time", "width": 120},
                {"field": "total_permits", "headerName": "Total Permits", "chartDataType": "series", "formatterFn": "int", "width": 130},
                {"field": "residential_permits", "headerName": "Residential", "chartDataType": "series", "formatterFn": "int", "width": 120},
                {"field": "commercial_permits", "headerName": "Commercial", "chartDataType": "series", "formatterFn": "int", "width": 120},
                {"field": "yoy_change", "headerName": "YoY Change (%)", "formatterFn": "percent", "renderFn": "greenRed", "width": 130}
            ]
        }
    },
    "params": [
        {
            "paramName": "geo",
            "type": "text",
            "label": "Geography",
            "value": "national",
            "options": [
                {"label": "National", "value": "national"},
                {"label": "Northeast", "value": "northeast"},
                {"label": "Southeast", "value": "southeast"},
                {"label": "Midwest", "value": "midwest"},
                {"label": "West", "value": "west"},
                {"label": "California", "value": "california"},
                {"label": "Texas", "value": "texas"},
                {"label": "Florida", "value": "florida"}
            ]
        }
    ]
})
@app.get("/building_permits")
def get_building_permits(
    geo: str = "national",
    date_range_start: str = "2024-01-01",
    date_range_end: str = "2024-12-31"
):
    """Get building permits tracker data"""
    dates = pd.date_range(start=date_range_start, end=date_range_end, freq='M')
    
    # Base permit levels vary by geography
    geo_factors = {
        "national": {"base": 140000, "volatility": 0.08},
        "california": {"base": 35000, "volatility": 0.12},
        "texas": {"base": 42000, "volatility": 0.10},
        "florida": {"base": 28000, "volatility": 0.15},
        "northeast": {"base": 25000, "volatility": 0.09},
        "southeast": {"base": 45000, "volatility": 0.11},
        "midwest": {"base": 32000, "volatility": 0.07},
        "west": {"base": 38000, "volatility": 0.13}
    }
    
    base_permits = geo_factors.get(geo, {"base": 140000, "volatility": 0.08})
    
    data = []
    for i, date in enumerate(dates):
        # Seasonal patterns (more permits in spring/summer)
        seasonal_factor = 1 + 0.15 * np.sin(2 * np.pi * (i + 3) / 12)  # Peak in June
        
        # Economic cycle and trend
        trend_factor = 1.0 + 0.002 * i  # Slight growth over time
        economic_noise = random.uniform(-0.1, 0.15)
        
        total_permits = base_permits["base"] * seasonal_factor * trend_factor * (1 + economic_noise)
        total_permits = max(1000, int(total_permits))
        
        # Split between residential and commercial
        residential_share = 0.72 + random.uniform(-0.05, 0.05)
        residential_permits = int(total_permits * residential_share)
        commercial_permits = total_permits - residential_permits
        
        # YoY change calculation (mock comparison to same month prior year)
        yoy_change = random.uniform(-0.20, 0.25)  # -20% to +25% YoY
        
        data.append({
            "date": date.strftime("%Y-%m"),
            "total_permits": total_permits,
            "residential_permits": residential_permits,
            "commercial_permits": commercial_permits,
            "yoy_change": round(yoy_change, 3),
            "geography": geo
        })
    
    return data

# =============================================================================
# 5. MOBILITY & MACRO APP WIDGETS
# =============================================================================

@register_widget({
    "name": "Weather Impact Index",
    "description": "Weather anomaly vs sales delta with regression analysis",
    "category": "Mobility & Macro",
    "subcategory": "Environmental Analytics", 
    "widgetId": "weather_impact",
    "gridData": {"x": 0, "y": 0, "w": 25, "h": 12},
    "endpoint": "weather_impact",
    "type": "table",
    "data": {
        "table": {
            "enableCharts": True,
            "chartView": {
                "enabled": True,
                "chartType": "scatter"
            },
            "columnsDefs": [
                {"field": "date", "headerName": "Date", "chartDataType": "time", "width": 120},
                {"field": "temp_anomaly", "headerName": "Temp Anomaly (°F)", "chartDataType": "series", "formatterFn": "none", "renderFn": "greenRed", "width": 140},
                {"field": "precip_anomaly", "headerName": "Precipitation Anomaly", "chartDataType": "series", "formatterFn": "none", "width": 150},
                {"field": "sales_impact", "headerName": "Sales Impact (%)", "formatterFn": "percent", "renderFn": "greenRed", "width": 130},
                {"field": "weather_adjusted_sales", "headerName": "Weather-Adj Sales ($)", "formatterFn": "int", "width": 180}
            ]
        }
    },
    "params": [
        {
            "paramName": "geo",
            "type": "text",
            "label": "Geography",
            "value": "national",
            "options": [
                {"label": "National", "value": "national"},
                {"label": "Northeast", "value": "northeast"},
                {"label": "Southeast", "value": "southeast"},
                {"label": "Midwest", "value": "midwest"},
                {"label": "West", "value": "west"}
            ]
        }
    ]
})
@app.get("/weather_impact")
def get_weather_impact(
    geo: str = "national",
    metric_to_explain: str = "sales"
):
    """Get weather impact analysis"""
    dates = pd.date_range(start="2024-01-01", end="2024-12-31", freq='W')
    
    data = []
    for date in dates:
        # Generate weather anomalies
        temp_anomaly = random.uniform(-15, 20)  # Temperature anomaly in °F
        precip_anomaly = random.uniform(-2.5, 3.0)  # Precipitation anomaly
        
        # Weather impact on sales (simplified relationship)
        # Cold weather might boost indoor categories, hot weather outdoor categories
        sales_impact = (
            -0.003 * abs(temp_anomaly) +  # Extreme temps hurt sales
            -0.02 * max(0, precip_anomaly) +  # Excess rain hurts sales
            random.uniform(-0.05, 0.05)  # Random noise
        )
        
        base_sales = 50000
        actual_sales = base_sales * (1 + sales_impact)
        weather_adjusted_sales = base_sales  # What sales would be without weather impact
        
        data.append({
            "date": date.strftime("%Y-%m-%d"),
            "temp_anomaly": round(temp_anomaly, 1),
            "precip_anomaly": round(precip_anomaly, 1),
            "sales_impact": round(sales_impact, 4),
            "weather_adjusted_sales": round(weather_adjusted_sales, 0),
            "actual_sales": round(actual_sales, 0),
            "geography": geo
        })
    
    return data

@register_widget({
    "name": "Vehicle Registrations",
    "description": "Rolling 12-month chart with share by make/brand",
    "category": "Mobility & Macro",
    "subcategory": "Automotive Analytics",
    "widgetId": "vehicle_registrations", 
    "gridData": {"x": 25, "y": 0, "w": 25, "h": 12},
    "endpoint": "vehicle_registrations",
    "type": "table",
    "data": {
        "table": {
            "enableCharts": True,
            "chartView": {
                "enabled": True,
                "chartType": "stackedColumn"
            },
            "columnsDefs": [
                {"field": "month", "headerName": "Month", "chartDataType": "time", "width": 120},
                {"field": "toyota", "headerName": "Toyota", "chartDataType": "series", "formatterFn": "int", "width": 100},
                {"field": "ford", "headerName": "Ford", "chartDataType": "series", "formatterFn": "int", "width": 100},
                {"field": "honda", "headerName": "Honda", "chartDataType": "series", "formatterFn": "int", "width": 100},
                {"field": "chevrolet", "headerName": "Chevrolet", "chartDataType": "series", "formatterFn": "int", "width": 110},
                {"field": "total_registrations", "headerName": "Total", "formatterFn": "int", "width": 100}
            ]
        }
    },
    "params": [
        {
            "paramName": "make_brand",
            "type": "text",
            "multiSelect": True,
            "label": "Make/Brand",
            "value": "toyota,ford,honda",
            "options": [
                {"label": "Toyota", "value": "toyota"},
                {"label": "Ford", "value": "ford"},
                {"label": "Honda", "value": "honda"},
                {"label": "Chevrolet", "value": "chevrolet"},
                {"label": "Nissan", "value": "nissan"},
                {"label": "BMW", "value": "bmw"}
            ]
        }
    ]
})
@app.get("/vehicle_registrations")
def get_vehicle_registrations(
    make_brand: str = "toyota,ford,honda,chevrolet",
    geo: str = "national"
):
    """Get vehicle registration data"""
    months = pd.date_range(start="2024-01-01", end="2024-12-31", freq='M')
    
    data = []
    for month in months:
        # Market shares with some evolution over time
        total_market = 120000 + random.uniform(-15000, 20000)
        
        toyota_share = 0.15 + random.uniform(-0.01, 0.01)
        ford_share = 0.13 + random.uniform(-0.015, 0.01) 
        honda_share = 0.10 + random.uniform(-0.01, 0.015)
        chevrolet_share = 0.11 + random.uniform(-0.01, 0.01)
        other_share = 1 - toyota_share - ford_share - honda_share - chevrolet_share
        
        data.append({
            "month": month.strftime("%Y-%m"),
            "toyota": round(total_market * toyota_share, 0),
            "ford": round(total_market * ford_share, 0),
            "honda": round(total_market * honda_share, 0),
            "chevrolet": round(total_market * chevrolet_share, 0),
            "others": round(total_market * other_share, 0),
            "total_registrations": round(total_market, 0),
            "geography": geo
        })
    
    return data

@register_widget({
    "name": "Macro Composite Index",
    "description": "Composite score with z-scores per factor",
    "category": "Mobility & Macro",
    "subcategory": "Economic Indicators",
    "widgetId": "macro_composite",
    "gridData": {"x": 0, "y": 12, "w": 50, "h": 10},
    "endpoint": "macro_composite",
    "type": "table",
    "data": {
        "table": {
            "enableCharts": True,
            "chartView": {
                "enabled": True,
                "chartType": "line"
            },
            "columnsDefs": [
                {"field": "date", "headerName": "Date", "chartDataType": "time", "width": 120},
                {"field": "composite_score", "headerName": "Composite Score", "chartDataType": "series", "formatterFn": "none", "renderFn": "greenRed", "width": 130},
                {"field": "employment_zscore", "headerName": "Employment Z-Score", "formatterFn": "none", "width": 140},
                {"field": "housing_zscore", "headerName": "Housing Z-Score", "formatterFn": "none", "width": 130},
                {"field": "consumer_confidence_zscore", "headerName": "Consumer Confidence Z", "formatterFn": "none", "width": 160},
                {"field": "inflation_zscore", "headerName": "Inflation Z-Score", "formatterFn": "none", "width": 130}
            ]
        }
    },
    "params": [
        {
            "paramName": "weights_employment",
            "type": "text",
            "label": "Employment Weight",
            "value": "0.3",
            "options": [
                {"label": "Low (0.2)", "value": "0.2"},
                {"label": "Medium (0.3)", "value": "0.3"},
                {"label": "High (0.4)", "value": "0.4"}
            ]
        },
        {
            "paramName": "weights_housing",
            "type": "text",
            "label": "Housing Weight", 
            "value": "0.25",
            "options": [
                {"label": "Low (0.15)", "value": "0.15"},
                {"label": "Medium (0.25)", "value": "0.25"},
                {"label": "High (0.35)", "value": "0.35"}
            ]
        }
    ]
})
@app.get("/macro_composite")
def get_macro_composite(
    weights_employment: str = "0.3",
    weights_housing: str = "0.25",
    geo: str = "national"
):
    """Get macro composite index"""
    dates = pd.date_range(start="2024-01-01", end="2024-12-31", freq='W')
    
    # Parse weights
    w_employment = float(weights_employment)
    w_housing = float(weights_housing)
    w_consumer = 0.25
    w_inflation = 1 - w_employment - w_housing - w_consumer
    
    data = []
    for date in dates:
        # Generate z-scores for different macro factors
        employment_z = random.uniform(-2.5, 2.0)  # Generally positive employment
        housing_z = random.uniform(-1.5, 1.8)
        consumer_confidence_z = random.uniform(-2.0, 2.2)
        inflation_z = random.uniform(-1.0, -2.5)  # Negative is good (low inflation)
        
        # Calculate weighted composite score
        composite_score = (
            w_employment * employment_z +
            w_housing * housing_z + 
            w_consumer * consumer_confidence_z +
            w_inflation * inflation_z
        )
        
        data.append({
            "date": date.strftime("%Y-%m-%d"),
            "composite_score": round(composite_score, 2),
            "employment_zscore": round(employment_z, 2),
            "housing_zscore": round(housing_z, 2),
            "consumer_confidence_zscore": round(consumer_confidence_z, 2),
            "inflation_zscore": round(inflation_z, 2),
            "geography": geo
        })
    
    return data

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)