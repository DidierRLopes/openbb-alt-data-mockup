# Carbon Arc - OpenBB Workspace Widget Backend

A clean, minimal backend for OpenBB Workspace widgets demonstrating Carbon Arc mock data integration.

## Files Structure

```
├── carbon_arc_backend.py     # Main FastAPI backend with widgets
├── carbon_arc_apps.json      # OpenBB apps configuration  
├── reference-backend/        # Reference implementation
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

## Quick Start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the backend:
   ```bash
   python carbon_arc_backend.py
   ```

3. Access widgets configuration:
   ```
   http://localhost:8002/widgets.json
   ```

4. Access apps configuration:
   ```
   http://localhost:8002/apps.json
   ```

## Available App

### POS & Transactions Monitor
A comprehensive retail analytics dashboard with two tabs:

**Overview Tab:**
- **Sales Trend**: Revenue trends with seasonal patterns and multi-brand support
- **Average Ticket & Basket**: Average ticket size trends over time  
- **Channel Mix**: Revenue distribution across channels (retail, e-commerce, mobile, wholesale)

**Geography Tab:**
- **Geo Heatmap**: Geographic sales distribution with YoY growth indicators
- **Sales vs Spend Surprise**: Actual vs expected performance analysis

## Available Widgets

1. **Sales Trend** (`/sales_trend`)
   - Parameters: brands (multi-select dropdown)

2. **Average Ticket & Basket** (`/avg_ticket_basket`) 
   - Parameters: brands (multi-select dropdown)

3. **Channel Mix** (`/channel_mix`)
   - Parameters: brands (multi-select dropdown)

4. **Geo Heatmap** (`/geo_heatmap`)
   - Parameters: geo_level (dropdown: state/city/region)

5. **Sales vs Spend Surprise** (`/sales_surprise`)
   - Parameters: compare_set (dropdown: consensus/yoy/budget)

## Widget Features

- **Dark/Light Theme Support**: All widgets support theme switching
- **Proper Parameter Validation**: Uses correct OpenBB parameter format with `paramName`, `type: "text"`, and `options`
- **Multi-Select Support**: Brand parameters support multiple selection
- **Raw Data Support**: All widgets support `?raw=true` for JSON data
- **Responsive Design**: Proper grid sizing for OpenBB Workspace

## Testing Widgets

Test individual widgets with different parameters:

```bash
# Sales trend with multiple brands
curl "http://localhost:8002/sales_trend?brands=Brand%20A,Brand%20B"

# Geographic data at city level  
curl "http://localhost:8002/geo_heatmap?geo_level=city"

# Sales surprise with year-over-year comparison
curl "http://localhost:8002/sales_surprise?compare_set=yoy"

# Raw data format
curl "http://localhost:8002/channel_mix?raw=true"
```

## OpenBB Integration

To use with OpenBB Workspace:

1. Ensure backend is running on port 8002
2. Add to OpenBB Workspace as data source
3. Configure widgets from `/widgets.json` endpoint
4. Import apps from `/apps.json` endpoint

## Reference Backend

The `reference-backend/` folder contains the original implementation for comparison and reference.