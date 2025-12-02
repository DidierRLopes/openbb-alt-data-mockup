#!/usr/bin/env python3
"""
Carbon Arc to OpenBB Workspace Integration
Dynamically generates OpenBB widgets from Carbon Arc framework data
"""

import json
import glob
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from datetime import datetime
from functools import wraps
import asyncio

# Initialize FastAPI application
app = FastAPI(
    title="Carbon Arc to OpenBB Backend",
    description="Dynamic widget generator for Carbon Arc data in OpenBB Workspace",
    version="1.0.0"
)

# Configure CORS
origins = [
    "https://pro.openbb.co",
    "https://pro.openbb.dev", 
    "http://localhost:1420",
    "http://localhost:3000",  # Common development port
    "http://localhost:5050",  # Another common port
    "*"  # Allow all origins for testing (remove in production)
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global storage for widgets and data
WIDGETS = {}
FRAMEWORK_DATA = {}
INSIGHT_GROUPS = {}

def register_widget(widget_config):
    """
    Decorator to register a widget configuration
    """
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            return await func(*args, **kwargs)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        
        endpoint = widget_config.get("endpoint")
        if endpoint:
            if "widgetId" not in widget_config:
                widget_config["widgetId"] = endpoint
            
            widget_id = widget_config["widgetId"]
            WIDGETS[widget_id] = widget_config
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    return decorator

def load_framework_data():
    """
    Load all framework JSON files and process the data
    """
    global FRAMEWORK_DATA, INSIGHT_GROUPS
    
    # Find all framework JSON files
    framework_files = glob.glob('frameworks/**/*.json', recursive=True)
    
    # Also check for framework-* directories
    framework_files.extend(glob.glob('framework-*/*.json', recursive=True))
    
    all_series_data = []
    
    print(f"Found {len(framework_files)} framework files")
    
    for file_path in framework_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Extract time series data
            series_list = data.get('data', [])
            all_series_data.extend(series_list)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue
    
    # Convert to DataFrame for easier processing
    if all_series_data:
        df = pd.DataFrame(all_series_data)
        
        # Group by insight type
        for insight_name, group_df in df.groupby('insight'):
            INSIGHT_GROUPS[insight_name] = group_df.reset_index(drop=True)
            
        FRAMEWORK_DATA['all_data'] = df
        print(f"Loaded {len(all_series_data)} time series from {len(framework_files)} files")
        print(f"Found {len(INSIGHT_GROUPS)} unique insight types: {list(INSIGHT_GROUPS.keys())}")
    else:
        print("No framework data found")

def create_insight_endpoints():
    """
    Create endpoints for each insight type found in the data
    """
    for insight_name, insight_data in INSIGHT_GROUPS.items():
        create_single_insight_endpoint(insight_name, insight_data)

def create_single_insight_endpoint(insight_name: str, insight_data: pd.DataFrame):
    """
    Create a single endpoint for a specific insight type
    """
    # Create clean endpoint name
    endpoint_slug = insight_name.lower().replace('_', '-')
    
    # Get unique values for filtering
    unique_entities = sorted(insight_data['entity_name'].unique().tolist())
    category = insight_data['entity_representation'].iloc[0] if not insight_data.empty else "unknown"
    
    # Extract unique filter keys and values
    filter_params = {}
    for filters_dict in insight_data['filters'].tolist():
        if isinstance(filters_dict, dict):
            for key, value in filters_dict.items():
                if key not in filter_params:
                    filter_params[key] = set()
                if not isinstance(value, dict):
                    filter_params[key].add(str(value))
    
    # Convert sets to sorted lists
    for key in filter_params:
        filter_params[key] = sorted(list(filter_params[key]))
    
    # Create widget configuration
    widget_config = {
        "name": insight_name.replace('_', ' ').title(),
        "description": f"Displays {insight_name.replace('_', ' ')} data from Carbon Arc",
        "endpoint": endpoint_slug,
        "widgetId": endpoint_slug,
        "gridData": {"w": 20, "h": 10},
        "type": "table",
        "category": category,
        "params": [],
        "data": {
            "table": {
                "enableCharts": True,
                "showAll": True,  # Show all data to handle dynamic columns
                "chartView": {
                    "enabled": True,
                    "chartType": "line"
                }
                # Remove columnsDefs to let OpenBB auto-detect columns
                # OpenBB will automatically set date as category and numeric columns as series
            }
        }
    }
    
    # Add entity parameter if there are multiple entities
    if len(unique_entities) > 0:
        entity_param = {
            "paramName": "entity_name",
            "description": "Select entity",
            "value": unique_entities[0] if unique_entities else "",
            "label": "Entity",
            "type": "text",
            "show": True,  # Added show parameter
            "options": [{"value": name, "label": name} for name in unique_entities]
        }
        widget_config["params"].append(entity_param)
    
    # Add filter parameters
    for key, values in filter_params.items():
        if values:
            filter_param = {
                "paramName": key,
                "description": f"Filter by {key.replace('_', ' ')}",
                "value": values[0] if values else "",
                "label": key.replace('_', ' ').title(),
                "type": "text",
                "multiSelect": True,
                "show": True,  # Added show parameter
                "options": [{"value": v, "label": v} for v in values]
            }
            widget_config["params"].append(filter_param)
    
    # Register the widget
    WIDGETS[endpoint_slug] = widget_config
    
    # Create the endpoint handler function
    # This is the key part - we need to create a proper function that FastAPI can use
    def make_endpoint_handler(data_df, filters_dict):
        """Create a closure that captures the data"""
        def endpoint_handler(
            entity_name: Optional[str] = None,
            **kwargs
        ):
            """Handle requests for this insight endpoint"""
            # Parse multiSelect parameters
            multi_select_params = {}
            for key in filters_dict.keys():
                param_value = kwargs.get(key)
                if param_value:
                    # Split comma-separated values for multiSelect parameters
                    multi_select_params[key] = [v.strip() for v in param_value.split(',')]
            
            # If we have multiple selections, we need to return a combined dataset
            if any(len(values) > 1 for values in multi_select_params.values()):
                # Collect data for each selected value combination
                combined_data = {}
                dates_set = set()
                
                # Start with base filtering
                base_df = data_df.copy()
                if entity_name:
                    base_df = base_df[base_df['entity_name'] == entity_name]
                
                # If we have multiple parameters with multiple values, we need cartesian product
                from itertools import product
                
                # Separate multi-value params from single-value params
                multi_params = {k: v for k, v in multi_select_params.items() if len(v) > 1}
                single_params = {k: v for k, v in multi_select_params.items() if len(v) == 1}
                
                # Process each unique combination
                all_combos = []
                
                if multi_params:
                    # Get all combinations of multi-select values
                    param_keys = list(multi_params.keys())
                    param_values = [multi_params[k] for k in param_keys]
                    
                    # Generate all combinations
                    for combo in product(*param_values):
                        # Create filter dict for this combination
                        combo_dict = dict(zip(param_keys, combo))
                        # Merge with single params
                        full_combo = {**combo_dict, **{k: v[0] for k, v in single_params.items()}}
                        all_combos.append(full_combo)
                else:
                    # Just single params
                    if single_params:
                        all_combos.append({k: v[0] for k, v in single_params.items()})
                
                # Process each combination
                for combo_dict in all_combos:
                    if combo_dict:
                        # Filter data for this exact combination
                        filtered_df = base_df.copy()
                        
                        # Apply all filters
                        for key, value in combo_dict.items():
                            filtered_df = filtered_df[
                                filtered_df['filters'].apply(
                                    lambda x: str(x.get(key)) == value if isinstance(x, dict) else False
                                )
                            ]
                        
                        if not filtered_df.empty:
                            # Get the series data
                            series_data = filtered_df.iloc[0]['series']
                            if isinstance(series_data, str):
                                try:
                                    series_data = json.loads(series_data)
                                except:
                                    continue
                            
                            if isinstance(series_data, list):
                                # Create descriptive column name from combo_dict
                                name_parts = []
                                
                                # Sort keys for consistent naming
                                sorted_keys = sorted(combo_dict.keys())
                                
                                for key in sorted_keys:
                                    value = combo_dict[key]
                                    # If only one parameter total, just use the value
                                    if len(combo_dict) == 1:
                                        name_parts.append(str(value))
                                    else:
                                        # Format as "param: value" for clarity when multiple params
                                        param_label = key.replace('_', ' ').title()
                                        name_parts.append(f"{param_label}: {value}")
                                
                                # Join all parts to create column name
                                col_name = ' | '.join(name_parts) if name_parts else "value"
                                
                                # Store data by date
                                for point in series_data:
                                    date = point.get('date')
                                    value = point.get('value')
                                    if date:
                                        dates_set.add(date)
                                        if date not in combined_data:
                                            combined_data[date] = {}
                                        combined_data[date][col_name] = value
                
                # Convert to list format expected by OpenBB
                if combined_data:
                    result = []
                    for date in sorted(dates_set):
                        row = {'date': date}
                        row.update(combined_data.get(date, {}))
                        result.append(row)
                    return result
                
            else:
                # Single selection - return as before
                result_df = data_df.copy()
                
                # Filter by entity if provided
                if entity_name:
                    result_df = result_df[result_df['entity_name'] == entity_name]
                
                # Apply additional filters
                for key, values in multi_select_params.items():
                    if values:
                        result_df = result_df[
                            result_df['filters'].apply(
                                lambda x: str(x.get(key)) == values[0] if isinstance(x, dict) else False
                            )
                        ]
                
                # Get the time series data
                if not result_df.empty:
                    series_data = result_df.iloc[0]['series']
                    
                    # Return the series data directly
                    if isinstance(series_data, list):
                        return series_data
                    elif isinstance(series_data, str):
                        try:
                            return json.loads(series_data)
                        except:
                            return []
            
            return []
        
        return endpoint_handler
    
    # Create the endpoint handler
    handler = make_endpoint_handler(insight_data, filter_params)
    
    # Add dynamic parameters to the function signature
    import inspect
    params = []
    params.append(
        inspect.Parameter(
            'entity_name',
            inspect.Parameter.KEYWORD_ONLY,
            default=Query(default=None, description="Entity name"),
            annotation=Optional[str]
        )
    )
    
    for key in filter_params.keys():
        params.append(
            inspect.Parameter(
                key,
                inspect.Parameter.KEYWORD_ONLY,
                default=Query(default=None, description=f"Filter by {key}"),
                annotation=Optional[str]
            )
        )
    
    handler.__signature__ = inspect.Signature(parameters=params)
    handler.__name__ = f"get_{endpoint_slug.replace('-', '_')}"
    
    # Register the route with FastAPI
    app.add_api_route(
        f"/{endpoint_slug}",
        handler,
        methods=["GET"],
        name=handler.__name__,
        tags=["Carbon Arc Data"]
    )
    
    print(f"Created endpoint: /{endpoint_slug}")

# Load data and create endpoints on startup
@app.on_event("startup")
async def startup_event():
    """Load framework data and create endpoints on startup"""
    load_framework_data()
    create_insight_endpoints()
    print(f"\nStartup complete!")
    print(f"Total widgets registered: {len(WIDGETS)}")
    print(f"Available endpoints: {list(WIDGETS.keys())}")

# Root endpoint
@app.get("/")
def read_root():
    """Root endpoint"""
    return {
        "name": "Carbon Arc to OpenBB Integration",
        "status": "running",
        "insights": list(INSIGHT_GROUPS.keys()),
        "total_series": len(FRAMEWORK_DATA.get('all_data', [])),
        "widgets_count": len(WIDGETS)
    }

# Widgets configuration endpoint
@app.get("/widgets.json")
def get_widgets():
    """Returns all registered widget configurations"""
    return WIDGETS

# # Apps configuration endpoint - serve static file
# @app.get("/apps.json")
# def get_apps():
#     """Returns the apps configuration for OpenBB Workspace"""
#     apps_file = Path("apps.json")
#     if apps_file.exists():
#         try:
#             with open(apps_file, 'r') as f:
#                 return json.load(f)
#         except Exception as e:
#             print(f"Error reading apps.json: {e}")
    
#     # Fallback if file doesn't exist
#     return {
#         "apps": [
#             {
#                 "name": "Carbon Arc Data",
#                 "description": "Time series data from Carbon Arc frameworks",
#                 "id": "carbon-arc",
#                 "widgets": list(WIDGETS.keys())
#             }
#         ]
#     }

# Helper endpoints for debugging
@app.get("/insights")
def get_insights():
    """Get a list of all available insights with details"""
    insights = []
    for insight_name, insight_data in INSIGHT_GROUPS.items():
        insights.append({
            "name": insight_name,
            "display_name": insight_name.replace('_', ' ').title(),
            "endpoint": f"/{insight_name.lower().replace('_', '-')}",
            "entity_count": insight_data['entity_name'].nunique(),
            "series_count": len(insight_data),
            "category": insight_data['entity_representation'].iloc[0] if not insight_data.empty else "unknown"
        })
    
    return insights

@app.get("/data-summary")
def get_data_summary():
    """Get a summary of all loaded data"""
    if 'all_data' not in FRAMEWORK_DATA or FRAMEWORK_DATA['all_data'].empty:
        return {"message": "No data loaded"}
    
    df = FRAMEWORK_DATA['all_data']
    
    summary = {
        "total_series": len(df),
        "insights": {
            insight: {
                "count": len(data),
                "entities": data['entity_name'].unique().tolist(),
                "category": data['entity_representation'].iloc[0] if not data.empty else "unknown"
            }
            for insight, data in INSIGHT_GROUPS.items()
        },
        "entity_types": df['entity_representation'].value_counts().to_dict(),
        "total_entities": df['entity_name'].nunique()
    }
    
    return summary

# Status widget
@register_widget({
    "name": "Carbon Arc Status",
    "description": "Shows the status of Carbon Arc data integration",
    "type": "markdown",
    "endpoint": "carbon_arc_status",
    "gridData": {"w": 12, "h": 6},
})
@app.get("/carbon_arc_status")
def carbon_arc_status():
    """Returns status information about the Carbon Arc integration"""
    
    num_insights = len(INSIGHT_GROUPS)
    num_series = len(FRAMEWORK_DATA.get('all_data', []))
    num_widgets = len(WIDGETS) - 1  # Subtract the status widget itself
    
    insights_list = "\n".join([f"- **{name.replace('_', ' ').title()}**: /{name.lower().replace('_', '-')}" 
                               for name in INSIGHT_GROUPS.keys()])
    
    return f"""# Carbon Arc Integration Status

## Data Summary
- **Total Time Series:** {num_series}
- **Insight Types:** {num_insights}
- **Generated Widgets:** {num_widgets}

## Available Endpoints
{insights_list}

## Usage
Select any widget from the left panel to view the data. 
Use the parameters to filter by entity and other dimensions.

## Last Updated
{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8036)