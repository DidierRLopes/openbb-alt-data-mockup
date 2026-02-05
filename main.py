#!/usr/bin/env python3
"""
Carbon Arc to OpenBB Workspace Integration
Dynamically generates OpenBB widgets from Carbon Arc framework data
"""

import asyncio
import glob
import inspect
import json
import uuid
from datetime import datetime
from functools import wraps
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

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
ALL_SERIES = {}  # Store all series by series_name for Framework Comparison

# ==============================================================================
# GROUP FRAMEWORK APP - Storage and Persistence
# ==============================================================================
GROUPS = {}  # Store framework groups: {group_id: {name, description, frameworks}}
GROUPS_FILE = Path(__file__).parent / "groups.json"


def load_groups():
    """Load groups from JSON file"""
    global GROUPS
    if GROUPS_FILE.exists():
        try:
            with open(GROUPS_FILE, 'r') as f:
                GROUPS = json.load(f)
            print(f"Loaded {len(GROUPS)} groups from {GROUPS_FILE}")
        except Exception as e:
            print(f"Error loading groups: {e}")
            GROUPS = {}
    else:
        GROUPS = {}


def save_groups():
    """Save groups to JSON file"""
    try:
        with open(GROUPS_FILE, 'w') as f:
            json.dump(GROUPS, f, indent=2)
        print(f"Saved {len(GROUPS)} groups to {GROUPS_FILE}")
    except Exception as e:
        print(f"Error saving groups: {e}")

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
    global FRAMEWORK_DATA, INSIGHT_GROUPS, ALL_SERIES

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

        # Build ALL_SERIES dictionary keyed by series_name
        for item in all_series_data:
            series_name = item.get('series_name')
            if series_name:
                ALL_SERIES[series_name] = item

        FRAMEWORK_DATA['all_data'] = df
        print(f"Loaded {len(all_series_data)} time series from {len(framework_files)} files")
        print(f"Found {len(INSIGHT_GROUPS)} unique insight types: {list(INSIGHT_GROUPS.keys())}")
        print(f"Found {len(ALL_SERIES)} unique series names")
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
    
    # Remove _id parameters if there's a corresponding parameter without _id
    keys_to_remove = []
    for key in filter_params:
        if key.endswith('_id'):
            base_key = key[:-3]  # Remove '_id' suffix
            if base_key in filter_params:
                keys_to_remove.append(key)
    
    for key in keys_to_remove:
        del filter_params[key]
    
    # Create widget configuration
    widget_config = {
        "name": insight_name.replace('_', ' ').title(),
        "description": f"Displays {insight_name.replace('_', ' ')} data from Carbon Arc",
        "endpoint": endpoint_slug,
        "widgetId": endpoint_slug,
        "gridData": {"w": 20, "h": 10},
        "type": "table",
        "category": category,
        "params": [[], []],  # [[entity params], [filter params]]
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
    
    # Add entity parameter if there are multiple entities (first array)
    if len(unique_entities) > 0:
        entity_param = {
            "paramName": "entity_name",
            "description": "Select entity",
            "value": unique_entities[0] if unique_entities else "",  # Default to first entity
            "label": "Entity",
            "type": "text",
            "multiSelect": True,  # Enable multiSelect for entities
            "show": True,  # Added show parameter
            "options": [{"value": name, "label": name} for name in unique_entities]
        }
        widget_config["params"][0].append(entity_param)
    
    # Add filter parameters (sorted alphabetically) to second array
    for key in sorted(filter_params.keys()):
        values = filter_params[key]
        if values:
            filter_param = {
                "paramName": key,
                "description": f"Filter by {key.replace('_', ' ')}",
                "value": values[0] if values else "",  # Default to first value
                "label": key.replace('_', ' ').title(),
                "type": "text",
                "multiSelect": True,
                "show": True,  # Added show parameter
                "options": [{"value": v, "label": v} for v in values]
            }
            widget_config["params"][1].append(filter_param)
    
    # Register the widget
    WIDGETS[endpoint_slug] = widget_config
    
    # Create the endpoint handler function
    # This is the key part - we need to create a proper function that FastAPI can use
    def make_endpoint_handler(data_df, filters_dict):
        """Create a closure that captures the data"""
        # Get unique entities to check if entity should be included in column names
        unique_entities = sorted(data_df['entity_name'].unique().tolist())
        
        # Identify parameters with only one option (should be excluded from column names)
        single_option_params = set()
        for key, values in filters_dict.items():
            if len(values) == 1:
                single_option_params.add(key)
        
        # Check if entity_name has only one option
        if len(unique_entities) == 1:
            single_option_params.add('entity_name')
        
        def endpoint_handler(
            entity_name: Optional[str] = None,
            **kwargs
        ):
            """Handle requests for this insight endpoint"""
            # Parse entity_name as multiSelect parameter
            entity_names = []
            if entity_name:
                # Split comma-separated values for multiSelect entity parameter
                entity_names = [v.strip() for v in entity_name.split(',')]
            
            # Parse multiSelect parameters for filters
            multi_select_params = {}
            for key in filters_dict.keys():
                param_value = kwargs.get(key)
                if param_value:
                    # Split comma-separated values for multiSelect parameters
                    multi_select_params[key] = [v.strip() for v in param_value.split(',')]
            
            # Check if ALL required parameters have values
            # We need entity_name AND all filter parameters to be provided
            missing_params = []
            
            # Check entity_name
            if not entity_names:
                missing_params.append("entity_name")
            
            # Check filter parameters
            for key in filters_dict.keys():
                if not kwargs.get(key):
                    missing_params.append(key)
            
            # If any parameters are missing, raise an error with details
            if missing_params:
                if len(missing_params) == 1:
                    detail = f"Please select at least one {missing_params[0]}"
                elif len(missing_params) == 2:
                    detail = f"Please select at least one {missing_params[0]} and {missing_params[1]}"
                else:
                    # For 3 or more parameters
                    last_param = missing_params[-1]
                    other_params = ", ".join(missing_params[:-1])
                    detail = f"Please select at least one {other_params} and {last_param}"
                
                raise HTTPException(
                    status_code=400,
                    detail=detail
                )
            
            # Check if we have multiple selections (including entity)
            has_multiple_entities = len(entity_names) > 1
            has_multiple_filters = any(len(values) > 1 for values in multi_select_params.values())
            
            # If we have multiple selections, we need to return a combined dataset
            if has_multiple_entities or has_multiple_filters:
                # Collect data for each selected value combination
                combined_data = {}
                dates_set = set()
                
                # We don't pre-filter by entity anymore since it can be multiple
                base_df = data_df.copy()
                
                # If we have multiple parameters with multiple values, we need cartesian product
                # Include entity_names in the multi-params if there are multiple
                all_multi_params = {}
                if has_multiple_entities:
                    all_multi_params['entity_name'] = entity_names
                
                # Add filter parameters
                multi_params = {k: v for k, v in multi_select_params.items() if len(v) > 1}
                single_params = {k: v for k, v in multi_select_params.items() if len(v) == 1}
                all_multi_params.update(multi_params)
                
                # Process each unique combination
                all_combos = []
                
                if all_multi_params:
                    # Get all combinations of multi-select values (including entity)
                    param_keys = list(all_multi_params.keys())
                    param_values = [all_multi_params[k] for k in param_keys]
                    
                    # Generate all combinations
                    for combo in product(*param_values):
                        # Create combo dict for this combination
                        combo_dict = dict(zip(param_keys, combo))
                        # Merge with single params
                        full_combo = {**combo_dict, **{k: v[0] for k, v in single_params.items()}}
                        # Add single entity if not multiple
                        if not has_multiple_entities and entity_names:
                            full_combo['entity_name'] = entity_names[0]
                        all_combos.append(full_combo)
                else:
                    # Just single params and single entity
                    combo = {}
                    if single_params:
                        combo.update({k: v[0] for k, v in single_params.items()})
                    if entity_names:
                        combo['entity_name'] = entity_names[0]
                    if combo:
                        all_combos.append(combo)
                
                # Process each combination
                for combo_dict in all_combos:
                    if combo_dict:
                        # Filter data for this exact combination
                        filtered_df = base_df.copy()
                        
                        # First filter by entity if specified
                        if 'entity_name' in combo_dict:
                            filtered_df = filtered_df[filtered_df['entity_name'] == combo_dict['entity_name']]
                        
                        # Apply other filters
                        for key, value in combo_dict.items():
                            if key != 'entity_name' and 'filters' in filtered_df.columns:
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
                                # Create column name from values only
                                # Sort keys for consistent naming
                                sorted_keys = sorted(combo_dict.keys())
                                
                                # Filter out parameters that don't vary in this request
                                # Check which parameters actually have multiple values in all_combos
                                varying_params = set()
                                for param_key in sorted_keys:
                                    unique_vals = set(c.get(param_key) for c in all_combos if c.get(param_key))
                                    if len(unique_vals) > 1:
                                        varying_params.add(param_key)
                                
                                # Also exclude parameters with only one option globally
                                relevant_keys = [key for key in sorted_keys 
                                               if key in varying_params or (key not in single_option_params and key not in varying_params)]
                                
                                # Actually, we only want keys that vary in this request
                                relevant_keys = [key for key in sorted_keys if key in varying_params]
                                
                                # Get the values for relevant keys only
                                values = [str(combo_dict[key]) for key in relevant_keys]
                                
                                # If no relevant values (all params have single options), use a default name
                                if not values:
                                    col_name = "value"
                                # For single parameter, just use the value itself
                                elif len(values) == 1:
                                    col_name = values[0]
                                else:
                                    # Multiple parameters - join with |
                                    col_name = ' | '.join(values)
                                
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
                
                # Filter by entity if provided (take first one since it's single selection)
                if entity_names:
                    result_df = result_df[result_df['entity_name'] == entity_names[0]]
                
                # Apply additional filters
                for key, values in multi_select_params.items():
                    if values and 'filters' in result_df.columns:
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
    update_framework_comparison_params()  # Populate series options for comparison widget
    load_groups()  # Load saved groups for Group Framework app
    print(f"\nStartup complete!")
    print(f"Total widgets registered: {len(WIDGETS)}")
    print(f"Available endpoints: {list(WIDGETS.keys())}")
    print(f"Groups loaded: {len(GROUPS)}")

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

# Apps configuration endpoint - dynamically generate
@app.get("/apps.json")
def get_apps():
    """Returns the apps configuration for OpenBB Workspace"""

    # Widgets to exclude from the frameworks tab (special widgets and Group Framework widgets)
    excluded_widgets = {
        'carbon_arc_status',
        'framework-comparison',
        'series-list',
        # Group Framework app widgets - keep separate from Carbon Arc Frameworks
        'group-manager-html',
        'groups-list',
        'grouped-framework-comparison'
    }

    # Get all insight widgets (excluding special widgets)
    insight_widgets = [w for w in WIDGETS.keys() if w not in excluded_widgets]

    # Build the frameworks tab layout with all insight widgets
    frameworks_layout = []
    y_position = 0

    for widget_id in insight_widgets:
        widget_layout = {
            "i": widget_id,
            "x": 0,
            "y": y_position,
            "w": 40,
            "h": 11,
            "state": {
                "chartModel": {
                    "modelType": "range",
                    "chartType": "line",
                    "chartOptions": {},
                    "suppressChartRanges": True
                },
                "chartView": {
                    "enabled": True,
                    "chartType": "line"
                }
            },
            "groups": []
        }
        frameworks_layout.append(widget_layout)
        y_position += 11  # Stack widgets vertically

    # Return array of apps
    return [
        # App 1: Carbon Arc Frameworks
        {
            "name": "Carbon Arc Frameworks",
            "img": "https://media.licdn.com/dms/image/v2/D4E0BAQFBGi28odVUxg/company-logo_200_200/company-logo_200_200/0/1734471549764/carbonarc_logo?e=2147483647&v=beta&t=DlOWF5Orlwx_6NTDE7Xf8ivJwDSXgpug6y9ORIlOfNk",
            "img_dark": "",
            "img_light": "",
            "description": "Carbon Arc frameworks",
            "allowCustomization": True,
            "tabs": {
                "overview": {
                    "id": "overview",
                    "name": "Overview",
                    "layout": [
                        {
                            "i": "carbon_arc_status",
                            "x": 0,
                            "y": 0,
                            "w": 40,
                            "h": 25,
                            "groups": []
                        }
                    ]
                },
                "frameworks": {
                    "id": "frameworks",
                    "name": "Frameworks",
                    "layout": frameworks_layout
                }
            },
            "groups": []
        },
        # App 2: Framework Comparison
        {
            "name": "Framework Comparison",
            "img": "https://media.licdn.com/dms/image/v2/D4E0BAQFBGi28odVUxg/company-logo_200_200/company-logo_200_200/0/1734471549764/carbonarc_logo?e=2147483647&v=beta&t=DlOWF5Orlwx_6NTDE7Xf8ivJwDSXgpug6y9ORIlOfNk",
            "img_dark": "https://media.licdn.com/dms/image/v2/D4E0BAQFBGi28odVUxg/company-logo_200_200/company-logo_200_200/0/1734471549764/carbonarc_logo?e=2147483647&v=beta&t=DlOWF5Orlwx_6NTDE7Xf8ivJwDSXgpug6y9ORIlOfNk",
            "img_light": "https://media.licdn.com/dms/image/v2/D4E0BAQFBGi28odVUxg/company-logo_200_200/company-logo_200_200/0/1734471549764/carbonarc_logo?e=2147483647&v=beta&t=DlOWF5Orlwx_6NTDE7Xf8ivJwDSXgpug6y9ORIlOfNk",
            "description": "Compare framework series side by side",
            "allowCustomization": True,
            "tabs": {
                "": {
                    "id": "",
                    "name": "",
                    "layout": [
                        {
                            "i": "framework-comparison",
                            "x": 0,
                            "y": 15,
                            "w": 40,
                            "h": 15,
                            "state": {
                                "chartModel": {
                                    "modelType": "range",
                                    "chartType": "line",
                                    "chartOptions": {},
                                    "suppressChartRanges": True,
                                    "cellRange": {
                                        "columns": [
                                            "date",
                                            "AE + Aerie | App Downloads | iOS | 16 | Canada | v2025.11.1 | Reinstated On: 2025-12-01"
                                        ]
                                    }
                                },
                                "chartView": {
                                    "enabled": True,
                                    "chartType": "line"
                                },
                                "columnState": {
                                    "default": {
                                        "columnOrder": {
                                            "orderedColIds": [
                                                "date",
                                                "Dell | Advertising Count | Desktop Site | v2025.11.1 | Reinstated On: 2025-11-29",
                                                "Amazon | Average Website Traffic | Mobile Site | Male | Gender | v2025.11.1 | Reinstated On: 2025-12-01"
                                            ]
                                        }
                                    }
                                }
                            },
                            "groups": []
                        },
                        {
                            "i": "series-list",
                            "x": 0,
                            "y": 0,
                            "w": 40,
                            "h": 15,
                            "state": {
                                "chartView": {
                                    "enabled": False,
                                    "chartType": "line"
                                },
                                "columnState": {
                                    "default": {
                                        "columnOrder": {
                                            "orderedColIds": [
                                                "entity_name",
                                                "insight",
                                                "category",
                                                "data_points",
                                                "series_name"
                                            ]
                                        }
                                    }
                                }
                            },
                            "groups": []
                        }
                    ]
                }
            },
            "groups": []
        },
        # App 3: Group Framework
        {
            "name": "Group Framework",
            "img": "https://media.licdn.com/dms/image/v2/D4E0BAQFBGi28odVUxg/company-logo_200_200/company-logo_200_200/0/1734471549764/carbonarc_logo?e=2147483647&v=beta&t=DlOWF5Orlwx_6NTDE7Xf8ivJwDSXgpug6y9ORIlOfNk",
            "img_dark": "https://media.licdn.com/dms/image/v2/D4E0BAQFBGi28odVUxg/company-logo_200_200/company-logo_200_200/0/1734471549764/carbonarc_logo?e=2147483647&v=beta&t=DlOWF5Orlwx_6NTDE7Xf8ivJwDSXgpug6y9ORIlOfNk",
            "img_light": "https://media.licdn.com/dms/image/v2/D4E0BAQFBGi28odVUxg/company-logo_200_200/company-logo_200_200/0/1734471549764/carbonarc_logo?e=2147483647&v=beta&t=DlOWF5Orlwx_6NTDE7Xf8ivJwDSXgpug6y9ORIlOfNk",
            "description": "Organize frameworks into groups for easier comparison",
            "allowCustomization": True,
            "tabs": {
                "": {
                    "id": "",
                    "name": "",
                    "layout": [
                        {
                            "i": "group-manager-html",
                            "x": 0,
                            "y": 0,
                            "w": 40,
                            "h": 16,
                            "groups": []
                        },
                        {
                            "i": "groups-list",
                            "x": 0,
                            "y": 16,
                            "w": 40,
                            "h": 10,
                            "state": {
                                "chartView": {
                                    "enabled": False,
                                    "chartType": "line"
                                },
                                "columnState": {
                                    "default": {
                                        "columnPinning": {
                                            "leftColIds": ["name"],
                                            "rightColIds": []
                                        },
                                        "columnOrder": {
                                            "orderedColIds": [
                                                "name",
                                                "description",
                                                "framework_count",
                                                "created_at"
                                            ]
                                        }
                                    }
                                }
                            },
                            "groups": ["Group 1", "Group 2"]
                        },
                        {
                            "i": "grouped-framework-comparison",
                            "x": 0,
                            "y": 26,
                            "w": 40,
                            "h": 15,
                            "state": {
                                "paramOrder": ["group_name", "series_name"],
                                "params": {
                                    "group_name": "Theo",
                                    "series_name": [
                                        "Amazon | Average Website Traffic | Desktop Site | Male | Gender | v2025.11.1 | Reinstated On: 2025-12-01",
                                        "Amazon | Average Website Traffic | Mobile Site | Female | Gender | v2025.11.1 | Reinstated On: 2025-12-01"
                                    ]
                                },
                                "chartModel": {
                                    "modelType": "range",
                                    "chartType": "line",
                                    "chartOptions": {
                                        "line": {
                                            "series": {
                                                "label": {"enabled": False},
                                                "tooltip": {"enabled": True}
                                            }
                                        }
                                    },
                                    "suppressChartRanges": True,
                                    "cellRange": {
                                        "columns": [
                                            "date",
                                            "Amazon | Average Website Traffic | Desktop Site | Male | Gender | v2025.11.1 | Reinstated On: 2025-12-01",
                                            "Amazon | Average Website Traffic | Mobile Site | Female | Gender | v2025.11.1 | Reinstated On: 2025-12-01"
                                        ]
                                    }
                                },
                                "chartView": {
                                    "enabled": True,
                                    "chartType": "line"
                                }
                            },
                            "groups": ["Group 1", "Group 2"]
                        }
                    ]
                }
            },
            "groups": [
                {
                    "name": "Group 1",
                    "type": "param",
                    "paramName": "group_name",
                    "defaultValue": "Theo"
                },
                {
                    "name": "Group 2",
                    "type": "endpointParam",
                    "paramName": "group_name",
                    "defaultValue": "Theo"
                }
            ]
        }
    ]

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


# ==================== Framework Comparison App ====================

# Series List Widget - shows all available series names
@register_widget({
    "name": "Series List",
    "description": "List of all available framework series",
    "type": "table",
    "endpoint": "series-list",
    "gridData": {"w": 40, "h": 15},
    "data": {
        "table": {
            "enableCharts": False,
            "showAll": True
        }
    }
})
@app.get("/series-list")
def get_series_list():
    """Returns a list of all available series names with metadata"""
    if not ALL_SERIES:
        return []

    result = []
    for series_name, item in ALL_SERIES.items():
        # Extract metadata from the item
        result.append({
            "series_name": series_name,
            "entity_name": item.get("entity_name", ""),
            "insight": item.get("insight", ""),
            "category": item.get("entity_representation", ""),
            "data_points": len(item.get("series", []))
        })

    # Sort by series_name
    result.sort(key=lambda x: x["series_name"])
    return result


# Framework Comparison Widget
@register_widget({
    "name": "Framework Comparison",
    "description": "Compare multiple framework series side by side",
    "type": "table",
    "endpoint": "framework-comparison",
    "gridData": {"w": 40, "h": 15},
    "params": [
        {
            "paramName": "series_name",
            "description": "Select series to compare",
            "label": "Series",
            "type": "endpoint",
            "optionsEndpoint": "/series-options",
            "multiSelect": True,
            "style": {
                "popupWidth": 900
            },
            "show": True
        }
    ],
    "data": {
        "table": {
            "enableCharts": True,
            "showAll": True,
            "chartView": {
                "enabled": True,
                "chartType": "line"
            }
        }
    }
})
@app.get("/framework-comparison")
def get_framework_comparison(series_name: Optional[str] = Query(default=None, description="Comma-separated series names to compare")):
    """Compare multiple framework series side by side with date as the common index"""
    if not series_name:
        raise HTTPException(
            status_code=400,
            detail="Please select at least one series to compare"
        )

    # Parse comma-separated series names
    selected_series = [s.strip() for s in series_name.split(',')]

    # Collect data for each selected series
    combined_data = {}
    dates_set = set()

    for sname in selected_series:
        if sname in ALL_SERIES:
            item = ALL_SERIES[sname]
            series_data = item.get('series', [])

            if isinstance(series_data, str):
                try:
                    series_data = json.loads(series_data)
                except:
                    continue

            if isinstance(series_data, list):
                for point in series_data:
                    date = point.get('date')
                    value = point.get('value')
                    if date:
                        dates_set.add(date)
                        if date not in combined_data:
                            combined_data[date] = {}
                        combined_data[date][sname] = value

    # Convert to list format
    if combined_data:
        result = []
        for date in sorted(dates_set):
            row = {'date': date}
            row.update(combined_data.get(date, {}))
            result.append(row)
        return result

    return []


# Endpoint to get series options for the Framework Comparison widget
@app.get("/series-options")
def get_series_options():
    """Returns all available series names as options for the multi-select"""
    options = [{"value": name, "label": name} for name in sorted(ALL_SERIES.keys())]
    return options


# Update the Framework Comparison widget params on startup
def update_framework_comparison_params():
    """Update the Framework Comparison widget with available series options"""
    if "framework-comparison" in WIDGETS:
        options = [{"value": name, "label": name} for name in sorted(ALL_SERIES.keys())]
        # Update the series_name param options (flat array structure)
        params = WIDGETS["framework-comparison"].get("params", [])
        for param in params:
            if isinstance(param, dict) and param.get("paramName") == "series_name":
                param["options"] = options
                # Set default to first option if available
                if options:
                    param["value"] = options[0]["value"]
                break


# ==============================================================================
# GROUP FRAMEWORK APP - CRUD Endpoints
# ==============================================================================

# GET all groups
@app.get("/groups")
def get_groups():
    """Returns all framework groups"""
    return list(GROUPS.values())


# POST create new group
@app.post("/groups")
def create_group(
    name: str = Query(..., description="Group name"),
    description: str = Query("", description="Group description"),
    frameworks: str = Query("", description="Comma-separated framework series names")
):
    """Create a new framework group"""
    group_id = str(uuid.uuid4())[:8]

    # Parse frameworks from comma-separated string
    framework_list = [f.strip() for f in frameworks.split(',') if f.strip()] if frameworks else []

    # Validate frameworks exist
    valid_frameworks = [f for f in framework_list if f in ALL_SERIES]

    GROUPS[group_id] = {
        "group_id": group_id,
        "name": name,
        "description": description,
        "frameworks": valid_frameworks,
        "created_at": datetime.now().isoformat()
    }

    save_groups()
    return GROUPS[group_id]


# PUT update existing group
@app.put("/groups/{group_id}")
def update_group(
    group_id: str,
    name: Optional[str] = Query(None, description="New group name"),
    description: Optional[str] = Query(None, description="New group description"),
    frameworks: Optional[str] = Query(None, description="Comma-separated framework series names")
):
    """Update an existing framework group"""
    if group_id not in GROUPS:
        raise HTTPException(status_code=404, detail=f"Group {group_id} not found")

    if name is not None:
        GROUPS[group_id]["name"] = name
    if description is not None:
        GROUPS[group_id]["description"] = description
    if frameworks is not None:
        framework_list = [f.strip() for f in frameworks.split(',') if f.strip()]
        valid_frameworks = [f for f in framework_list if f in ALL_SERIES]
        GROUPS[group_id]["frameworks"] = valid_frameworks

    GROUPS[group_id]["updated_at"] = datetime.now().isoformat()
    save_groups()
    return GROUPS[group_id]


# DELETE group
@app.delete("/groups/{group_id}")
def delete_group(group_id: str):
    """Delete a framework group"""
    if group_id not in GROUPS:
        raise HTTPException(status_code=404, detail=f"Group {group_id} not found")

    deleted = GROUPS.pop(group_id)
    save_groups()
    return {"message": f"Group '{deleted['name']}' deleted successfully"}


# ==============================================================================
# GROUP FRAMEWORK APP - Dropdown Options Endpoints
# ==============================================================================

# Get group options for dropdown (by ID)
@app.get("/group-options")
def get_group_options():
    """Returns all groups as options for dropdown (value=id, label=name)"""
    options = [
        {"value": g["group_id"], "label": g["name"]}
        for g in GROUPS.values()
    ]
    return sorted(options, key=lambda x: x["label"])


# Get group options for dropdown (by name)
@app.get("/group-name-options")
def get_group_name_options():
    """Returns all groups as options for dropdown (value=name, label=name)"""
    options = [
        {"value": g["name"], "label": g["name"]}
        for g in GROUPS.values()
    ]
    return sorted(options, key=lambda x: x["label"])


# Get frameworks filtered by group (by ID)
@app.get("/group-frameworks")
def get_group_frameworks(
    group_id: Optional[str] = Query(None, description="Group ID to filter frameworks")
):
    """Returns frameworks within a specific group as dropdown options"""
    if not group_id or group_id not in GROUPS:
        # Return all frameworks if no group specified
        return [{"value": name, "label": name} for name in sorted(ALL_SERIES.keys())]

    group = GROUPS[group_id]
    return [
        {"value": name, "label": name}
        for name in sorted(group.get("frameworks", []))
    ]


# Get frameworks filtered by group (by name)
@app.get("/group-frameworks-by-name")
def get_group_frameworks_by_name(
    group_name: Optional[str] = Query(None, description="Group name to filter frameworks")
):
    """Returns frameworks within a specific group (by name) as dropdown options"""
    if not group_name:
        # Return all frameworks if no group specified
        return [{"value": name, "label": name} for name in sorted(ALL_SERIES.keys())]

    # Find group by name
    group = next((g for g in GROUPS.values() if g["name"] == group_name), None)
    if not group:
        return [{"value": name, "label": name} for name in sorted(ALL_SERIES.keys())]

    return [
        {"value": name, "label": name}
        for name in sorted(group.get("frameworks", []))
    ]


# ==============================================================================
# GROUP FRAMEWORK APP - Grouped Framework Comparison Widget
# ==============================================================================

@register_widget({
    "name": "Grouped Framework Comparison",
    "description": "Compare frameworks within a selected group",
    "type": "table",
    "endpoint": "grouped-framework-comparison",
    "gridData": {"w": 40, "h": 15},
    "params": [
        {
            "paramName": "group_name",
            "description": "Select a group to filter frameworks",
            "label": "Group",
            "type": "endpoint",
            "optionsEndpoint": "group-name-options",
            "show": True
        },
        {
            "paramName": "series_name",
            "description": "Select frameworks to compare",
            "label": "Frameworks",
            "type": "endpoint",
            "optionsEndpoint": "group-frameworks-by-name",
            "optionsParams": {"group_name": "$group_name"},
            "multiSelect": True,
            "style": {"popupWidth": 900},
            "show": True
        }
    ],
    "data": {
        "table": {
            "enableCharts": True,
            "showAll": True,
            "chartView": {
                "enabled": True,
                "chartType": "line"
            }
        }
    }
})
@app.get("/grouped-framework-comparison")
def get_grouped_framework_comparison(
    group_name: Optional[str] = Query(None, description="Group name"),
    series_name: Optional[str] = Query(None, description="Comma-separated series names")
):
    """Compare frameworks within a group, with date as common index"""

    # Find group by name if provided
    group = None
    if group_name:
        group = next((g for g in GROUPS.values() if g["name"] == group_name), None)

    # Determine which series to use
    if series_name:
        # Parse comma-separated series names
        selected_series = [s.strip() for s in series_name.split(',')]
        # If group specified, validate frameworks are in group
        if group:
            group_frameworks = set(group.get("frameworks", []))
            selected_series = [s for s in selected_series if s in group_frameworks]
    elif group:
        # No series specified but group is - automatically use ALL frameworks from group
        selected_series = group.get("frameworks", [])
    else:
        raise HTTPException(
            status_code=400,
            detail="Please select a group or at least one framework to compare"
        )

    if not selected_series:
        raise HTTPException(
            status_code=400,
            detail="No frameworks found. Please select a group with frameworks or select frameworks manually."
        )

    # Collect data for each selected series
    combined_data = {}
    dates_set = set()

    for sname in selected_series:
        if sname in ALL_SERIES:
            item = ALL_SERIES[sname]
            series_data = item.get('series', [])

            if isinstance(series_data, str):
                try:
                    series_data = json.loads(series_data)
                except:
                    continue

            if isinstance(series_data, list):
                for point in series_data:
                    date = point.get('date')
                    value = point.get('value')
                    if date:
                        dates_set.add(date)
                        if date not in combined_data:
                            combined_data[date] = {}
                        combined_data[date][sname] = value

    # Convert to list format
    if combined_data:
        result = []
        for date in sorted(dates_set):
            row = {'date': date}
            row.update(combined_data.get(date, {}))
            result.append(row)
        return result

    return []


# ==============================================================================
# GROUP FRAMEWORK APP - Group Manager HTML Widget
# ==============================================================================

# Load the HTML widget content
GROUP_MANAGER_HTML_FILE = Path(__file__).parent / "group_manager_widget.html"


@register_widget({
    "name": "Group Manager",
    "description": "Create, edit, and manage framework groups",
    "type": "html",
    "endpoint": "group-manager-html",
    "gridData": {"w": 40, "h": 16}
})
@app.get("/group-manager-html")
def get_group_manager_html():
    """Returns the HTML widget for group management"""
    from fastapi.responses import HTMLResponse

    if GROUP_MANAGER_HTML_FILE.exists():
        with open(GROUP_MANAGER_HTML_FILE, 'r') as f:
            html_content = f.read()

        # Inject the backend URL directly into the HTML
        # This ensures the widget knows where to make API calls
        backend_url = "http://127.0.0.1:8037"

        # Replace the getApiBase function with a simple constant
        html_content = html_content.replace(
            "const API_BASE = getApiBase();",
            f'const API_BASE = "{backend_url}"; // Injected by backend'
        )

        return HTMLResponse(content=html_content)
    else:
        return HTMLResponse(content="<html><body><p>Widget not found</p></body></html>")


# Legacy endpoints kept for backwards compatibility with the HTML widget
@app.post("/manage-group")
def manage_group(
    group_id: str = Query(..., description="Group ID"),
    action: str = Query(..., description="Action: edit or delete"),
    new_name: Optional[str] = Query(None, description="New name for edit"),
    new_frameworks: Optional[str] = Query(None, description="New frameworks for edit")
):
    """Handle group edit or delete actions (legacy endpoint)"""
    if group_id not in GROUPS:
        raise HTTPException(status_code=404, detail=f"Group {group_id} not found")

    if action == "delete":
        deleted = GROUPS.pop(group_id)
        save_groups()
        return {"message": f"Group '{deleted['name']}' deleted successfully"}

    elif action == "edit":
        if new_name:
            GROUPS[group_id]["name"] = new_name
        if new_frameworks:
            framework_list = [f.strip() for f in new_frameworks.split(',') if f.strip()]
            valid_frameworks = [f for f in framework_list if f in ALL_SERIES]
            GROUPS[group_id]["frameworks"] = valid_frameworks

        GROUPS[group_id]["updated_at"] = datetime.now().isoformat()
        save_groups()
        return {"message": f"Group '{GROUPS[group_id]['name']}' updated successfully"}

    else:
        raise HTTPException(status_code=400, detail=f"Invalid action: {action}")


# ==============================================================================
# GROUP FRAMEWORK APP - Groups List Widget
# ==============================================================================

@register_widget({
    "name": "Groups List",
    "description": "View all created groups - click a group to update comparison",
    "type": "table",
    "endpoint": "groups-list",
    "gridData": {"w": 40, "h": 10},
    "data": {
        "table": {
            "enableCharts": False,
            "showAll": True,
            "columnsDefs": [
                {
                    "field": "name",
                    "headerName": "Select",
                    "cellDataType": "text",
                    "pinned": "left",
                    "renderFn": "cellOnClick",
                    "renderFnParams": {
                        "actionType": "groupBy",
                        "groupByParamName": "group_name"
                    }
                },
                {
                    "field": "description",
                    "headerName": "Description",
                    "cellDataType": "text"
                },
                {
                    "field": "framework_count",
                    "headerName": "Frameworks",
                    "cellDataType": "number"
                },
                {
                    "field": "created_at",
                    "headerName": "Created At",
                    "cellDataType": "text"
                }
            ]
        }
    },
    "params": [
        {
            "paramName": "group_name",
            "description": "Selected group",
            "label": "Group",
            "type": "endpoint",
            "optionsEndpoint": "group-name-options",
            "show": False
        }
    ]
})
@app.get("/groups-list")
def get_groups_list(
    group_name: Optional[str] = Query(None, description="Selected group name (for groupBy sync)")
):
    """Returns all groups as a table"""
    if not GROUPS:
        return []

    result = []
    for group in GROUPS.values():
        result.append({
            "name": group["name"],
            "description": group.get("description", ""),
            "framework_count": len(group.get("frameworks", [])),
            "created_at": group.get("created_at", "")
        })

    return sorted(result, key=lambda x: x["name"])


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
    uvicorn.run(app, host="0.0.0.0", port=8037)