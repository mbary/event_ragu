


def pydantic_to_openai_tool(pydantic_schema: dict) -> dict:
    """
    Convert a Pydantic model's JSON schema to OpenAI's tool format.
    
    Args:
        pydantic_schema: The output of model.model_json_schema()
    
    Returns:
        dict: OpenAI tool format specification
    """
    # Extract basic info
    tool_name = pydantic_schema.get('title', 'UnnamedTool')
    description = pydantic_schema.get('description', 'No description provided')
    
    # Build parameters object
    parameters = {
        "type": "object",
        "properties": {},
        "required": []
    }
    
    # Copy properties, filtering out internal/output fields
    properties = pydantic_schema.get('properties', {})
    required_fields = pydantic_schema.get('required', [])
    
    for prop_name, prop_schema in properties.items():
        # Skip internal action_type field and output fields
        if prop_name in ['action_type', 'result']:
            continue
            
        # Copy the property schema
        parameters['properties'][prop_name] = {
            "type": prop_schema.get('type', 'string'),
            "description": prop_schema.get('description', '')
        }
        
        # Add examples if present
        if 'examples' in prop_schema:
            parameters['properties'][prop_name]['examples'] = prop_schema['examples']
        
        # Add enum values if it's a const/literal
        if 'const' in prop_schema:
            parameters['properties'][prop_name]['enum'] = [prop_schema['const']]
        
        # Handle array items
        if prop_schema.get('type') == 'array' and 'items' in prop_schema:
            parameters['properties'][prop_name]['items'] = prop_schema['items']
    
    # Update required fields (excluding filtered properties)
    parameters['required'] = [
        field for field in required_fields 
        if field in parameters['properties']
    ]
    
    # Include definitions if present (for nested models)
    if '$defs' in pydantic_schema:
        parameters['$defs'] = pydantic_schema['$defs']
    
    # Build the final tool specification
    tool_spec = {
        "type": "function",
        "function": {
            "name": tool_name,
            "description": description,
            "parameters": parameters
        }
    }
    
    return tool_spec


def pydantic_to_openai_tool2(pydantic_schema: dict) -> dict:
    """
    Convert a Pydantic model's JSON schema to OpenAI's tool format.
    Resolves all $ref references by inlining them.
    
    Args:
        pydantic_schema: The output of model.model_json_schema()
    
    Returns:
        dict: OpenAI tool format specification with resolved references
    """
    import copy
    
    # Create a deep copy to avoid modifying the original
    schema = copy.deepcopy(pydantic_schema)
    
    # Extract definitions for reference resolution
    definitions = schema.get('$defs', {})
    
    def resolve_refs(obj):
        """Recursively resolve all $ref references in the schema."""
        if isinstance(obj, dict):
            if '$ref' in obj:
                # Extract the reference name
                ref_path = obj['$ref']
                if ref_path.startswith('#/$defs/'):
                    ref_name = ref_path.split('/')[-1]
                    if ref_name in definitions:
                        # Replace the $ref with the actual definition
                        return resolve_refs(copy.deepcopy(definitions[ref_name]))
                return obj
            else:
                # Recursively process all values
                return {k: resolve_refs(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [resolve_refs(item) for item in obj]
        else:
            return obj
    
    # Resolve all references in the schema
    resolved_schema = resolve_refs(schema)
    
    # Extract basic info
    tool_name = resolved_schema.get('title', 'UnnamedTool')
    description = resolved_schema.get('description', 'No description provided')
    
    # Build parameters object
    parameters = {
        "type": "object",
        "properties": {},
        "required": []
    }
    
    # Copy properties, filtering out internal/output fields
    properties = resolved_schema.get('properties', {})
    required_fields = resolved_schema.get('required', [])
    
    for prop_name, prop_schema in properties.items():
        # Skip internal action_type field and output fields
        if prop_name in ['action_type', 'result']:
            continue
            
        # Copy the resolved property schema
        param_schema = {
            "type": prop_schema.get('type', 'string'),
            "description": prop_schema.get('description', '')
        }
        
        # Add examples if present
        if 'examples' in prop_schema:
            param_schema['examples'] = prop_schema['examples']
        
        # Add enum values if it's a const/literal
        if 'const' in prop_schema:
            param_schema['enum'] = [prop_schema['const']]
        
        # Handle arrays - the items should now be resolved
        if prop_schema.get('type') == 'array' and 'items' in prop_schema:
            param_schema['items'] = prop_schema['items']
        
        parameters['properties'][prop_name] = param_schema
    
    # Update required fields (excluding filtered properties)
    parameters['required'] = [
        field for field in required_fields 
        if field in parameters['properties']
    ]
    
    # Build the final tool specification
    tool_spec = {
        "type": "function",
        "function": {
            "name": tool_name,
            "description": description,
            "parameters": parameters
        }
    }
    
    return tool_spec