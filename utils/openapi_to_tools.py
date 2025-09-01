import json
from typing import Dict, List, Any, Union, Optional
from pathlib import Path
try:
    import requests
except ImportError:
    requests = None
try:
    import httpx
except ImportError:
    httpx = None


class OpenAPIToLangGraphTools:
    """Utility to convert OpenAPI/Swagger JSON into LangGraph tools"""
    
    def __init__(self, base_url: Optional[str] = None):
        self.base_url = base_url
        self.openapi_spec = None
        
    def load_openapi(self, source: Union[str, Path]) -> Dict[str, Any]:
        """Load OpenAPI spec from URL or file path"""
        if isinstance(source, (str, Path)):
            if str(source).startswith(('http://', 'https://')):
                if requests is None:
                    raise ImportError("requests library is required for loading from URL. Install with: pip install requests")
                response = requests.get(str(source))
                response.raise_for_status()
                self.openapi_spec = response.json()
            else:
                with open(source, 'r', encoding='utf-8') as f:
                    self.openapi_spec = json.load(f)
        else:
            self.openapi_spec = source
            
        return self.openapi_spec
    
    def _get_parameter_schema(self, parameters: List[Dict]) -> Dict[str, Any]:
        """Convert OpenAPI parameters to Pydantic schema"""
        properties = {}
        required = []
        
        for param in parameters:
            name = param['name']
            param_schema = param.get('schema', {})
            
            properties[name] = {
                'type': param_schema.get('type', 'string'),
                'description': param.get('description', f'{name} parameter')
            }
            
            if param.get('required', False):
                required.append(name)
        
        return {
            'type': 'object',
            'properties': properties,
            'required': required
        }
    
    def _get_request_body_schema(self, request_body: Dict) -> Dict[str, Any]:
        """Convert OpenAPI request body to schema"""
        if not request_body:
            return {}
            
        content = request_body.get('content', {})
        json_content = content.get('application/json', {})
        schema = json_content.get('schema', {})
        
        if '$ref' in schema:
            ref_path = schema['$ref'].split('/')
            if ref_path[0] == '#' and len(ref_path) > 1:
                components = self.openapi_spec.get('components', {})
                schemas = components.get('schemas', {})
                schema_name = ref_path[-1]
                return schemas.get(schema_name, {})
        
        return schema
    
    def _create_tool_function(self, path: str, method: str, operation: Dict[str, Any]) -> callable:
        """Create a function that can be used as a LangGraph tool"""
        operation_id = operation.get('operationId', f"{method}_{path.replace('/', '_')}")
        summary = operation.get('summary', f'{method.upper()} {path}')
        
        def tool_function(**kwargs) -> str:
            """Generated tool function for API endpoint"""
            url = f"{self.base_url}{path}" if self.base_url else path
            
            # Replace path parameters
            for key, value in kwargs.items():
                if f"{{{key}}}" in url:
                    url = url.replace(f"{{{key}}}", str(value))
            
            # Prepare request
            params = {}
            data = None
            
            # Handle parameters
            parameters = operation.get('parameters', [])
            path_params = {p['name'] for p in parameters if p.get('in') == 'path'}
            query_params = {p['name'] for p in parameters if p.get('in') == 'query'}
            
            for key, value in kwargs.items():
                if key not in path_params:
                    if key in query_params:
                        params[key] = value
                    else:
                        # Assume it's request body data
                        if data is None:
                            data = {}
                        data[key] = value
            
            # Make request
            try:
                if httpx is not None:
                    response = httpx.request(
                        method=method.upper(),
                        url=url,
                        params=params,
                        json=data if data else None
                    )
                    response.raise_for_status()
                    
                    try:
                        return json.dumps(response.json(), ensure_ascii=False, indent=2)
                    except:
                        return response.text
                elif requests is not None:
                    response = requests.request(
                        method=method.upper(),
                        url=url,
                        params=params,
                        json=data if data else None
                    )
                    response.raise_for_status()
                    
                    try:
                        return json.dumps(response.json(), ensure_ascii=False, indent=2)
                    except:
                        return response.text
                else:
                    return f"No HTTP client available. Install httpx or requests: pip install httpx requests"
                    
            except Exception as e:
                return f"Error calling {method.upper()} {path}: {str(e)}"
        
        tool_function.__name__ = operation_id
        tool_function.__doc__ = summary
        
        return tool_function
    
    def _create_tool_schema(self, path: str, method: str, operation: Dict[str, Any]) -> Dict[str, Any]:
        """Create tool schema for LangGraph"""
        operation_id = operation.get('operationId', f"{method}_{path.replace('/', '_')}")
        summary = operation.get('summary', f'{method.upper()} {path}')
        description = operation.get('description', summary)
        
        # Combine parameters and request body into single schema
        properties = {}
        required = []
        
        # Handle path/query parameters
        parameters = operation.get('parameters', [])
        for param in parameters:
            name = param['name']
            param_schema = param.get('schema', {})
            
            properties[name] = {
                'type': param_schema.get('type', 'string'),
                'description': param.get('description', f'{name} parameter')
            }
            
            if param.get('required', False):
                required.append(name)
        
        # Handle request body
        request_body = operation.get('requestBody', {})
        if request_body:
            body_schema = self._get_request_body_schema(request_body)
            if body_schema.get('properties'):
                properties.update(body_schema['properties'])
                if body_schema.get('required'):
                    required.extend(body_schema['required'])
        
        return {
            'name': operation_id,
            'description': description,
            'parameters': {
                'type': 'object',
                'properties': properties,
                'required': required
            }
        }
    
    def generate_tools(self) -> List[Dict[str, Any]]:
        """Generate LangGraph tools from OpenAPI spec"""
        if not self.openapi_spec:
            raise ValueError("OpenAPI spec not loaded. Call load_openapi() first.")
        
        tools = []
        paths = self.openapi_spec.get('paths', {})
        
        for path, methods in paths.items():
            for method, operation in methods.items():
                if method.lower() in ['get', 'post', 'put', 'delete', 'patch']:
                    tool_schema = self._create_tool_schema(path, method, operation)
                    tool_function = self._create_tool_function(path, method, operation)
                    
                    tools.append({
                        'schema': tool_schema,
                        'function': tool_function
                    })
        
        return tools
    
    def generate_tools_dict(self) -> Dict[str, Dict[str, Any]]:
        """Generate tools as a dictionary with operation_id as key"""
        tools = self.generate_tools()
        return {tool['schema']['name']: tool for tool in tools}
    
    def _generate_python_function(self, path: str, method: str, operation: Dict[str, Any]) -> str:
        """Generate Python function code with @tool decorator"""
        operation_id = operation.get('operationId', f"{method}_{path.replace('/', '_').replace('-', '_')}")
        summary = operation.get('summary', f'{method.upper()} {path}')
        description = operation.get('description', summary)
        
        # Get all parameters
        parameters = operation.get('parameters', [])
        request_body = operation.get('requestBody', {})
        
        # Build function signature
        func_params = []
        doc_params = []
        required_params = []
        
        # Path parameters
        path_params = [p for p in parameters if p.get('in') == 'path']
        for param in path_params:
            name = param['name']
            param_type = self._get_python_type(param.get('schema', {}))
            func_params.append(f"{name}: {param_type}")
            doc_params.append(f"        {name}: {param.get('description', f'{name} parameter')}")
            if param.get('required', False):
                required_params.append(name)
        
        # Query parameters
        query_params = [p for p in parameters if p.get('in') == 'query']
        for param in query_params:
            name = param['name']
            param_type = self._get_python_type(param.get('schema', {}))
            default = "None" if not param.get('required', False) else ""
            if default:
                func_params.append(f"{name}: Optional[{param_type}] = {default}")
            else:
                func_params.append(f"{name}: {param_type}")
            doc_params.append(f"        {name}: {param.get('description', f'{name} parameter')}")
            if param.get('required', False):
                required_params.append(name)
        
        # Request body parameters
        if request_body:
            body_schema = self._get_request_body_schema(request_body)
            if body_schema.get('properties'):
                for prop_name, prop_schema in body_schema['properties'].items():
                    param_type = self._get_python_type(prop_schema)
                    is_required = prop_name in body_schema.get('required', [])
                    if not is_required:
                        func_params.append(f"{prop_name}: Optional[{param_type}] = None")
                    else:
                        func_params.append(f"{prop_name}: {param_type}")
                    doc_params.append(f"        {prop_name}: {prop_schema.get('description', f'{prop_name} parameter')}")
                    if is_required:
                        required_params.append(prop_name)
        
        # Generate function code
        func_signature = f"def {operation_id}({', '.join(func_params)}) -> Dict[str, Any]:"
        
        # Generate docstring
        docstring_lines = [f'    """', f'    {description}']
        if doc_params:
            docstring_lines.extend(['', '    Args:'] + doc_params)
        docstring_lines.extend(['', '    Returns:', '        Dict contendo a resposta da API ou mensagem de erro', '    """'])
        
        # Generate function body
        url_line = f'    url = f"{{BASE_URL}}{path}"'
        
        # Replace path parameters in URL
        url_replacements = []
        for param in path_params:
            name = param['name']
            url_replacements.append(f'    url = url.replace("{{{{{name}}}}}", str({name}))')
        
        # Prepare request parameters
        request_preparation = [
            '    try:',
            '        params = {}',
            '        json_data = {}'
        ]
        
        # Add query parameters
        if query_params:
            for param in query_params:
                name = param['name']
                if param.get('required', False):
                    request_preparation.append(f'        params["{name}"] = {name}')
                else:
                    request_preparation.append(f'        if {name} is not None:')
                    request_preparation.append(f'            params["{name}"] = {name}')
        
        # Add request body
        if request_body:
            body_schema = self._get_request_body_schema(request_body)
            if body_schema.get('properties'):
                for prop_name in body_schema['properties'].keys():
                    is_required = prop_name in body_schema.get('required', [])
                    if is_required:
                        request_preparation.append(f'        json_data["{prop_name}"] = {prop_name}')
                    else:
                        request_preparation.append(f'        if {prop_name} is not None:')
                        request_preparation.append(f'            json_data["{prop_name}"] = {prop_name}')
        
        # HTTP method call
        method_upper = method.upper()
        if method_upper in ['GET', 'DELETE']:
            http_call = f'        response = requests.{method.lower()}(url, params=params)'
        else:
            http_call = f'        response = requests.{method.lower()}(url, params=params, json=json_data if json_data else None)'
        
        # Complete function
        function_lines = [
            '@tool',
            func_signature
        ] + docstring_lines + [
            url_line
        ] + url_replacements + request_preparation + [
            '',
            http_call,
            '        response.raise_for_status()',
            '        return {"status": "success", "data": response.json()}',
            '    except requests.exceptions.RequestException as e:',
            '        return {"status": "error", "message": str(e)}'
        ]
        
        return '\n'.join(function_lines)
    
    def _get_python_type(self, schema: Dict[str, Any]) -> str:
        """Convert OpenAPI schema type to Python type"""
        schema_type = schema.get('type', 'string')
        
        type_mapping = {
            'string': 'str',
            'integer': 'int',
            'number': 'float',
            'boolean': 'bool',
            'array': 'List[Any]',
            'object': 'Dict[str, Any]'
        }
        
        return type_mapping.get(schema_type, 'Any')
    
    def generate_python_file(self, output_path: str = "lang_tools.py", base_url: str = None) -> str:
        """Generate a complete Python file with @tool decorated functions"""
        if not self.openapi_spec:
            raise ValueError("OpenAPI spec not loaded. Call load_openapi() first.")
        
        base_url = base_url or self.base_url or "http://localhost:8000"
        
        # File header
        header = f'''#!/usr/bin/env python3
"""
Tools geradas automaticamente do OpenAPI para uso com LangGraph
Gerado automaticamente a partir do OpenAPI spec
"""

from typing import Dict, Any, Optional, List
import requests
from langchain_core.tools import tool


# Configuração da URL base da API
BASE_URL = "{base_url}"

'''
        
        # Generate functions
        functions = []
        function_names = []
        
        paths = self.openapi_spec.get('paths', {})
        for path, methods in paths.items():
            for method, operation in methods.items():
                if method.lower() in ['get', 'post', 'put', 'delete', 'patch']:
                    func_code = self._generate_python_function(path, method, operation)
                    functions.append(func_code)
                    operation_id = operation.get('operationId', f"{method}_{path.replace('/', '_').replace('-', '_')}")
                    function_names.append(operation_id)
        
        # Footer with tools list
        footer = f'''

# Lista de todas as tools disponíveis
AVAILABLE_TOOLS = [
    {',\n    '.join(function_names)}
]


def get_tools() -> List:
    """
    Retorna a lista de tools prontas para uso no LangGraph.
    
    Usage:
        from {output_path.replace('.py', '')} import get_tools
        tools = get_tools()
        
        # Para usar com LangGraph:
        # from langgraph.prebuilt import ToolExecutor  
        # tool_executor = ToolExecutor(tools)
    """
    return AVAILABLE_TOOLS


if __name__ == "__main__":
    print("=== Tools Disponíveis ===")
    for i, tool in enumerate(AVAILABLE_TOOLS, 1):
        print(f"{{i}}. {{tool.name}}: {{tool.description}}")
'''
        
        # Combine all parts
        complete_file = header + '\n\n'.join(functions) + footer
        
        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(complete_file)
            
        return complete_file


def convert_openapi_to_tools(source: Union[str, Path], base_url: Optional[str] = None) -> List[Dict[str, Any]]:
    """Convenience function to convert OpenAPI spec to LangGraph tools
    
    Args:
        source: URL or file path to OpenAPI JSON
        base_url: Base URL for API calls
        
    Returns:
        List of tool dictionaries with 'schema' and 'function' keys
    """
    converter = OpenAPIToLangGraphTools(base_url=base_url)
    converter.load_openapi(source)
    return converter.generate_tools()


def generate_langraph_tools_file(openapi_source: Union[str, Path], 
                                output_file: str = "lang_tools.py", 
                                base_url: str = "http://localhost:8000") -> str:
    """Função principal para gerar arquivo lang_tools.py a partir do OpenAPI
    
    Args:
        openapi_source: Caminho para o arquivo openapi.json ou URL
        output_file: Nome do arquivo de saída (padrão: lang_tools.py)
        base_url: URL base da API
        
    Returns:
        String com o código gerado
        
    Example:
        generate_langraph_tools_file("openapi.json", "minha_api_tools.py", "http://localhost:3000")
    """
    converter = OpenAPIToLangGraphTools(base_url=base_url)
    converter.load_openapi(openapi_source)
    return converter.generate_python_file(output_file, base_url)


if __name__ == "__main__":
    # Example usage
    converter = OpenAPIToLangGraphTools(base_url="http://localhost:8000")
    converter.load_openapi("openapi.json")
    tools = converter.generate_tools()
    
    print(f"Generated {len(tools)} tools:")
    for tool in tools:
        print(f"- {tool['schema']['name']}: {tool['schema']['description']}")