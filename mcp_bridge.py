"""
Bridge: registra tools do LangChain como tools MCP (FastMCP).
"""

import asyncio
import inspect
from typing import Any, Dict, List, Optional, Type

# Import genérico para compatibilidade
from pydantic import BaseModel
from langchain.tools.base import BaseTool


def _tool_metadata(lc_tool: BaseTool):
    """Extrai nome, descrição e schema (quando houver) da tool do LangChain."""
    name = getattr(lc_tool, "name", lc_tool.__class__.__name__)
    description = getattr(lc_tool, "description", lc_tool.__doc__ or "").strip()

    args_schema: Optional[Type[BaseModel]] = getattr(lc_tool, "args_schema", None)
    return name, description, args_schema


def _kwargs_from_args_schema(args_schema: Type[BaseModel]) -> Dict[str, Any]:
    """
    Retorna um dicionário {param: default_ou_Required} apenas para documentação;
    O FastMCP infere JSON Schema a partir de anotações de tipos da função wrapper,
    então vamos criar *hints* coerentes no wrapper dinâmico.
    """
    hints = {}
    for field_name, field in args_schema.model_fields.items():
        # Tenta obter tipo e default
        ann = field.annotation or Any
        default = field.default if field.default is not None else ...
        hints[field_name] = (ann, default)
    return hints


def register_langchain_tools_as_mcp(mcp, tools: List[BaseTool]):
    """
    Para cada LangChain Tool, cria e registra uma tool MCP equivalente.
    """
    for lc_tool in tools:
        try:
            name, description, args_schema = _tool_metadata(lc_tool)
            print(f"📝 Registrando tool: {name}")
        except Exception as e:
            print(f"❌ Erro ao extrair metadados da tool {lc_tool}: {e}")
            continue

        # Cria um wrapper assíncrono chamando lc_tool.invoke(...)
        if args_schema is not None:
            # StructuredTool: registrar assinatura com kwargs tipados
            hints = _kwargs_from_args_schema(args_schema)

            params = []
            annotations = {}
            for pname, (ann, default) in hints.items():
                param = inspect.Parameter(
                    pname,
                    inspect.Parameter.KEYWORD_ONLY,
                    default=default,
                    annotation=ann,
                )
                params.append(param)
                annotations[pname] = ann
            annotations["return"] = str
            sig = inspect.Signature(params)

            async def wrapper(**kwargs):
                # Executa tool síncrona em thread pool para não bloquear event loop
                def _run():
                    return lc_tool.invoke(kwargs)

                result = await asyncio.to_thread(_run)
                return result if isinstance(result, str) else str(result)

            # Injeta metadados para o FastMCP
            wrapper.__name__ = name
            wrapper.__doc__ = description
            wrapper.__signature__ = sig
            wrapper.__annotations__ = annotations

            try:
                mcp.tool()(wrapper)  # registra com nome=__name__ e docstring
                print(f"✅ Tool registrada com sucesso: {name}")
            except Exception as e:
                print(f"❌ Erro ao registrar tool estruturada {name}: {e}")
        else:
            async def simple_wrapper(input: str) -> str:
                def _run():
                    try:
                        return lc_tool.invoke(input)
                    except TypeError:
                        return lc_tool.invoke({"input": input})

                result = await asyncio.to_thread(_run)
                return result if isinstance(result, str) else str(result)

            simple_wrapper.__name__ = name
            simple_wrapper.__doc__ = description

            try:
                mcp.tool()(simple_wrapper)
                print(f"✅ Tool simples registrada com sucesso: {name}")
            except Exception as e:
                print(f"❌ Erro ao registrar tool simples {name}: {e}")


__all__ = ["register_langchain_tools_as_mcp"]
