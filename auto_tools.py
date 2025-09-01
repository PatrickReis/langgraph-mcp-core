#!/usr/bin/env python3
"""
Script para automatizar completamente a geração e integração de tools
"""

import re
from utils.openapi_to_tools import generate_langraph_tools_file


def extract_tool_names_from_file(file_path: str):
    """Extrai nomes das tools de um arquivo Python gerado"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Busca por funções decoradas com @tool
    pattern = r'@tool\s+def\s+(\w+)\s*\('
    matches = re.findall(pattern, content)
    return matches


def update_main_tools_file(new_tools_file: str, new_tool_names: list):
    """Atualiza tools/tools.py com as novas tools"""
    
    # Lê o arquivo atual
    with open('tools/tools.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Nome do módulo (remove tools/ e .py)
    module_name = new_tools_file.replace('tools/', '').replace('.py', '')
    
    # Adiciona import das novas tools
    import_line = f"from .{module_name} import {', '.join(new_tool_names)}"
    
    # Encontra onde inserir o import (após os imports existentes)
    lines = content.split('\n')
    import_insert_index = 0
    
    for i, line in enumerate(lines):
        if line.startswith('from ') or line.startswith('import '):
            import_insert_index = i + 1
        elif line.strip() == '' or line.startswith('#'):
            continue
        else:
            break
    
    # Insere o novo import
    lines.insert(import_insert_index, import_line)
    
    # Atualiza a lista de tools no final
    for i, line in enumerate(lines):
        if line.strip().startswith('tools = ['):
            # Extrai tools atuais
            current_tools = []
            j = i
            while j < len(lines):
                if ']' in lines[j]:
                    # Extrai tools da linha atual
                    tools_text = lines[j].split('[')[1].split(']')[0] if '[' in lines[j] else lines[j].split(']')[0]
                    if tools_text.strip():
                        current_tools.extend([t.strip() for t in tools_text.split(',') if t.strip()])
                    break
                else:
                    # Linha intermediária
                    if '[' in lines[j]:
                        tools_text = lines[j].split('[')[1]
                    else:
                        tools_text = lines[j]
                    if tools_text.strip():
                        current_tools.extend([t.strip() for t in tools_text.split(',') if t.strip()])
                j += 1
            
            # Adiciona novas tools
            all_tools = list(dict.fromkeys(current_tools + new_tool_names))  # Remove duplicatas
            
            # Reconstroi a linha
            lines[i] = f"tools = [{', '.join(all_tools)}]"
            
            # Remove linhas antigas da lista tools se houver
            while j > i + 1:
                lines.pop(i + 1)
                j -= 1
            break
    
    # Escreve arquivo atualizado
    with open('tools/tools.py', 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def main():
    """Função principal que automatiza todo o processo"""
    
    print("🚀 Iniciando automação completa de tools...")
    
    # 1. Gera tools do OpenAPI
    openapi_file = "openapi/openapi.json"
    output_file = "tools/api_tools_auto.py"
    base_url = "http://localhost:8000"
    
    try:
        generate_langraph_tools_file(openapi_file, output_file, base_url)
        print(f"✅ Tools geradas: {output_file}")
    except Exception as e:
        print(f"❌ Erro ao gerar tools: {e}")
        return
    
    # 2. Extrai nomes das tools geradas
    try:
        new_tool_names = extract_tool_names_from_file(output_file)
        print(f"🔧 Tools encontradas: {', '.join(new_tool_names)}")
    except Exception as e:
        print(f"❌ Erro ao extrair nomes das tools: {e}")
        return
    
    # 3. Atualiza arquivo principal
    try:
        update_main_tools_file(output_file, new_tool_names)
        print(f"✅ Arquivo tools/tools.py atualizado!")
        print(f"📦 Novas tools integradas: {len(new_tool_names)}")
    except Exception as e:
        print(f"❌ Erro ao atualizar arquivo principal: {e}")
        return
        
    print("\n🎉 Processo completado!")
    print("💡 Próximos passos:")
    print("   1. Reiniciar o agente para carregar as novas tools")
    print("   2. Testar as funcionalidades da API")


if __name__ == "__main__":
    main()