#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# scripts/resume.py — Research state parser and pretty printer

import os
import sys
import re

# ANSI colors
BOLD = "\033[1m"
GREEN = "\033[38;5;82m"
BLUE = "\033[38;5;39m"
YELLOW = "\033[38;5;214m"
RED = "\033[38;5;196m"
CYAN = "\033[38;5;51m"
RESET = "\033[0m"

STATE_FILE = "research/state.md"

if not os.path.exists(STATE_FILE):
    print(f"{RED}Error: {STATE_FILE} no existe.{RESET}")
    sys.exit(1)

with open(STATE_FILE, "r", encoding="utf-8") as f:
    text = f.read()

print(f"{BOLD}{BLUE}================================================================{RESET}")
print(f"{BOLD}{GREEN}               RESEARCH AUTOMATION SYSTEM (Research OS)        {RESET}")
print(f"{BOLD}{BLUE}================================================================{RESET}\n")

# Split sections by ## headers
sections = re.split(r'\n##\s+', text)

for sec in sections:
    if not sec.strip():
        continue
    
    lines = sec.strip().split('\n')
    title = lines[0]
    body = '\n'.join(lines[1:])
    
    # Determine section color and icon based on title
    if "Hito Actual" in title:
        color = CYAN
        title_display = "📌 HITO ACTIVO ACTUAL"
    elif "Caminos Validados" in title:
        color = GREEN
        title_display = "✅ CAMINOS VALIDADOS (KNOWN GOOD)"
    elif "Caminos Rechazados" in title:
        color = RED
        title_display = "❌ CAMINOS RECHAZADOS (KNOWN BAD / EVITAR!)"
    elif "Hipótesis" in title:
        color = YELLOW
        title_display = "🔬 HIPÓTESIS ABIERTAS"
    elif "Lista de Tareas" in title:
        color = BLUE
        title_display = "📋 TAREAS PENDIENTES (TODO)"
    else:
        continue # Skip other sections like the main title
        
    print(f"{BOLD}{color}{title_display}:{RESET}")
    
    # Format the body
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        
        # Skip horizontal rules
        if line.startswith("---") or line.strip() == "---":
            continue
        
        # Subheaders
        if line.startswith("###"):
            sub = line.replace("###", "").strip()
            print(f"  {BOLD}{sub}{RESET}")
        # Bullet points
        elif line.startswith("-") or line.startswith("*"):
            bullet = re.sub(r'^[-*]\s*', '', line)
            print(f"    • {bullet}")
        # Numbered lists
        elif re.match(r'^\d+\.', line):
            num = re.sub(r'^\d+\.\s*', '', line)
            print(f"    • {num}")
        # Plain paragraphs (except intro lines)
        elif not line.startswith("Las siguientes") and not line.startswith("¡NO volver"):
            print(f"    {line}")
    print()

print(f"{BOLD}{BLUE}----------------------------------------------------------------{RESET}")
print(f"{BOLD}Instrucciones para la nueva sesión de IA:{RESET}")
print("  1. Identifica tu rol actual: Architect, Implementer, Validator o Reviewer.")
print("  2. Lee el prompt correspondiente en el directorio prompts/.")
print(f"  3. No repitas investigaciones ni implementaciones listadas en {RED}KNOWN BAD{RESET}.")
print(f"{BOLD}{BLUE}================================================================{RESET}")
