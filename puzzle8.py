"""
=============================================================
  PROYECTO: Resolviendo Problemas Clásicos con Búsqueda
  Problema: 8-Puzzle (tablero 3x3)
  Algoritmos: BFS, DFS, Costo Uniforme (UCS), A*
  Materia: Inteligencia Artificial
  Integrantes: Pérez Moncayo Gonzalo Sebastian
               Sandoval Vargas Luis Antonio
=============================================================

DESCRIPCIÓN DEL PROBLEMA
--------------------------
El 8-Puzzle consiste en un tablero de 3x3 con 8 fichas numeradas
(1-8) y una celda vacía (representada con 0). El objetivo es
alcanzar el estado meta deslizando las fichas hacia la celda vacía.

Estado inicial (ejemplo):    Estado meta:
  2 8 3                        1 2 3
  1 6 4                        4 5 6
  7 0 5                        7 8 0

Representación formal:
  - Estado: tupla de 9 enteros (lectura fila por fila)
  - Operadores: mover la celda vacía ↑ ↓ ← →
  - Costo: cada movimiento tiene costo 1
  - Heurísticas disponibles: distancia Manhattan, fichas fuera de lugar
"""

import heapq
import time
from collections import deque


GOAL_STATE = (1, 2, 3,
              4, 5, 6,
              7, 8, 0)

MOVES = {
    'UP':    -3,
    'DOWN':  +3,
    'LEFT':  -1,
    'RIGHT': +1,
}


def is_solvable(state):
    """
    Verifica si un estado del 8-Puzzle es resoluble.
    Un estado es resoluble si el número de inversiones es par.
    Una inversión es un par (a, b) donde a aparece antes que b pero a > b
    (sin contar el 0).
    """
    tiles = [x for x in state if x != 0]
    inversions = sum(
        1 for i in range(len(tiles))
          for j in range(i + 1, len(tiles))
          if tiles[i] > tiles[j]
    )
    return inversions % 2 == 0


def get_neighbors(state):
    """Genera los estados vecinos moviendo la celda vacía."""
    neighbors = []
    zero_idx = state.index(0)
    row, col = divmod(zero_idx, 3)

    for direction, delta in MOVES.items():
        new_idx = zero_idx + delta

        if direction == 'LEFT'  and col == 0: continue
        if direction == 'RIGHT' and col == 2: continue
        if direction == 'UP'    and row == 0: continue
        if direction == 'DOWN'  and row == 2: continue

        new_state = list(state)
        new_state[zero_idx], new_state[new_idx] = new_state[new_idx], new_state[zero_idx]
        neighbors.append((tuple(new_state), direction, 1))

    return neighbors


def reconstruct_path(parent_map, state):
    """Reconstruye el camino desde el estado inicial hasta 'state'."""
    path = []
    while state in parent_map:
        state, action = parent_map[state]
        path.append(action)
    path.reverse()
    return path


def print_state(state, label=""):
    """Imprime el tablero de forma visual."""
    if label:
        print(f"\n  {label}")
    for i in range(0, 9, 3):
        row = state[i:i+3]
        print("  " + " ".join(str(x) if x != 0 else "·" for x in row))


def print_separator(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)

def heuristic_misplaced(state):
    """
    H1 – Fichas fuera de lugar:
    Cuenta cuántas fichas NO están en su posición meta.
    Es admisible (nunca sobreestima) porque cada ficha fuera
    de lugar necesita al menos un movimiento.
    """
    return sum(
        1 for i, val in enumerate(state)
        if val != 0 and val != GOAL_STATE[i]
    )


def heuristic_manhattan(state):
    """
    H2 – Distancia Manhattan:
    Suma de distancias (|Δfila| + |Δcol|) de cada ficha a su
    posición meta. Es admisible y más informativa que H1.
    """
    total = 0
    for i, val in enumerate(state):
        if val == 0:
            continue
        goal_idx = GOAL_STATE.index(val)
        cur_row,  cur_col  = divmod(i, 3)
        goal_row, goal_col = divmod(goal_idx, 3)
        total += abs(cur_row - goal_row) + abs(cur_col - goal_col)
    return total

def bfs(initial_state):
    """
    Búsqueda en Anchura (BFS)
    ──────────────────────────
    Explora nivel por nivel usando una cola FIFO.
    Garantiza encontrar la solución óptima (menor número de pasos).
    Puede consumir mucha memoria al almacenar todos los nodos de la frontera.
    """
    if initial_state == GOAL_STATE:
        return [], 0, 0

    frontier = deque()
    frontier.append(initial_state)
    visited = {initial_state}
    parent = {}         
    nodes_generated = 1

    while frontier:
        state = frontier.popleft()

        for neighbor, action, _ in get_neighbors(state):
            nodes_generated += 1
            if neighbor == GOAL_STATE:
                parent[neighbor] = (state, action)
                return reconstruct_path(parent, neighbor), nodes_generated, 0
            if neighbor not in visited:
                visited.add(neighbor)
                parent[neighbor] = (state, action)
                frontier.append(neighbor)

    return None, nodes_generated, 0


def dfs(initial_state, depth_limit=50):
    """
    Búsqueda en Profundidad (DFS) con límite de profundidad
    ────────────────────────────────────────────────────────
    Explora tan profundo como sea posible usando una pila LIFO.
    NO garantiza la solución óptima.
    Se añade un límite de profundidad para evitar ciclos infinitos.
    Usa poca memoria comparado con BFS.
    """
    if initial_state == GOAL_STATE:
        return [], 0, 0

    stack = [(initial_state, 0)]
    visited = set()
    parent = {}
    nodes_generated = 1

    while stack:
        state, depth = stack.pop()

        if state in visited:
            continue
        visited.add(state)

        if state == GOAL_STATE:
            return reconstruct_path(parent, state), nodes_generated, depth

        if depth >= depth_limit:
            continue

        for neighbor, action, _ in get_neighbors(state):
            nodes_generated += 1
            if neighbor not in visited:
                parent[neighbor] = (state, action)
                stack.append((neighbor, depth + 1))

    return None, nodes_generated, 0


def ucs(initial_state):
    """
    Búsqueda de Costo Uniforme (UCS)
    ──────────────────────────────────
    Explora por orden de costo acumulado usando una cola de prioridad.
    Garantiza la solución de menor costo.
    En el 8-Puzzle (costo uniforme = 1 por movimiento) es equivalente a BFS,
    pero su implementación permite costos variables.
    """
    if initial_state == GOAL_STATE:
        return [], 0, 0

    counter = 0
    frontier = [(0, counter, initial_state)]
    visited = {}
    parent  = {}
    nodes_generated = 1

    while frontier:
        cost, _, state = heapq.heappop(frontier)

        if state in visited and visited[state] <= cost:
            continue
        visited[state] = cost

        if state == GOAL_STATE:
            return reconstruct_path(parent, state), nodes_generated, cost

        for neighbor, action, step_cost in get_neighbors(state):
            nodes_generated += 1
            new_cost = cost + step_cost
            if neighbor not in visited or visited.get(neighbor, float('inf')) > new_cost:
                parent[neighbor] = (state, action)
                counter += 1
                heapq.heappush(frontier, (new_cost, counter, neighbor))

    return None, nodes_generated, 0


def astar(initial_state, heuristic=heuristic_manhattan):
    """
    Búsqueda A*
    ────────────
    Combina el costo acumulado g(n) con una heurística h(n): f(n) = g(n) + h(n).
    Con heurísticas admisibles garantiza la solución óptima.
    Es significativamente más eficiente que BFS/UCS gracias a la guía heurística.
    """
    if initial_state == GOAL_STATE:
        return [], 0, 0

    counter = 0
    h0 = heuristic(initial_state)
    frontier = [(h0, counter, 0, initial_state)]
    g_cost  = {initial_state: 0}
    parent  = {}
    nodes_generated = 1

    while frontier:
        f, _, g, state = heapq.heappop(frontier)

        if state == GOAL_STATE:
            return reconstruct_path(parent, state), nodes_generated, g

        if g > g_cost.get(state, float('inf')):
            continue

        for neighbor, action, step_cost in get_neighbors(state):
            nodes_generated += 1
            new_g = g + step_cost
            if new_g < g_cost.get(neighbor, float('inf')):
                g_cost[neighbor] = new_g
                h = heuristic(neighbor)
                f_val = new_g + h
                parent[neighbor] = (state, action)
                counter += 1
                heapq.heappush(frontier, (f_val, counter, new_g, neighbor))

    return None, nodes_generated, 0

def run_algorithm(name, func, initial_state, *args):
    """Ejecuta un algoritmo, mide tiempo y retorna resultados."""
    start = time.perf_counter()
    path, nodes, cost = func(initial_state, *args)
    elapsed = time.perf_counter() - start

    result = {
        'name':   name,
        'path':   path,
        'nodes':  nodes,
        'cost':   cost,
        'time':   elapsed,
        'length': len(path) if path else None,
    }
    return result


def print_results(results):
    """Imprime tabla comparativa de resultados."""
    print_separator("TABLA COMPARATIVA DE RESULTADOS")

    header = f"{'Algoritmo':<28} {'Nodos Gen.':>11} {'Long. Sol.':>11} {'Costo':>8} {'Tiempo (s)':>12}"
    print(header)
    print("-" * 74)

    for r in results:
        length = str(r['length']) if r['length'] is not None else "N/A"
        cost   = str(r['cost'])   if r['cost']   is not None else "N/A"
        print(
            f"  {r['name']:<26} {r['nodes']:>11,} {length:>11} {cost:>8} {r['time']:>12.6f}"
        )


def print_solution_path(result, initial_state):
    """Muestra el camino solución paso a paso (primeros y últimos pasos)."""
    if result['path'] is None:
        print(f"\n  {result['name']}: No encontró solución.")
        return

    print(f"\n  Solución con {result['name']} ({result['length']} movimientos):")
    print(f"  Movimientos: {' → '.join(result['path'])}")

    state = initial_state
    states = [state]
    for action in result['path']:
        zero_idx = state.index(0)
        delta = MOVES[action]
        new_state = list(state)
        new_state[zero_idx], new_state[zero_idx + delta] = \
            new_state[zero_idx + delta], new_state[zero_idx]
        state = tuple(new_state)
        states.append(state)

    show = list(range(min(4, len(states))))
    if len(states) > 5:
        show += ['...']
        show += list(range(len(states) - 2, len(states)))

    print()
    for idx in show:
        if idx == '...':
            print("       ...")
            continue
        label = f"Paso {idx}" if idx > 0 else "Inicio"
        if idx == len(states) - 1:
            label = f"Meta  (paso {idx})"
        print_state(states[idx], label)


def conclusions(results):
    """Imprime conclusiones basadas en los resultados obtenidos."""
    print_separator("CONCLUSIONES")

    valid = [r for r in results if r['path'] is not None]

    best_nodes  = min(valid, key=lambda r: r['nodes'])
    best_time   = min(valid, key=lambda r: r['time'])
    best_length = min(valid, key=lambda r: r['length'])

    print(f"""
  1. EFICIENCIA EN NODOS GENERADOS
     El algoritmo que exploró menos nodos fue: {best_nodes['name']}
     ({best_nodes['nodes']:,} nodos). Las heurísticas reducen drásticamente
     el espacio de búsqueda al dirigir la exploración hacia el objetivo.

  2. VELOCIDAD DE EJECUCIÓN
     El más rápido fue: {best_time['name']} ({best_time['time']:.6f} s).

  3. CALIDAD DE SOLUCIÓN
     La solución más corta la encontró: {best_length['name']}
     ({best_length['length']} movimientos).

  4. BÚSQUEDA NO INFORMADA (BFS / DFS / UCS)
     • BFS y UCS garantizan solución óptima pero generan muchos nodos.
     • DFS puede encontrar soluciones largas o no óptimas; usa poca memoria
       pero su solución puede ser mucho más larga que la óptima.
     • UCS es equivalente a BFS cuando todos los costos son iguales (=1).

  5. BÚSQUEDA INFORMADA (A*)
     • A* con distancia Manhattan es el más eficiente: combina el costo
       real recorrido con una estimación del costo restante.
     • La distancia Manhattan es más informativa que "fichas fuera de lugar"
       porque considera cuán lejos está cada ficha de su meta.
     • Las heurísticas admisibles garantizan que A* sea óptimo.

  6. IMPORTANCIA DE LAS HEURÍSTICAS
     Una buena heurística es la clave para escalar los algoritmos de búsqueda
     a problemas complejos. Sin heurística, el espacio de estados del 8-Puzzle
     (9! / 2 ≈ 181,440 estados alcanzables) puede hacer impráctico a BFS/DFS
     para estados iniciales difíciles.
""")

def main():
    INITIAL_STATE = (4, 2, 3,
                     5, 0, 6,
                     7, 1, 8)

    print_separator("8-PUZZLE – BÚSQUEDA INFORMADA Y NO INFORMADA")
    print_state(INITIAL_STATE, "Estado Inicial")
    print_state(GOAL_STATE,    "Estado Meta")

    if not is_solvable(INITIAL_STATE):
        print("\n  ⚠️  ERROR: El estado inicial NO es resoluble.")
        print("  El número de inversiones es impar — no existe solución.")
        print("  Cambia INITIAL_STATE por un estado con inversiones pares.\n")
        return

    print("  ✓ Estado verificado: es resoluble (inversiones pares)\n")

    print_separator("EJECUTANDO ALGORITMOS...")

    results = []

    print("  [1/5] BFS  ...", end=" ", flush=True)
    r = run_algorithm("BFS (Anchura)", bfs, INITIAL_STATE)
    results.append(r)
    print(f"✓  ({r['nodes']:,} nodos, {r['time']:.4f}s)")

    print("  [2/5] DFS  ...", end=" ", flush=True)
    r = run_algorithm("DFS (Profundidad, lím=50)", dfs, INITIAL_STATE, 50)
    results.append(r)
    print(f"✓  ({r['nodes']:,} nodos, {r['time']:.4f}s)")

    print("  [3/5] UCS  ...", end=" ", flush=True)
    r = run_algorithm("UCS (Costo Uniforme)", ucs, INITIAL_STATE)
    results.append(r)
    print(f"✓  ({r['nodes']:,} nodos, {r['time']:.4f}s)")

    print("  [4/5] A* (Fichas fuera de lugar) ...", end=" ", flush=True)
    r = run_algorithm("A* (Fichas fuera de lugar)", astar, INITIAL_STATE, heuristic_misplaced)
    results.append(r)
    print(f"✓  ({r['nodes']:,} nodos, {r['time']:.4f}s)")

    print("  [5/5] A* (Distancia Manhattan)   ...", end=" ", flush=True)
    r = run_algorithm("A* (Distancia Manhattan)", astar, INITIAL_STATE, heuristic_manhattan)
    results.append(r)
    print(f"✓  ({r['nodes']:,} nodos, {r['time']:.4f}s)")

    print_results(results)

    print_separator("CAMINOS SOLUCIÓN")
    for r in results:
        print_solution_path(r, INITIAL_STATE)

    conclusions(results)


if __name__ == "__main__":
    main()
