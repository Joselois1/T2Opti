from pulp import LpProblem, LpMinimize, LpVariable, lpSum, value
import ast
import numpy as np

def preprocess_content(contenido):
    preprocessed_content = ""
    for line in contenido.splitlines():
        if line.startswith(" ") or line == "":
            preprocessed_content += line + "\n"
    return preprocessed_content

def crear_matrix(contenido):
    matrices = []
    matrix = []
    for line in contenido.splitlines():
        if line != "":
            # Convert each line into a list of integers and add it to the current matrix
            matrix.append(list(map(int, line.split())))
        else:
            # If the line is empty, add the current matrix to the list of matrices and start a new one
            matrices.append(matrix)
            matrix = []
    # Don't forget to add the last matrix if it's not empty
    if matrix:
        matrices.append(matrix)
    return matrices

def make_matrix_square(matrix):
    num_rows = len(matrix)
    num_cols = len(matrix[0])

    if num_rows == num_cols:
        return matrix  # La matriz ya es cuadrada
    max_dim = max(num_rows, num_cols)
    # Agregar filas y columnas adicionales con valores altos (por ejemplo, 1000)
    for i in range(max_dim - num_rows):
        matrix.append([1000] * num_cols)

    for i in range(num_cols, max_dim):
        for row in matrix:
            row.append(1000)

    return matrix

def create_matrices_from_file(filename):
    matrices = []
    with open(filename, 'r') as file:
        content = file.read()
        preprocessed_content = preprocess_content(content)
        raw_matrices = crear_matrix(preprocessed_content)
        for matrix in raw_matrices:
            square_matrix = make_matrix_square(matrix)
            matrices.append(square_matrix)
    return matrices

def imprimir_matrices(matrices):
    for i, matrix in enumerate(matrices):
        print(f"Matrix {i+1}:")
        for row in matrix:
            print(' '.join(str(x) for x in row))
        print("\n")  # Print a newline character after each matrix for better readability

def generate_subsets(cities):
    if len(cities) == 0:
        return [[]]
    subsets = []
    first_city = cities[0]
    remaining_cities = cities[1:]
    # For all subsets of remaining_cities
    for subset in generate_subsets(remaining_cities):
        # Include first_city in the subset
        subsets.append([first_city] + subset)
        # Do not include first_city in the subset
        subsets.append(subset)
    return subsets

def resolver_atsp_dfj(matrices):
    # Create a list to store the solutions for each matrix
    solutions = []

    for k, matrix in enumerate(matrices):
        # Determine the dimensions of the current matrix
        num_rows = len(matrix)
        num_cols = len(matrix[0])

        # Create a problem for minimization
        prob = LpProblem("ATSP_DFJ", LpMinimize)
        # Create decision variables
        x = LpVariable.dicts('x', ((i, j, k) for i in range(num_rows) for j in range(num_cols)), cat='Binary')
        # Create variables for MTZ constraints
        u = LpVariable.dicts('u', range(num_rows), lowBound=0, upBound=num_rows-1, cat='Integer')
        # Create the objective function
        prob += lpSum([matrix[i][j] * x[(i, j, k)] for i in range(num_rows) for j in range(num_cols)])

        # Create constraints

        # Each city must be visited exactly once
        for i in range(num_rows):
            prob += lpSum([x[(i, j, k)] for j in range(num_cols)]) == 1

        # Prevent subcircuits using Miller-Tucker-Zemlin (MTZ) constraints
        for i in range(1, num_rows):
            for j in range(1, num_rows):
                if i != j:
                    prob += u[i] - u[j] + (num_rows - 1) * x[(i, j, k)] <= num_rows - 2

        # Solve the problem
        prob.solve()

        # Store the solution for this matrix
        solution = []
        for i in range(num_rows):
            for j in range(num_cols):
                if value(x[(i, j, k)]) == 1:
                    solution.append((i, j))
        
        # Append the solution to the list of solutions
        solutions.append(solution)

    return solutions


def resolver_atsp_mtz(matrices):
    solutions = []

    for k, matrix in enumerate(matrices):
        num_rows = len(matrix)
        num_cols = len(matrix[0])

        prob = LpProblem("ATSP_MTZ", LpMinimize)
        x = LpVariable.dicts('x', ((i, j, k) for i in range(num_rows) for j in range(num_cols)), cat='Binary')
        u = LpVariable.dicts('u', range(num_rows), lowBound=0, upBound=num_rows-1, cat='Integer')

        prob += lpSum([matrix[i][j] * x[(i, j, k)] for i in range(num_rows) for j in range(num_cols)])

        for i in range(num_rows):
            prob += lpSum([x[(i, j, k)] for j in range(num_cols)]) == 1

        for i in range(num_rows):
            prob += lpSum([x[(j, i, k)] for j in range(num_cols)]) == 1

        for i in range(1, num_rows):
            for j in range(1, num_rows):
                if i != j:
                    prob += u[i] - u[j] + (num_rows - 1) * x[(i, j, k)] <= num_rows - 2

        prob.solve()

        solution = []
        for i in range(num_rows):
            for j in range(num_cols):
                if value(x[(i, j, k)]) == 1:
                    solution.append((i, j))

        solutions.append(solution)

    return solutions

def resolver_atsp_gg(matrices):
    solutions = []

    for k, matrix in enumerate(matrices):
        num_rows = len(matrix)
        num_cols = len(matrix[0])

        prob = LpProblem("ATSP_GG", LpMinimize)
        x = LpVariable.dicts('x', ((i, j, k) for i in range(num_rows) for j in range(num_cols)), cat='Binary')

        prob += lpSum([matrix[i][j] * x[(i, j, k)] for i in range(num_rows) for j in range(num_cols)])

        for i in range(num_rows):
            prob += lpSum([x[(i, j, k)] for j in range(num_cols)]) == 1

        for j in range(num_cols):
            prob += lpSum([x[(i, j, k)] for i in range(num_rows)]) == 1

        prob.solve()

        solution = []
        for i in range(num_rows):
            for j in range(num_cols):
                if value(x[(i, j, k)]) == 1:
                    solution.append((i, j))

        solutions.append(solution)

    return solutions


# Open the file and read its contents
archivo = open("rbg323.atsp", "r")
contenido = archivo.read()
archivo.close()

# Preprocess the content to remove irrelevant lines
preprocessed_content = preprocess_content(contenido)

matrices = create_matrices_from_file("rbg323.atsp")
# Convert the preprocessed content into a list of lists
#matrices = ast.literal_eval(preprocessed_content + "\n" + matrix_data_as_string)

# Print the matrices

#imprimir_matrices(matrices)
#soluciones_djf = resolver_atsp_dfj(matrices)
#soluciones_mtz = resolver_atsp_mtz(matrices)
#soluciones_gg = resolver_atsp_gg(matrices)


def calcular_distancia_ruta(ruta, matriz_distancias):
    distancia = 0
    k = ruta[0][2]  # Obtener el índice de instancia desde la primera ruta
    for i in range(len(ruta) - 1):
        ciudad_actual = ruta[i][0]  # Ciudad actual en la ruta
        ciudad_siguiente = ruta[i + 1][0]  # Ciudad siguiente en la ruta
        distancia += matriz_distancias[k][ciudad_actual][ciudad_siguiente]
    # Agregar la distancia de regreso a la ciudad inicial
    distancia += matriz_distancias[k][ruta[-1][0]][ruta[0][0]]
    return distancia

def comparar_resultados(formulacion1, formulacion2, formulacion3, matrices_distancias):
    resultados = []
    
    for i, matriz_distancias_instancia in enumerate(matrices_distancias):
        ruta_dfj = formulacion1[i]
        ruta_mtz = formulacion2[i]
        ruta_gg = formulacion3[i]
        
        distancia_dfj = calcular_distancia_ruta(ruta_dfj, matriz_distancias_instancia)
        distancia_mtz = calcular_distancia_ruta(ruta_mtz, matriz_distancias_instancia)
        distancia_gg = calcular_distancia_ruta(ruta_gg, matriz_distancias_instancia)
        
        resultados.append({
            "Instancia": i + 1,
            "Distancia_DFJ": distancia_dfj,
            "Distancia_MTZ": distancia_mtz,
            "Distancia_GG": distancia_gg
        })
    
    return resultados


# Llama a las funciones para resolver las instancias con las tres formulaciones
soluciones_dfj = resolver_atsp_dfj(matrices)
soluciones_mtz = resolver_atsp_mtz(matrices)
soluciones_gg = resolver_atsp_gg(matrices)

# Suponiendo que tienes una lista de matrices de distancias llamada matrices_distancias
resultados = comparar_resultados(soluciones_dfj, soluciones_mtz, soluciones_gg, matrices)

# Imprime los resultados
for resultado in resultados:
    print(f"Instancia {resultado['Instancia']}:")
    print(f"Distancia DFJ: {resultado['Distancia_DFJ']}")
    print(f"Distancia MTZ: {resultado['Distancia_MTZ']}")
    print(f"Distancia GG: {resultado['Distancia_GG']}")
    print()





