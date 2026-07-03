import numpy as np
from matplotlib.path import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import os

results_dir = "resultadosEPextra"

def salvar_grafico(plt, nome_pasta, nome_arquivo):
    os.makedirs(nome_pasta, exist_ok=True)
    caminho_completo = os.path.join(nome_pasta, nome_arquivo)
    plt.savefig(caminho_completo, dpi=300, bbox_inches='tight')
    print(f"-> Gráfico salvo com sucesso em: {caminho_completo}")
    plt.close()

class Material:
    def __init__(self, E, poisson, thickness):
        self.E = E
        self.poisson = poisson
        self.thickness = thickness
    
    def constitutive_matrix(self):
        coef = self.E / (1 - self.poisson**2)
        C = coef * np.array([
            [1, self.poisson, 0],
            [self.poisson, 1, 0],
            [0, 0, (1-self.poisson)/2]
        ])
        return C

class Geometry:
    def __init__(self, vertices):
        """
        vertices: lista de tuplas (x, y) do contorno externo da peça.
        Exemplo:
        [(0,0), (100,0), (100,50), (50,80), (0,50)]
        """
        self.vertices = vertices
        self.circular_hole = []
        self.rectangular_hole = []
        self.elliptical_hole = []
    
    def add_circular_hole(self, center, radius):
        """
        center: tupla (x, y) do centro do furo
        radius: raio do furo
        """
        self.circular_hole.append((center, radius))
    
    def add_elliptical_hole(self, center, radius_x, radius_y):
        """
        center: tupla (x, y) do centro do furo
        radius_x: raio do furo no eixo x
        radius_y: raio do furo no eixo y
        """
        self.elliptical_hole.append((center, radius_x, radius_y))
    
    def add_rectangular_hole(self, bottom_left, top_right):
        """
        bottom_left: tupla (x, y) do canto inferior esquerdo do furo
        top_right: tupla (x, y) do canto superior direito do furo
        """
        self.rectangular_hole.append((bottom_left, top_right))

    def bounding_box(self):
        xs = [v[0] for v in self.vertices]
        ys = [v[1] for v in self.vertices]

        xmin = min(xs)
        xmax = max(xs)
        ymin = min(ys)
        ymax = max(ys)

        return xmin, xmax, ymin, ymax
    
    def point_in_polygon(self, x, y):
        polygon = Path(self.vertices)
        return polygon.contains_point((x, y))

    def contains(self, x, y):
        xmin, xmax, ymin, ymax = self.bounding_box()

        # 1) Bounding box
        if x < xmin or x > xmax:
            return False

        if y < ymin or y > ymax:
            return False

        # 2) Contorno externo
        if not self.point_in_polygon(x, y):
            return False

        # 3) Furos circulares
        for center, radius in self.circular_hole:
            dx = x - center[0]
            dy = y - center[1]

            if dx*dx + dy*dy <= radius*radius:
                return False

        # 4) Furos elípticos
        for center, radius_x, radius_y in self.elliptical_hole:
            rx = radius_x
            ry = radius_y

            value = (
                ((x-center[0])/rx)**2
                +
                ((y-center[1])/ry)**2
            )

            if value <= 1:
                return False

        # 5) Furos retangulares
        for bl, tr in self.rectangular_hole:
            if (
                bl[0] <= x <= tr[0]
                and
                bl[1] <= y <= tr[1]
            ):
                return False

        return True
    
class Node:
    def __init__(self, id, x, y):
        self.id = id
        self.x = x
        self.y = y

    def __repr__(self):
        return f"Node({self.id}, {self.x:.3f}, {self.y:.3f})"
    
class Quad4Element:
    def __init__(self, id, nodes, material=None):
        self.id = id
        self.nodes = nodes
        self.material = material
        self.is_inside = None  # True se o elemento estiver dentro da geometria, False caso contrário

    def __repr__(self):
        ids = [n.id for n in self.nodes]
        return f"Elemento {self.id}: {ids}"
    
    def centroid(self):
        xc = sum(node.x for node in self.nodes) / 4
        yc = sum(node.y for node in self.nodes) / 4
        return xc, yc
    
    def coordinates(self):
        x = np.array([node.x for node in self.nodes])
        y = np.array([node.y for node in self.nodes])
        return x, y

    def shape_functions(self, xi, eta):
        N = np.array([
            0.25*(1-xi)*(1-eta),
            0.25*(1+xi)*(1-eta),
            0.25*(1+xi)*(1+eta),
            0.25*(1-xi)*(1+eta)
        ])
        return N
    
    def shape_function_derivatives(self, xi, eta):
        dN_dxi = np.array([
            -0.25*(1-eta),
            0.25*(1-eta),
            0.25*(1+eta),
            -0.25*(1+eta)
        ])
        dN_deta = np.array([
            -0.25*(1-xi),
            -0.25*(1+xi),
            0.25*(1+xi),
            0.25*(1-xi)
        ])
        return dN_dxi, dN_deta
    
    def jacobian(self, xi, eta):
        x, y = self.coordinates()
        dN_dxi, dN_deta = self.shape_function_derivatives(xi, eta)
        J = np.array([
            [
                np.dot(dN_dxi, x),
                np.dot(dN_dxi, y)
            ],
            [
                np.dot(dN_deta, x),
                np.dot(dN_deta, y)
            ]
        ])
        detJ = np.linalg.det(J)
        invJ = np.linalg.inv(J)
        return J, detJ, invJ

    def shape_function_gradients(self, xi, eta):
        # Derivadas em coordenadas naturais
        dN_dxi, dN_deta = self.shape_function_derivatives(xi, eta)
        # Jacobiano
        J, detJ, invJ = self.jacobian(xi, eta)
        dN_dx = np.zeros(4)
        dN_dy = np.zeros(4)
        for i in range(4):
            grad_nat = np.array([
                dN_dxi[i],
                dN_deta[i]
            ])
            grad_phys = invJ @ grad_nat
            dN_dx[i] = grad_phys[0]
            dN_dy[i] = grad_phys[1]
        return dN_dx, dN_dy, detJ

    def B_matrix(self, xi, eta):
        dN_dx, dN_dy, detJ = self.shape_function_gradients(xi, eta)
        B = np.array([
            [dN_dx[0], 0.0,      dN_dx[1], 0.0,      dN_dx[2], 0.0,      dN_dx[3], 0.0],
            [0.0,      dN_dy[0], 0.0,      dN_dy[1], 0.0,      dN_dy[2], 0.0,      dN_dy[3]],
            [dN_dy[0], dN_dx[0], dN_dy[1], dN_dx[1], dN_dy[2], dN_dx[2], dN_dy[3], dN_dx[3]]
        ])
        return B, detJ
    
    def gauss_points(self):
        a = np.sqrt(3.0 / 5.0)
        points = np.array([
            -a,
            0.0,
            a
        ])
        weights = np.array([
            5/9,
            8/9,
            5/9
        ])
        return points, weights
    
    def stiffness_matrix(self):
        Ke = np.zeros((8, 8))
        C = self.material.constitutive_matrix()
        points, weights = self.gauss_points()
        t = self.material.thickness
        for i, xi in enumerate(points):
            for j, eta in enumerate(points):
                wx = weights[i]
                wy = weights[j]
                B, detJ = self.B_matrix(xi, eta)
                Ke += (B.T@ C @ B) * detJ * wx * wy * t
        return Ke

    def dofs(self):
        """
        Retorna os índices dos graus de liberdade do elemento.

        Ordem:
        [u1, v1, u2, v2, u3, v3, u4, v4]
        """
        dofs = []
        for node in self.nodes:
            dofs.extend([
                2*node.id,
                2*node.id+1
            ])
        return dofs
    
class Mesh:
    def __init__(self, geometry, Nx, Ny):
        self.geometry = geometry
        self.Nx = Nx
        self.Ny = Ny
        self.nodes = []
        self.elements = []
        self.K = None  # Matriz global de rigidez
        self.F = None  # Vetor global de forças

    def generate_nodes(self):
        xmin, xmax, ymin, ymax = self.geometry.bounding_box()
        dx = (xmax - xmin) / self.Nx
        dy = (ymax - ymin) / self.Ny
        node_id = 0
        for j in range(self.Ny + 1):
            y = ymin + j * dy
            for i in range(self.Nx + 1):
                x = xmin + i * dx
                self.nodes.append(Node(node_id, x, y))
                node_id += 1

    def generate_elements(self):
        elem_id = 0
        ncols = self.Nx + 1
        for j in range(self.Ny):
            for i in range(self.Nx):
                n1 = j * ncols + i
                n2 = n1 + 1
                n3 = n2 + ncols
                n4 = n1 + ncols
                self.elements.append(
                    Quad4Element(
                        elem_id,
                        [
                            self.nodes[n1],
                            self.nodes[n2],
                            self.nodes[n3],
                            self.nodes[n4]
                        ]
                    )
                )
                elem_id += 1

    def generate(self):
        self.generate_nodes()
        self.generate_elements()
        self.classify_elements()
        self.F = np.zeros(2 * len(self.nodes))

    def print_connectivity(self):
        for element in self.elements:
            print(
                element.id,
                [node.id for node in element.nodes]
            )
        
    def classify_elements(self):
        solid_material = Material(E=210e9, poisson=0.31, thickness=0.005)
        void_material = Material(E=500, poisson=0.31, thickness=0.005)
        for element in self.elements:
            xc, yc = element.centroid()
            element.is_inside = self.geometry.contains(xc, yc)
            if element.is_inside:
                element.material = solid_material
            else:
                element.material = void_material

    def plot_mesh(self):
        fig, ax = plt.subplots(figsize=(8,8))
        for element in self.elements:
            vertices = [
                (node.x, node.y)
                for node in element.nodes
            ]
            if element.is_inside:
                color = "lightblue"
            else:
                color = "white"
            poly = Polygon(
                vertices,
                closed=True,
                facecolor=color,
                edgecolor="black",
                linewidth=0.5
            )
            ax.add_patch(poly)
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        xmin, xmax, ymin, ymax = self.geometry.bounding_box()
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        plt.show()

    def assemble_global_stiffness(self):
        ndofs = 2 * len(self.nodes)
        K = np.zeros((ndofs, ndofs))
        for element in self.elements:
            Ke = element.stiffness_matrix()
            dofs = element.dofs()
            for i in range(8):
                I = dofs[i]
                for j in range(8):
                    J = dofs[j]
                    K[I, J] += Ke[i, j]
        
        self.K = K
        self.plot_global_stiffness()
        return K

    def plot_global_stiffness(self):
        if self.K is None:
            raise ValueError("A matriz global ainda não foi montada.")
        plt.figure(figsize=(8, 8))
        plt.spy(
            self.K,
            markersize=1
        )
        plt.title("Perfil de Esparsidade da Matriz Global de Rigidez (K)")
        plt.xlabel("Graus de liberdade")
        plt.ylabel("Graus de liberdade")
        plt.tight_layout()
        salvar_grafico(plt, results_dir, "perfil_esparsidade_K.png")

    def find_nodes_on_edge(self, p1, p2, tol=1e-6):
        edge_nodes = []
        x1, y1 = p1
        x2, y2 = p2
        edge = np.array([x2-x1, y2-y1])
        edge_length = np.linalg.norm(edge)
        for node in self.nodes:
            v = np.array([node.x-x1, node.y-y1])
            # distância perpendicular
            cross = abs(edge[0]*v[1] - edge[1]*v[0])
            # projeção ao longo da aresta
            dot = np.dot(v, edge)
            if (
                cross < tol*edge_length and
                0 <= dot <= edge_length**2
            ):
                edge_nodes.append(node)
        # ordenar ao longo da aresta
        edge_nodes.sort(
            key=lambda n: np.dot(
                np.array([n.x-x1, n.y-y1]),
                edge
            )
        )
        return edge_nodes
    
    def apply_edge_load(self, p1, p2, traction, angle_deg):
        F = np.zeros(2*len(self.nodes))
        nodes = self.find_nodes_on_edge(p1, p2)
        if len(nodes) < 2:
            raise ValueError("Nenhum nó encontrado na aresta.")
        angle = np.deg2rad(angle_deg)
        tx = traction * np.cos(angle)
        ty = traction * np.sin(angle)
        edge_length = np.linalg.norm(np.array(p2) - np.array(p1))
        Fx_total = tx * self.material.thickness * edge_length
        Fy_total = ty * self.material.thickness * edge_length
        n = len(nodes)
        for i, node in enumerate(nodes):
            if i == 0 or i == n - 1:
                factor = 0.5
            else:
                factor = 1.0
            Fx = factor * Fx_total / (n - 1)
            Fy = factor * Fy_total / (n - 1)
            F[2 * node.id] += Fx
            F[2 * node.id + 1] += Fy
        
        self.F += F
        return F

def main():
    geometry = Geometry([(0, 0), (0, 32.5), (23, 32.5), 
                         (69, 14), (69, 32.5), (115, 32.5), 
                         (161, 14), (161, 0) ])
    geometry.add_elliptical_hole(center=(0, 0), radius_x=23, radius_y=9.25)
    geometry.add_rectangular_hole(bottom_left=(92, -9.25), top_right=(115, 9.25))  
    mesh = Mesh(geometry=geometry, Nx=20, Ny=10)
    mesh.generate()
    mesh.print_connectivity()
    mesh.plot_mesh()
    mesh.assemble_global_stiffness()
    F1 = mesh.apply_edge_load(
        p1=(161, 14),
        p2=(161, 0),
        traction=20e6,
        angle_deg=0
    )

    F2 = mesh.apply_edge_load(
        p1=(69, 32.5),
        p2=(115, 32.5),
        traction=15e6,
        angle_deg=110
    )

if __name__ == "__main__":
    main()