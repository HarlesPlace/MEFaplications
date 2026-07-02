import numpy as np
from matplotlib.path import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

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
    def __init__(self, id, nodes):
        self.id = id
        self.nodes = nodes
        self.is_inside = None  # True se o elemento estiver dentro da geometria, False caso contrário

    def __repr__(self):
        ids = [n.id for n in self.nodes]
        return f"Elemento {self.id}: {ids}"
    
    def centroid(self):
        xc = sum(node.x for node in self.nodes) / 4
        yc = sum(node.y for node in self.nodes) / 4
        return xc, yc
    
class Mesh:
    def __init__(self, geometry, Nx, Ny):
        self.geometry = geometry

        self.Nx = Nx
        self.Ny = Ny

        self.nodes = []
        self.elements = []

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

    def print_connectivity(self):
        for element in self.elements:
            print(
                element.id,
                [node.id for node in element.nodes]
            )
        
    def classify_elements(self):
        for element in self.elements:
            xc, yc = element.centroid()
            element.is_inside = self.geometry.contains(xc, yc)

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

def main():
    geometry = Geometry([(0, 0), (100, 100), (200, 0), (0,0)])
    geometry.add_circular_hole(center=(100, 50), radius=10)
    mesh = Mesh(geometry=geometry, Nx=40, Ny=20)
    mesh.generate()
    mesh.print_connectivity()
    mesh.plot_mesh()

if __name__ == "__main__":
    main()