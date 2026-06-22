import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

class Portico():
    def __init__(self, p1, p2, id):
        self.r_in = 0.032 # m
        self.r_out = 0.04 # m
        self.E = 210e9 # Pa
        self.rho = 7850 # kg/m³
        self.id = id
        self.p1 = p1
        self.p2 = p2
        self.theta = np.arctan2((p2[1] - p1[1]), (p2[0] - p1[0])) # rad
        self.L = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2) # m
        self.A = np.pi * (self.r_out**2 - self.r_in**2) # m²
        self.I = (np.pi / 4) * (self.r_out**4 - self.r_in**4) # m^4
        self.m = self.rho * self.A * self.L # kg
        self.peso = self.m * 9.81 # N
    
    def __str__(self):
        return f"Portico {self.id}: p1={self.p1}, p2={self.p2}, L={self.L:.2f} m, Theta={np.degrees(self.theta):.2f}°"

class Trelica():
    def __init__(self, p1, p2, id):
        self.r = 0.012 # m
        self.E = 70e9 # Pa
        self.rho = 2700 # kg/m³
        self.id = id
        self.p1 = p1
        self.p2 = p2
        self.theta = np.arctan2((p2[1] - p1[1]), (p2[0] - p1[0])) # rad
        self.L = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2) # m
        self.A = np.pi * self.r**2  # m^2
        self.I = (np.pi / 4) * self.r**4  # m^4
        self.m = self.rho * self.A * self.L # kg
        self.peso = self.m * 9.81 # N

    def __str__(self):
        return f"Trelica {self.id}: p1={self.p1}, p2={self.p2}, L={self.L:.2f} m, Theta={np.degrees(self.theta):.2f}°"

class Estrutura():
    def __init__(self):
        self.elementos = []
        self.q = 7500 # N/m
        self.Fw = 2000 # N
        self.f = 1/60 # Hz
        self.build()
    
    def build(self):
        self.elementos.append(Portico((0.0, 0.0), (0.0, 1.5), 1))
        self.elementos.append(Portico((1.5, 0.0), (1.5, 1.5), 2))
        self.elementos.append(Portico((0.0, 1.5), (1.5, 1.5), 3))
        self.elementos.append(Portico((0.0, 1.5), (0.0, 3.0), 4))
        self.elementos.append(Portico((0.0, 1.5), (1.5, 3.0), 5))
        self.elementos.append(Portico((1.5, 1.5), (1.5, 3.0), 6))
        self.elementos.append(Portico((0.0, 3.0), (1.5, 3.0), 7))
        self.elementos.append(Portico((0.0, 3.0), (0.0, 4.5), 8))
        self.elementos.append(Portico((1.5, 3.0), (0.0, 4.5), 9))
        self.elementos.append(Portico((1.5, 3.0), (1.5, 4.5), 10))
        self.elementos.append(Portico((0.0, 4.5), (1.5, 4.5), 11))
        self.elementos.append(Portico((0.0, 4.5), (0.0, 6.0), 12))
        self.elementos.append(Portico((0.0, 4.5), (1.5, 6.0), 13))
        self.elementos.append(Portico((1.5, 4.5), (1.5, 6.0), 14))
        self.elementos.append(Portico((0.0, 6.0), (1.5, 6.0), 15))
        self.elementos.append(Portico((0.0, 6.0), (0.0, 7.5), 16))
        self.elementos.append(Portico((1.5, 6.0), (0.0, 7.5), 17))
        self.elementos.append(Portico((1.5, 6.0), (1.5, 7.5), 18))
        self.elementos.append(Portico((0.0, 7.5), (1.5, 7.5), 19))
        self.elementos.append(Trelica((0.0, 7.5), (1.5, 9.0), 20))
        self.elementos.append(Portico((1.5, 7.5), (1.5, 9.0), 21))
        self.elementos.append(Portico((1.5, 9.0), (3.0, 9.0), 22))
        self.elementos.append(Trelica((3.0, 7.5), (1.5, 9.0), 23))
        self.elementos.append(Portico((1.5, 7.5), (3.0, 7.5), 24))
        self.elementos.append(Portico((3.0, 7.5), (3.0, 9.0), 25))
        self.elementos.append(Portico((3.0, 9.0), (4.5, 9.0), 26))
        self.elementos.append(Trelica((4.5, 7.5), (3.0, 9.0), 27))
        self.elementos.append(Portico((3.0, 7.5), (4.5, 7.5), 28))
        self.elementos.append(Portico((4.5, 7.5), (4.5, 9.0), 29))
        self.elementos.append(Trelica((6.0, 7.5), (4.5, 9.0), 30))
        self.elementos.append(Portico((4.5, 7.5), (6.0, 7.5), 31))
    
    def plot_estrutura(self):
        plt.figure(figsize=(10, 8))
        
        for elemento in self.elementos:
            x = [elemento.p1[0], elemento.p2[0]]
            y = [elemento.p1[1], elemento.p2[1]]
            if isinstance(elemento, Portico):
                plt.plot(x, y, color="royalblue", linewidth=4.0)
            elif isinstance(elemento, Trelica):
                plt.plot(x, y, color="darkorange", linewidth=1.5)
                
        legenda_portico = mlines.Line2D([], [], color="royalblue", linewidth=4.0, label="Pórtico")
        legenda_trelica = mlines.Line2D([], [], color="darkorange", linewidth=1.5, label="Treliça")
        plt.legend(handles=[legenda_portico, legenda_trelica], loc="upper right")
        
        plt.title("Visualização da Estrutura - EP Mecânica Computacional", fontsize=14, fontweight="bold")
        plt.xlabel("Posição X (m)", fontsize=12)
        plt.ylabel("Posição Y (m)", fontsize=12)
        plt.grid(True, linestyle=":", alpha=0.6)
        passo = 1.5
        ticks_x = np.arange(-1.5, 7.5, passo)  # [0.0, 1.5, 3.0, 4.5, 6.0]
        ticks_y = np.arange(-1.5, 10.5, passo)  # [0.0, 1.5, 3.0, 4.5, 6.0, 7.5, 9.0]
        plt.xticks(ticks_x)
        plt.yticks(ticks_y)
        plt.axis("equal") 
        
        plt.show()


def main():
    estrutura = Estrutura()
    estrutura.plot_estrutura()

if __name__ == "__main__":    
    main()