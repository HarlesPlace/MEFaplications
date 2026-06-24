import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import os
from scipy.linalg import eigh

results_dir = "resultados"

def salvar_grafico(plt, nome_pasta, nome_arquivo):
    os.makedirs(nome_pasta, exist_ok=True)
    caminho_completo = os.path.join(nome_pasta, nome_arquivo)
    plt.savefig(caminho_completo, dpi=300, bbox_inches='tight')
    print(f"-> Gráfico salvo com sucesso em: {caminho_completo}")

def salvar_matriz(M, nome_pasta, nome_base="estrutura"):
    os.makedirs(nome_pasta, exist_ok=True)
    caminho_txt = os.path.join(nome_pasta, f"M_{nome_base}.txt")
    np.savetxt(caminho_txt, M, fmt='%.4e', delimiter='\t')
    print(f"-> Matriz {nome_base} (.txt) salva em: {caminho_txt}")

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
        self.k_matriz = self.build_matriz_rigidez()
        self.m_matriz = self.build_matriz_massa()
    
    def __str__(self):
        return f"Portico {self.id}: p1={self.p1}, p2={self.p2}, L={self.L:.2f} m, Theta={np.degrees(self.theta):.2f}°"
    
    def build_matriz_rigidez(self):
        c = np.cos(self.theta)
        s = np.sin(self.theta)
        L = self.L
        
        T = np.array([
            [ c,  s,  0,  0,  0,  0],
            [-s,  c,  0,  0,  0,  0],
            [ 0,  0,  1,  0,  0,  0],
            [ 0,  0,  0,  c,  s,  0],
            [ 0,  0,  0, -s,  c,  0],
            [ 0,  0,  0,  0,  0,  1]
        ])
        
        EA_L = (self.E * self.A) / L
        EI_L3 = (self.E * self.I) / (L**3)
        EI_L2 = (self.E * self.I) / (L**2)
        EI_L  = (self.E * self.I) / L
        
        K_local = np.array([
            [ EA_L,     0,          0,       -EA_L,     0,          0      ],
            [   0,   12*EI_L3,   6*EI_L2,       0,  -12*EI_L3,   6*EI_L2   ],
            [   0,    6*EI_L2,   4*EI_L,        0,   -6*EI_L2,   2*EI_L    ],
            [-EA_L,     0,          0,        EA_L,     0,          0      ],
            [   0,  -12*EI_L3,  -6*EI_L2,       0,   12*EI_L3,  -6*EI_L2   ],
            [   0,    6*EI_L2,   2*EI_L,        0,   -6*EI_L2,   4*EI_L    ]
        ])
        
        return T.T @ K_local @ T
    
    def build_matriz_massa(self):
        c = np.cos(self.theta)
        s = np.sin(self.theta)
        L = self.L
        
        T = np.array([
            [ c,  s,  0,  0,  0,  0],
            [-s,  c,  0,  0,  0,  0],
            [ 0,  0,  1,  0,  0,  0],
            [ 0,  0,  0,  c,  s,  0],
            [ 0,  0,  0, -s,  c,  0],
            [ 0,  0,  0,  0,  0,  1]
        ])
        
        M_total = self.m
        
        M_local = (M_total / 420) * np.array([
            [420*(2/6),    0,       0,      420*(1/6),    0,       0     ],
            [ 0,   156,    22*L,       0,   54,   -13*L     ],
            [ 0,  22*L,  4*L**2,       0, 13*L,  -3*L**2    ],
            [ 420*(1/6),    0,       0,     420*(2/6),    0,       0     ],
            [ 0,    54,    13*L,       0,  156,   -22*L     ],
            [ 0, -13*L, -3*L**2,       0,-22*L,  4*L**2     ]
        ])
        
        return T.T @ M_local @ T
    
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
        self.k_matriz = self.build_matriz_rigidez()
        self.m_matriz = self.build_matriz_massa()

    def __str__(self):
        return f"Trelica {self.id}: p1={self.p1}, p2={self.p2}, L={self.L:.2f} m, Theta={np.degrees(self.theta):.2f}°"

    def build_matriz_rigidez(self):
        c = np.cos(self.theta)
        s = np.sin(self.theta)
        k = (self.E * self.A) / self.L
        k_matrix = k * np.array([[c**2, c*s, -c**2, -c*s],
                                  [c*s, s**2, -c*s, -s**2],
                                  [-c**2, -c*s, c**2, c*s],
                                  [-c*s, -s**2, c*s, s**2]])
        return k_matrix
    
    def build_matriz_massa(self):
        c = np.cos(self.theta)
        s = np.sin(self.theta)
        m_matriz = (self.m / 6) * np.array([[2*c**2, 2*c*s, c**2, c*s],
                                            [2*c*s, 2*s**2, c*s, s**2],
                                            [c**2, c*s, 2*c**2, c*s],
                                            [c*s, s**2, c*s, 2*s**2]])
        return m_matriz

class Estrutura():
    def __init__(self):
        self.elementos = []
        self.q = 7500 # N/m
        self.Fw = 2000 # N
        self.f = 1/60 # Hz
        self.nos = []
        self.no_para_id = {}
        self.build()
        self.mapear_nos()
        self.K_global, self.M_global = self.montar_matrizes_globais()
        self.seis_frequencias_hz = []
        self.seis_modos_vibracao = []
    
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
        salvar_grafico(self.plot_estrutura(), results_dir, "EstrturaConstruida.png")
    
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
        return plt 
    
    def mapear_nos(self):
        conjunto_nos = set()
        for elem in self.elementos:
            conjunto_nos.add(elem.p1)
            conjunto_nos.add(elem.p2)
            
        self.nos = sorted(list(conjunto_nos), key=lambda p: (p[1], p[0]))
        self.no_para_id = {no: idx for idx, no in enumerate(self.nos)}
        
    def montar_matrizes_globais(self):
        num_nos = len(self.nos)
        num_gdl = 3 * num_nos
        
        # Inicializa matrizes globais zeradas
        K_g = np.zeros((num_gdl, num_gdl))
        M_g = np.zeros((num_gdl, num_gdl))
        
        for elem in self.elementos:
            id_n1 = self.no_para_id[elem.p1]
            id_n2 = self.no_para_id[elem.p2]
            
            ke = elem.k_matriz
            me = elem.m_matriz
            
            if isinstance(elem, Portico):
                gdl_globais = [
                    3*id_n1, 3*id_n1+1, 3*id_n1+2, # Nó 1: ux, uy, theta
                    3*id_n2, 3*id_n2+1, 3*id_n2+2  # Nó 2: ux, uy, theta
                ]
                # Soma elemento por elemento nas posições globais
                for i in range(6):
                    for j in range(6):
                        K_g[gdl_globais[i], gdl_globais[j]] += ke[i, j]
                        M_g[gdl_globais[i], gdl_globais[j]] += me[i, j]
                        
            elif isinstance(elem, Trelica):
                gdl_globais = [
                    3*id_n1, 3*id_n1+1, # Nó 1: ux, uy
                    3*id_n2, 3*id_n2+1  # Nó 2: ux, uy
                ]
                # Soma elemento por elemento nas posições globais correspondentes
                for i in range(4):
                    for j in range(4):
                        K_g[gdl_globais[i], gdl_globais[j]] += ke[i, j]
                        M_g[gdl_globais[i], gdl_globais[j]] += me[i, j]
       
        salvar_matriz(K_g, results_dir, nome_base="Rigidez_Global")
        plt.figure(figsize=(6, 6))
        plt.spy(K_g, markersize=1)
        plt.title("Perfil de Esparsidade da Matriz K")
        salvar_grafico(plt, results_dir, "Esparsidade_K.png")
        salvar_matriz(M_g, results_dir, nome_base="Massa_Global")
        plt.figure(figsize=(6, 6))
        plt.spy(M_g, markersize=1)
        plt.title("Perfil de Esparsidade da Matriz M")
        salvar_grafico(plt, results_dir, "Esparsidade_M.png")               
        return K_g, M_g
    
    def analise_modal(self):
        id_no_base1 = self.no_para_id[(0.0, 0.0)]
        id_no_base2 = self.no_para_id[(1.5, 0.0)]

        gdls_fixos = [
            3 * id_no_base1, 3 * id_no_base1 + 1, 3 * id_no_base1 + 2,
            3 * id_no_base2, 3 * id_no_base2 + 1, 3 * id_no_base2 + 2
        ]
        
        num_gdl_total = len(self.nos) * 3
        gdls_livres = [i for i in range(num_gdl_total) if i not in gdls_fixos]
        
        K_restrito = np.delete(self.K_global, gdls_fixos, axis=0)
        K_restrito = np.delete(K_restrito, gdls_fixos, axis=1)
        
        M_restrito = np.delete(self.M_global, gdls_fixos, axis=0)
        M_restrito = np.delete(M_restrito, gdls_fixos, axis=1)
        
        autovalores, autovetores_restritos = eigh(K_restrito, M_restrito)
        
        w2_6 = autovalores[:6]
        frequencias_hz = np.sqrt(w2_6) / (2 * np.pi)
        
        modos_globais = np.zeros((num_gdl_total, 6))
        
        for idx_livre, gdl_real in enumerate(gdls_livres):
            modos_globais[gdl_real, :] = autovetores_restritos[idx_livre, :6]
        
        self.seis_frequencias_hz = frequencias_hz
        self.seis_modos_vibracao = modos_globais
        return frequencias_hz, modos_globais

    def plotar_modos_vibracao(self, fator_escala=0.5):
        for i in range(6):
            plt.figure(figsize=(10, 9))
            
            u_modo = self.seis_modos_vibracao[:, i]
            freq_hz = self.seis_frequencias_hz[i]
            
            for elem in self.elementos:

                id1 = self.no_para_id[elem.p1]
                id2 = self.no_para_id[elem.p2]
                
                x_orig = [elem.p1[0], elem.p2[0]]
                y_orig = [elem.p1[1], elem.p2[1]]
                
                ux1 = u_modo[3 * id1]
                uy1 = u_modo[3 * id1 + 1]
                
                ux2 = u_modo[3 * id2]
                uy2 = u_modo[3 * id2 + 1]
                
                x_def = [elem.p1[0] + fator_escala * ux1, elem.p2[0] + fator_escala * ux2]
                y_def = [elem.p1[1] + fator_escala * uy1, elem.p2[1] + fator_escala * uy2]
                
                plt.plot(x_orig, y_orig, color="lightgray", linestyle="--", linewidth=1.2)
                
                if isinstance(elem, Portico):
                    plt.plot(x_def, y_def, color="royalblue", linewidth=3.0)
                elif isinstance(elem, Trelica):
                    plt.plot(x_def, y_def, color="darkorange", linewidth=1.5)
                    
            passo = 1.5
            plt.xticks(np.arange(0, 6.1, passo))
            plt.yticks(np.arange(0, 9.1, passo))
            plt.grid(True, linestyle=":", alpha=0.4)
            plt.axis("equal")
            
            plt.title(f"Modo {i+1} - Frequência Natural: {freq_hz:.2f} Hz", fontsize=13, fontweight="bold")
            plt.xlabel("X (m)")
            plt.ylabel("Y (m)")
            
            leg_orig = mlines.Line2D([], [], color="lightgray", linestyle="--", label="Não Deformada")
            leg_port = mlines.Line2D([], [], color="royalblue", linewidth=3.0, label="Pórtico (Deformado)")
            leg_trel = mlines.Line2D([], [], color="darkorange", linewidth=1.5, label="Treliça (Deformado)")
            plt.legend(handles=[leg_orig, leg_port, leg_trel], loc="upper right")
            
            nome_arquivo = f"modo_{i+1}_{freq_hz:.1f}Hz.png"
            salvar_grafico(plt, results_dir, nome_arquivo)

def main():
    estrutura = Estrutura()
    #estrutura.plot_estrutura()
    estrutura.analise_modal()
    print(estrutura.seis_frequencias_hz)
    estrutura.plotar_modos_vibracao(fator_escala=3)

if __name__ == "__main__":    
    main()