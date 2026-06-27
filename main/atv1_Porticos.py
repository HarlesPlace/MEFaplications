import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import os
from scipy.linalg import eigh
import matplotlib.animation as animation

results_dir = "resultados"

def salvar_grafico(plt, nome_pasta, nome_arquivo):
    os.makedirs(nome_pasta, exist_ok=True)
    caminho_completo = os.path.join(nome_pasta, nome_arquivo)
    plt.savefig(caminho_completo, dpi=300, bbox_inches='tight')
    print(f"-> Gráfico salvo com sucesso em: {caminho_completo}")
    plt.close()

def salvar_matriz(M, nome_pasta, nome_base="estrutura"):
    os.makedirs(nome_pasta, exist_ok=True)
    caminho_txt = os.path.join(nome_pasta, f"M_{nome_base}.txt")
    np.savetxt(caminho_txt, M, fmt='%.4e', delimiter='\t')
    print(f"-> Matriz {nome_base} (.txt) salva em: {caminho_txt}")

def gerar_gif_animado(estrutura, pasta_dados="resultados_simulacao", append = "", fator_escala=30, skip_frames=2):
    """
    Lê os dados salvos do transiente, renderiza a torre a cada instante de tempo
    e compila tudo num arquivo GIF animado com a força móvel sinalizada.
    
    Argumentos:
        estrutura: Instância da classe Estrutura (para ler a geometria original).
        pasta_dados (str): Pasta onde o arquivo 'dados_transiente.npz' está guardado.
        fator_escala (float): Multiplicador visual para amplificar os deslocamentos.
        skip_frames (int): Renderiza 1 quadro a cada X passos de tempo (otimização).
    """
    caminho_npz = os.path.join(pasta_dados, "dados_transiente.npz")
    
    # 1. Validação do arquivo de dados
    if not os.path.exists(caminho_npz):
        print(f"[Erro] Arquivo de dados não encontrado em: {caminho_npz}")
        print("Rode a simulação de Newmark primeiro para gerar o arquivo.")
        return

    print("-> Carregando dados do transiente para a animação...")
    dados = np.load(caminho_npz)
    hist_tempo = dados['tempo']
    hist_u = dados['deslocamentos']
    
    # Aplica a amostragem para reduzir o tamanho do GIF
    indices_frames = np.arange(0, len(hist_tempo), skip_frames)
    total_frames = len(indices_frames)
    print(f"-> Preparando animação com {total_frames} quadros (Skip={skip_frames}).")

    # 2. Configuração inicial do Plot
    fig, ax = plt.subplots(figsize=(10, 9))
    ax.set_aspect('equal')
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.set_xlabel("X (m)", fontsize=11)
    ax.set_ylabel("Y (m)", fontsize=11)
    
    # Define limites fixos dos eixos para o gráfico não ficar pulando
    ax.set_xlim(-1.5, 7.5)
    ax.set_ylim(-1.5, 10.5)
    ax.set_xticks(np.arange(0, 6.1, 1.5))
    ax.set_yticks(np.arange(0, 9.1, 1.5))

    # 3. Desenha a Estrutura Estática Não Deformada (Cinza ao fundo)
    for elem in estrutura.elementos:
        x_orig = [elem.p1[0], elem.p2[0]]
        y_orig = [elem.p1[1], elem.p2[1]]
        ax.plot(x_orig, y_orig, color="gainsboro", linestyle="--", linewidth=1.0)

    # 4. Criação dos objetos gráficos vazios que serão atualizados no loop do GIF
    linhas_deformadas = []
    for elem in estrutura.elementos:
        if isinstance(elem, Portico):
            line, = ax.plot([], [], color="royalblue", linewidth=2.5)
        else:
            line, = ax.plot([], [], color="darkorange", linewidth=1.2)
        linhas_deformadas.append((elem, line))

    # Marcador dinâmico vermelho para indicar a posição da força móvel Fw
    ponto_forca, = ax.plot([], [], color="crimson", marker="v", markersize=10, label="Carga Móvel Fw")
    
    # Título dinâmico que atualizará com o cronômetro do tempo
    titulo_dinamico = ax.text(0.5, 1.02, "", transform=ax.transAxes, 
                              ha="center", fontsize=12, fontweight="bold")

    # Legendas estáticas organizadas
    leg_orig = mlines.Line2D([], [], color="gainsboro", linestyle="--", label="Não Deformada")
    leg_port = mlines.Line2D([], [], color="royalblue", linewidth=2.5, label="Pórtico")
    leg_trel = mlines.Line2D([], [], color="darkorange", linewidth=1.2, label="Treliça")
    leg_f_mov = mlines.Line2D([], [], color="crimson", marker="v", linestyle="None", markersize=8, label="Força Móvel")
    ax.legend(handles=[leg_orig, leg_port, leg_trel, leg_f_mov], loc="upper right")

    # 5. Função de Atualização de Quadro (Frame por Frame)
    def update(frame_idx):
        t = hist_tempo[frame_idx]
        u_frame = hist_u[:, frame_idx]
        
        # Atualiza a geometria deformada de cada elemento
        for elem, line in linhas_deformadas:
            id1 = estrutura.no_para_id[elem.p1]
            id2 = estrutura.no_para_id[elem.p2]
            
            # Extrai deslocamentos nodais (X e Y) do instante atual
            ux1, uy1 = u_frame[3*id1], u_frame[3*id1+1]
            ux2, uy2 = u_frame[3*id2], u_frame[3*id2+1]
            
            # Aplica o fator de escala e soma à coordenada original
            x_def = [elem.p1[0] + fator_escala * ux1, elem.p2[0] + fator_escala * ux2]
            y_def = [elem.p1[1] + fator_escala * uy1, elem.p2[1] + fator_escala * uy2]
            
            line.set_data(x_def, y_def)
            
        # Atualiza a posição visual da Carga Móvel (p_ref=(1.5, 7.5))
        x_forca = 1.5 + 2.25 * (1.0 - np.cos(2.0 * np.pi * estrutura.f * t))
        y_forca = 7.5
        ponto_forca.set_data([x_forca], [y_forca + 0.1]) # Leve offset vertical para não sobrepor a viga
        
        # Atualiza o cronômetro do título
        titulo_dinamico.set_text(f"Resposta Dinâmica da Torre - Tempo: {t:.2f} s\n(Deslocamentos amplificados {fator_escala}x)")
        
        # Coleta todas as linhas modificadas para otimização de blitting
        return [line for _, line in linhas_deformadas] + [ponto_forca, titulo_dinamico]

    # 6. Geração do Objeto de Animação
    print("-> Renderizando quadros e compilando o GIF (isso pode levar alguns instantes)...")
    ani = animation.FuncAnimation(
        fig, update, frames=indices_frames, 
        interval=50, blit=True
    )

    # 7. Gravação física do arquivo no HD usando o motor Pillow
    caminho_gif = os.path.join(pasta_dados, f"animacao_simulacao{append}.gif")
    ani.save(caminho_gif, writer='pillow', fps=20)
    
    plt.close() # Libera a figura da memória
    print(f"=======================================================")
    print(f" GIF gerado com sucesso em: {caminho_gif}")
    print(f"=======================================================")


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
                                            [c**2, c*s, 2*c**2, 2*c*s],
                                            [c*s, s**2, 2*c*s, 2*s**2]])
        return m_matriz

class Estrutura():
    def __init__(self):
        self.elementos = []
        self.q = 7500 # N/m
        self.Fw = 2000 # N
        self.f = 1/60 # Hz
        self.nos = []
        self.no_para_id = {}
        self.barra_com_carga_distribuida = None
        self.build()
        self.mapear_nos()
        self.K_global, self.M_global = self.montar_matrizes_globais()
        self.F_estatico = self.montar_vetor_forcas_estaticas()
        self.F_q = self.aplicar_carga_distribuida(self.barra_com_carga_distribuida)
        self.seis_frequencias_hz = []
        self.seis_modos_vibracao = []
        self.analise_modal()
        self.alfa = None
        self.beta = None
        self.C_global = self.montar_matriz_amortecimento()
    
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
        elem_22 = Portico((1.5, 9.0), (3.0, 9.0), 22)
        self.elementos.append(elem_22)
        self.barra_com_carga_distribuida = elem_22
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
    
    def montar_vetor_forcas_estaticas(self):
        num_gdl = 3 * len(self.nos)
        F_estatico = np.zeros(num_gdl)
        
        for elem in self.elementos:
            id1 = self.no_para_id[elem.p1]
            id2 = self.no_para_id[elem.p2]
            
            F_estatico[3 * id1 + 1] -= elem.peso / 2.0
            F_estatico[3 * id2 + 1] -= elem.peso / 2.0
        
        return F_estatico

    def aplicar_carga_distribuida(self, elem):
        num_gdl = 3 * len(self.nos)
        F_q = np.zeros(num_gdl)
        
        if elem is None:
            return F_q
        if isinstance(elem, Trelica):
            print(f"[Aviso] O Elemento {elem.id} é uma Treliça! Treliças não suportam cargas distribuídas ao longo do corpo.")
            return F_q
            
        id1 = self.no_para_id[elem.p1]
        id2 = self.no_para_id[elem.p2]
        
        c = np.cos(elem.theta)
        s = np.sin(elem.theta)
        L = elem.L

        T = np.array([
            [ c,  s,  0,  0,  0,  0],
            [-s,  c,  0,  0,  0,  0],
            [ 0,  0,  1,  0,  0,  0],
            [ 0,  0,  0,  c,  s,  0],
            [ 0,  0,  0, -s,  c,  0],
            [ 0,  0,  0,  0,  0,  1]
        ])
        
        f_local = np.array([0, -self.q*L/2, -self.q*(L**2)/12, 0, -self.q*L/2, self.q*(L**2)/12])
        f_global_elem = T.T @ f_local
        
        gdl = [3*id1, 3*id1+1, 3*id1+2, 3*id2, 3*id2+1, 3*id2+2]
        
        for i in range(6):
            F_q[gdl[i]] += f_global_elem[i]
        print(F_q)
        return F_q

    def calcular_vetor_forca_movel(self, t, p_ref):
        """
        Calcula a posição global da força a partir de um ponto de referência p_ref=(x_ref, y_ref)
        e distribui a força pontual Fw nos nós do pórtico correspondente usando as funções
        de forma Hermitianas do MEF de maneira generalizada.
        """
        num_gdl = 3 * len(self.nos)
        F_movel = np.zeros(num_gdl)
        
        x_ref, y_ref = p_ref
        
        # 1. Calcula a coordenada X global da força no instante t
        x_global = x_ref + 2.25 * (1.0 - np.cos(2.0 * np.pi * self.f * t))
        y_global = y_ref
        
        # 2. Encontra qual elemento contém essa coordenada espacial (x_global, y_global)
        elem_atual = None
        for elem in self.elementos:
            # Verifica se o elemento está no mesmo nível Y de referência
            if np.isclose(elem.p1[1], y_global) and np.isclose(elem.p2[1], y_global):
                x_min = min(elem.p1[0], elem.p2[0])
                x_max = max(elem.p1[0], elem.p2[0])
                if x_min <= x_global <= x_max:
                    elem_atual = elem
                    break
                    
        if elem_atual is None:
            return F_movel # Salvaguarda caso a força saia da estrutura
            
        L = elem_atual.L
        
        # 3. Calcula a distância local 'd' a partir do Nó 1 (p1) do elemento
        # Isso garante robustez caso a barra tenha sido definida da direita para a esquerda
        if elem_atual.p2[0] >= elem_atual.p1[0]:
            d = x_global - elem_atual.p1[0]
        else:
            d = elem_atual.p1[0] - x_global
            
        xi = d / L # Coordenada normalizada local (varia de 0 a 1)
        
        # 4. Funções de Forma de Hermite Padronizadas (Viga de Euler-Bernoulli)
        N1 = 1.0 - 3.0 * (xi**2) + 2.0 * (xi**3)        # Translação nó 1
        N2 = L * xi * ((1.0 - xi)**2)                   # Rotação nó 1
        N3 = 3.0 * (xi**2) - 2.0 * (xi**3)              # Translação nó 2
        N4 = L * (xi**2) * (xi - 1.0)                   # Rotação nó 2
        
        # 5. Montagem do vetor de carga local do elemento (6x1)
        # Força Fw atua para baixo (-Y local), gerando forças cortantes e momentos fletores
        f_local = np.zeros(6)
        f_local[1] = -self.Fw * N1  # Esforço cortante no Nó 1
        f_local[2] = -self.Fw * N2  # Momento fletor no Nó 1
        f_local[4] = -self.Fw * N3  # Esforço cortante no Nó 2
        f_local[5] = -self.Fw * N4  # Momento fletor no Nó 2
        
        # 6. Rotação do vetor para o sistema Global (T^T @ f_local)
        c = np.cos(elem_atual.theta)
        s = np.sin(elem_atual.theta)
        
        T = np.array([
            [ c,  s,  0,  0,  0,  0],
            [-s,  c,  0,  0,  0,  0],
            [ 0,  0,  1,  0,  0,  0],
            [ 0,  0,  0,  c,  s,  0],
            [ 0,  0,  0, -s,  c,  0],
            [ 0,  0,  0,  0,  0,  1]
        ])
        f_global_elem = T.T @ f_local
        
        # 7. Assembly no vetor de forças da estrutura
        id1 = self.no_para_id[elem_atual.p1]
        id2 = self.no_para_id[elem_atual.p2]
        gdl = [3*id1, 3*id1+1, 3*id1+2, 3*id2, 3*id2+1, 3*id2+2]
        
        for i in range(6):
            F_movel[gdl[i]] += f_global_elem[i]
            
        return F_movel

    def montar_matriz_amortecimento(self):
        """
        Calcula os coeficientes alfa e beta de Rayleigh e monta as matrizes
        C_global e C_restrito utilizando as frequências do 1º e 6º modo.
        """
        # 1. Recupera as frequências em Hz obtidas na análise modal
        f1 = self.seis_frequencias_hz[0]
        f6 = self.seis_frequencias_hz[5]
        
        # 2. CONVERSÃO CRUCIAL: Hz para rad/s
        omega1 = 2.0 * np.pi * f1
        omega6 = 2.0 * np.pi * f6
        
        xi = 0.06 # Fator de amortecimento de 6%
        
        # 3. Cálculo dos coeficientes pelas fórmulas do enunciado
        self.alfa = (2.0 * xi * omega1 * omega6) / (omega1 + omega6)
        self.beta = (2.0 * xi) / (omega1 + omega6)
        
        print("\n--- AMORTECIMENTO DE RAYLEIGH CALCULADO ---")
        print(f"Frequência Modo 1: {f1:.4f} Hz -> w1 = {omega1:.4f} rad/s")
        print(f"Frequência Modo 6: {f6:.4f} Hz -> w6 = {omega6:.4f} rad/s")
        print(f"Coeficiente Alfa (Massa)  [α]: {self.alfa:.6f}")
        print(f"Coeficiente Beta (Rigidez) [β]: {self.beta:.6e}")
        
        # 4. Construção da Matriz Global de Amortecimento (54x54)
        C = self.alfa * self.M_global + self.beta * self.K_global
        return C

    def simular_newmark(self, dt=0.1, t_max=60.0):
        """
        Executa a integração temporal de Newmark-Beta baseada na formulação de aceleração.
        Salva os históricos completos em arquivos na pasta de resultados para pós-processamento.
        """
        beta = 0.25
        gamma = 0.5
        
        # --- 1. CONFIGURAÇÃO DOS GRAUS DE LIBERDADE ---
        id_no_base1 = self.no_para_id[(0.0, 0.0)]
        id_no_base2 = self.no_para_id[(1.5, 0.0)]
        gdls_fixos = [
            3 * id_no_base1, 3 * id_no_base1 + 1, 3 * id_no_base1 + 2,
            3 * id_no_base2, 3 * id_no_base2 + 1, 3 * id_no_base2 + 2
        ]
        num_gdl_total = len(self.nos) * 3
        gdls_livres = [i for i in range(num_gdl_total) if i not in gdls_fixos]
        
        # --- 2. EXTRAÇÃO DAS MATRIZES RESTRITAS (48 x 48) ---
        K_res = np.delete(self.K_global, gdls_fixos, axis=0)
        K_res = np.delete(K_res, gdls_fixos, axis=1)
        
        M_res = np.delete(self.M_global, gdls_fixos, axis=0)
        M_res = np.delete(M_res, gdls_fixos, axis=1)
        
        C_res = np.delete(self.C_global, gdls_fixos, axis=0)
        C_res = np.delete(C_res, gdls_fixos, axis=1)
        
        # --- 3. MATRIZ DE MASSA EQUIVALENTE (M_CHAPEU) ---
        M_hat = M_res + dt * gamma * C_res + (dt**2) * beta * K_res
        
        # --- 4. VETORES DE CARREGAMENTO ESTÁTICO FIXO (54 x 1) ---
        F_estatico_total = self.F_estatico + self.F_q  
        
        # --- 5. CRIAÇÃO DOS VETORES DE TEMPO E MATRIZES DE HISTÓRICO ---
        self.hist_tempo = np.arange(0, t_max + dt, dt)
        num_passos = len(self.hist_tempo)
        
        self.hist_u = np.zeros((num_gdl_total, num_passos))
        self.hist_R = np.zeros((6, num_passos))
        
        # --- 6. CONDIÇÕES INICIAIS CORRIGIDAS (t = 0) ---
        d = np.zeros(len(gdls_livres))  # Deslocamento inicial livre
        v = np.zeros(len(gdls_livres))  # Velocidade inicial livre
        
        F_global_0 = F_estatico_total + self.calcular_vetor_forca_movel(0.0, p_ref=(1.5, 7.5))
        F_res_0 = np.delete(F_global_0, gdls_fixos)
        
        # Aceleração inicial nos nós livres (48x1)
        a = np.linalg.solve(M_res, F_res_0 - C_res @ v - K_res @ d)
        
        # Guarda o deslocamento inicial (zeros) no histórico global
        self.hist_u[gdls_livres, 0] = d
        
        # Cálculo exato das reações de apoio em t=0 (R0 = M*a0 - F0)
        a_global_0 = np.zeros(num_gdl_total)
        a_global_0[gdls_livres] = a
        R_0 = (self.M_global @ a_global_0) - F_global_0
        self.hist_R[:, 0] = R_0[gdls_fixos]
        
        print("\n=======================================================")
        print(f" INICIANDO INTEGRAÇÃO TEMPORAL DE NEWMARK ({num_passos} passos)")
        print("=======================================================")
        
        # --- 7. LOOP DA MARCHA NO TEMPO ---
        for n in range(1, num_passos):
            t_next = self.hist_tempo[n]
            
            F_global_next = F_estatico_total + self.calcular_vetor_forca_movel(t_next, p_ref=(1.5, 7.5))
            F_res_next = np.delete(F_global_next, gdls_fixos)
            
            preditor_C = v + dt * (1.0 - gamma) * a
            preditor_K = d + dt * v + ((dt**2) / 2.0) * (1.0 - 2.0 * beta) * a
            
            # Montagem do {F_CHAPEU}
            F_hat = F_res_next - C_res @ preditor_C - K_res @ preditor_K
            
            # Solução do sistema linear para aceleração n+1
            a_next = np.linalg.solve(M_hat, F_hat)
            
            # Correção de velocidade e deslocamento
            v_next = v + dt * (1.0 - gamma) * a + dt * gamma * a_next
            d_next = d + dt * v + ((dt**2) / 2.0) * (1.0 - 2.0 * beta) * a + (dt**2) * beta * a_next
            
            # Reconstrução global para armazenamento (54 GDLs)
            u_global = np.zeros(num_gdl_total)
            v_global = np.zeros(num_gdl_total)
            a_global = np.zeros(num_gdl_total)
            
            u_global[gdls_livres] = d_next
            v_global[gdls_livres] = v_next
            a_global[gdls_livres] = a_next
            
            self.hist_u[:, n] = u_global
            
            # Reações de apoio na base
            R_total = (self.K_global @ u_global) + (self.C_global @ v_global) + (self.M_global @ a_global) - F_global_next
            self.hist_R[:, n] = R_total[gdls_fixos]
            
            # Atualização do passado
            d = d_next
            v = v_next
            a = a_next
            
            # Print de acompanhamento a cada 5 segundos virtuais simulados
            if np.isclose(t_next % 5.0, 0.0, atol=dt/2):
                percentual = (t_next / t_max) * 100
                print(f"-> Tempo: {t_next:5.2f} s / {t_max:.1f} s | Concluído: {percentual:6.2f}% | Config: Estável")
                
        print("\n Simulação Concluída com sucesso!")
        
        # --- 8. ROTINA DE SALVAMENTO DE DADOS PARA PÓS-PROCESSAMENTO ---
        os.makedirs(results_dir, exist_ok=True)
        
        # Opção A: Binário Compactado (.npz) -> Perfeito para carregar no script do GIF
        caminho_npz = os.path.join(results_dir, "dados_transiente.npz")
        np.savez_compressed(
            caminho_npz, 
            tempo=self.hist_tempo, 
            deslocamentos=self.hist_u, 
            reacoes=self.hist_R
        )
        print(f"-> [OK] Arquivo binário compactado salvo em: {caminho_npz}")
        
        # Opção B: Arquivo de Texto (.txt) -> Para verificação humana ou importar no Excel
        # Salvamos transposto (.T) para que cada LINHA represente um passo de tempo
        caminho_txt = os.path.join(results_dir, "historico_deslocamentos.txt")
        np.savetxt(
            caminho_txt, 
            self.hist_u.T, 
            fmt='%.6e', 
            delimiter='\t', 
            header="Historico de Deslocamentos (54 GDLs nas colunas, passos de tempo nas linhas)"
        )
        print(f"-> [OK] Arquivo de texto legível salvo em: {caminho_txt}")

    def plotar_movimento_no(self, no_alvo=(1.5, 9.0), pasta_dados=results_dir, append=""):
        """
        Carrega os dados salvos e plota o histórico de deslocamento X e Y de um nó específico.
        """
        # 1. Carrega os dados salvos pelo Newmark
        caminho_npz = os.path.join(pasta_dados, "dados_transiente.npz")
        dados = np.load(caminho_npz)
        tempo = dados['tempo']
        hist_u = dados['deslocamentos']
        
        # 2. Encontra os GDLs do nó alvo
        id_no = self.no_para_id[no_alvo]
        gdl_x = 3 * id_no
        gdl_y = 3 * id_no + 1
        
        # 3. Extrai as linhas de deslocamento (convertendo m para mm)
        u_x = hist_u[gdl_x, :] * 1000
        u_y = hist_u[gdl_y, :] * 1000
        
        # 4. Plotagem
        plt.figure(figsize=(10, 5))
        plt.plot(tempo, u_x, color="royalblue", linewidth=1.5, label="Deslocamento X (Horizontal)")
        plt.plot(tempo, u_y, color="darkorange", linewidth=1.5, label="Deslocamento Y (Vertical)")
        
        plt.title(f"Histórico de Movimento do Nó {no_alvo}", fontsize=12, fontweight="bold")
        plt.xlabel("Tempo (s)")
        plt.ylabel("Deslocamento (mm)")
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend(loc="upper right")
        salvar_grafico(plt, pasta_dados, f"movimento_no_{no_alvo[0]}_{no_alvo[1]}{append}.png")
    
    def plotar_top3_von_mises(self, pasta_dados="resultados_simulacao", skip_steps=5, append=""):
        """
        Calcula a história de tensões de Von Mises para todos os elementos,
        identifica os 3 elementos com maiores picos históricos e plota suas evoluções.
        """
        caminho_npz = os.path.join(pasta_dados, "dados_transiente.npz")
        dados = np.load(caminho_npz)
        tempo = dados['tempo']
        hist_u = dados['deslocamentos']
        
        num_passos = len(tempo)
        num_elementos = len(self.elementos)
        
        # Matriz para guardar o histórico de Von Mises de cada elemento
        # Guardaremos apenas os passos amostrados para economizar memória RAM
        passos_amostrados = np.arange(0, num_passos, skip_steps)
        tempo_amostrado = tempo[passos_amostrados]
        historico_vm = np.zeros((num_elementos, len(passos_amostrados)))
        
        print("-> Calculando tensões de Von Mises ao longo do tempo...")
        
        # Loop temporal amostrado
        for idx_t, passo in enumerate(passos_amostrados):
            u_frame = hist_u[:, passo]
            
            # Loop por todos os elementos da estrutura
            for idx_el, elem in enumerate(self.elementos):
                id1 = self.no_para_id[elem.p1]
                id2 = self.no_para_id[elem.p2]
                
                c = np.cos(elem.theta)
                s = np.sin(elem.theta)
                
                if isinstance(elem, Portico):
                    # Extrai os deslocamentos globais do elemento (6 GDLs)
                    ux1, uy1, theta1 = u_frame[3*id1 : 3*id1+3]
                    ux2, uy2, theta2 = u_frame[3*id2 : 3*id2+3]
                    
                    # Rotaciona para o sistema local do elemento
                    u_local = np.array([
                        ux1*c + uy1*s,  -ux1*s + uy1*c,  theta1,
                        ux2*c + uy2*s,  -ux2*s + uy2*c,  theta2
                    ])
                    
                    K_local = elem.k_matriz
                    # Esforços locais: f_local = K_local @ u_local
                    f_local = K_local @ u_local
                    N1, M1 = f_local[0], f_local[2]
                    N2, M2 = f_local[3], f_local[5]
                    
                    # Tensão nas fibras externas de ambas as extremidades
                    sigma1 = (abs(N1) / elem.A) + (abs(M1) * elem.r_out / elem.I)
                    sigma2 = (abs(N2) / elem.A) + (abs(M2) * elem.r_out / elem.I)
                    
                    # Von Mises máximo do elemento neste frame
                    historico_vm[idx_el, idx_t] = max(sigma1, sigma2)
                    
                elif isinstance(elem, Trelica):
                    ux1, uy1 = u_frame[3*id1], u_frame[3*id1+1]
                    ux2, uy2 = u_frame[3*id2], u_frame[3*id2+1]
                    
                    # Deformação axial delta_L
                    dl = (ux2 - ux1) * c + (uy2 - uy1) * s
                    sigma_axial = (elem.E * dl) / elem.L
                    historico_vm[idx_el, idx_t] = abs(sigma_axial)

        # 5. Identifica os 3 elementos com os maiores picos históricos
        picos_historicos = np.max(historico_vm, axis=1)
        # Pega os índices dos 3 maiores valores ordenados
        indices_top3 = np.argsort(picos_historicos)[-3:][::-1]
        
        print("\n--- ELEMENTOS MAIS SOLICITADOS (TOP 3 VON MISES) ---")
        for ranking, idx in enumerate(indices_top3):
            tipo = "Pórtico" if isinstance(self.elementos[idx], Portico) else "Treliça"
            print(f"{ranking+1}º Lugar -> {tipo} ID {self.elementos[idx].id} | Pico: {picos_historicos[idx]/1e6:.2f} MPa")
            
        # 6. Plotagem do gráfico comparativo do Top 3
        plt.figure(figsize=(11, 5.5))
        cores = ["crimson", "darkviolet", "forestgreen"]
        
        for i, idx in enumerate(indices_top3):
            elem = self.elementos[idx]
            tipo = "Pórtico" if isinstance(elem, Portico) else "Treliça"
            # Converte Pa para MPa no gráfico
            plt.plot(tempo_amostrado, historico_vm[idx, :] / 1e6, 
                    color=cores[i], linewidth=1.8, label=f"{tipo} ID {elem.id}")
            
        plt.title("Evolução Temporal da Tensão de Von Mises - Top 3 Elementos Críticos", fontsize=12, fontweight="bold")
        plt.xlabel("Tempo (s)")
        plt.ylabel("Tensão de Von Mises (MPa)")
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend(loc="upper right")
        salvar_grafico(plt, pasta_dados, "top3_von_mises_historico.png")
      

def main():
    estrutura = Estrutura()
    estrutura.plot_estrutura()
    print(estrutura.seis_frequencias_hz)
    estrutura.plotar_modos_vibracao(fator_escala=3)
    fps_alvo = 5  # 5 frames por segundo de vídeo
    dt_video = 1.0 / fps_alvo  # Cada frame do GIF deve representar 0.2s reais
    dts = [0.1, 0.05, 0.01, 0.005]
    for dt in dts:
        skip_dinamico = max(1, int(np.round(dt_video / dt)))
        estrutura.simular_newmark(dt=dt, t_max=5*60.0)
        gerar_gif_animado(estrutura, pasta_dados=results_dir, append=f"_dt{dt:.3f}", fator_escala=40, skip_frames=skip_dinamico)
        estrutura.plotar_movimento_no(no_alvo=(6.0, 7.5), pasta_dados=results_dir, append=f"_dt{dt:.3f}")
        estrutura.plotar_top3_von_mises(pasta_dados=results_dir, skip_steps=1, append=f"_dt{dt:.3f}")
if __name__ == "__main__":    
    main()