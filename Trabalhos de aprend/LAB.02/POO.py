class Personagem:
    def __init__(self, nome, classe, vida):
        self.nome = nome
        self.classe = classe
        self.vida = vida

    def mostrar_status(self):
        print(f'Nome: {self.nome} | Classe: {self.classe} | Vida: {self.vida}')

    def sofrer_dano(self, valor):
        self.vida -= valor
        if self.vida < 0:
            self.vida = 0
        print(f'{self.nome} sofreu {valor} de dano!')

    def curar(self, valor):
        self.vida += valor
        if self.vida > 100:
            self.vida = 100
        print(f'{self.nome} foi curada em {valor} pontos!')

    def get_vida(self):
        return self.__vida

    def set_vida(self, valor):
        print(f'Definir vida para {valor}...')
        if valor > 100:
            self.__vida = 100
            print("Valor acima do permitido! A vida será ajustada para 100.")
        elif valor < 0:
            self.__vida = 0
            print("Valor abaixo do permitido! A vida será ajustada para 0.")
        else:
            self.__vida = valor


personagem1 = Personagem("Aramis", "Guerreiro", 100)
personagem2 = Personagem("Nyla", "Mago", 80)

personagem1.mostrar_status()
personagem2.mostrar_status()

personagem2.sofrer_dano(30)
personagem2.mostrar_status()

personagem2.curar(20)
personagem2.mostrar_status()

personagem2.curar(50)
personagem2.mostrar_status()

personagem3 = Personagem("Barba Negra", "Pirata", 50)
personagem3.set_vida(135)
personagem3.mostrar_status()

personagem4 = Personagem("Viuva Negra", "Arqueira", 50)
personagem4.set_vida(-20)
personagem4.mostrar_status()


class Mago(Personagem):
    def __init__(self, nome, classe, vida, magia):
        super().__init__(nome, classe, vida)
        self.magia = magia

    def mostrar_status(self):
        print(f'Nome: {self.nome} | Classe: {self.classe} | Vida: {self.vida} | Magia: {self.magia}')


mago1 = Mago("Nyla", "Mago", 100, 50)
mago1.mostrar_status()


class Arqueiro(Personagem):
    def __init__(self, nome, classe, vida, flechas):
        super().__init__(nome, classe, vida)
        self.flechas = flechas

    def mostrar_status(self):
        print(f'Nome: {self.nome} | Classe: {self.classe} | Vida: {self.vida} | Flechas: {self.flechas}')


arqueiro1 = Arqueiro("Robin Hood", "Arqueiro", 100, 50)
arqueiro1.mostrar_status()

lista_personagens = [mago1, arqueiro1]
for personagem in lista_personagens:
    personagem.mostrar_status()
