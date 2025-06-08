from enum import IntEnum, auto
from models_utils import *
import os
from time import sleep

class MenuOption(IntEnum):
    TRAIN_MODELS = auto(),
    INFER = auto(),
    EXIT = auto(),

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def char_n_times(char, n):
    print(char * n)

def handle_menu():
    while True:
        clear_screen() # Limpa a tela antes da impressão do menu

        char_n_times("-", 50)
        print("Pediatric Appendicitis Model")
        for option in MenuOption:
            print(f"[{option.value}] - {option.name.replace("_", " ").capitalize()}") # Gera cada opção do menu dinamicamente
        char_n_times("-", 50)

        # Gerencia o fluxo conforme a opção escolhida
        opt = int(input("Enter your choice number: "))
        match opt:
            case MenuOption.TRAIN_MODELS: pass
            case MenuOption.INFER: pass
            case MenuOption.EXIT: exit(0)
            case _:
                print("\nInvalid option number. Try again")
        input("\nPress any key to continue...") # Aguarda por qualquer tecla para prosseguir
