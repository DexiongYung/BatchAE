import string

import torch

ENCODER_INPUT = {'<PAD>': 0, '<EOS>': 1, '<SOS>': 2, '0': 3, '1': 4, '2': 5, '3': 6, '4': 7, '5': 8, '6': 9, '7': 10,
                 '8': 11, '9': 12, 'a': 13, 'b': 14, 'c': 15, 'd': 16, 'e': 17, 'f': 18, 'g': 19, 'h': 20, 'i': 21,
                 'j': 22, 'k': 23, 'l': 24, 'm': 25, 'n': 26, 'o': 27, 'p': 28, 'q': 29, 'r': 30, 's': 31, 't': 32,
                 'u': 33, 'v': 34, 'w': 35, 'x': 36, 'y': 37, 'z': 38, 'A': 39, 'B': 40, 'C': 41, 'D': 42, 'E': 43,
                 'F': 44, 'G': 45, 'H': 46, 'I': 47, 'J': 48, 'K': 49, 'L': 50, 'M': 51, 'N': 52, 'O': 53, 'P': 54,
                 'Q': 55, 'R': 56, 'S': 57, 'T': 58, 'U': 59, 'V': 60, 'W': 61, 'X': 62, 'Y': 63, 'Z': 64, '!': 65,
                 '"': 66, '#': 67, '$': 68, '%': 69, '&': 70, "'": 71, '(': 72, ')': 73, '*': 74, '+': 75, ',': 76,
                 '-': 77, '.': 78, '/': 79, ':': 80, ';': 81, '<': 82, '=': 83, '>': 84, '?': 85, '@': 86, '[': 87,
                 '\\': 88, ']': 89, '^': 90, '_': 91, '`': 92, '{': 93, '|': 94, '}': 95, '~': 96, ' ': 97, '\t': 98,
                 '\n': 99, '\r': 100, '\x0b': 101, '\x0c': 102}
DECODER_INPUT = {'<PAD>': 0, '<EOS>': 1, '<SOS>': 2, 'a': 3, 'b': 4, 'c': 5, 'd': 6, 'e': 7, 'f': 8, 'g': 9, 'h': 10,
                 'i': 11, 'j': 12, 'k': 13, 'l': 14, 'm': 15, 'n': 16, 'o': 17, 'p': 18, 'q': 19, 'r': 20, 's': 21,
                 't': 22, 'u': 23, 'v': 24, 'w': 25, 'x': 26, 'y': 27, 'z': 28, 'A': 29, 'B': 30, 'C': 31, 'D': 32,
                 'E': 33, 'F': 34, 'G': 35, 'H': 36, 'I': 37, 'J': 38, 'K': 39, 'L': 40, 'M': 41, 'N': 42, 'O': 43,
                 'P': 44, 'Q': 45, 'R': 46, 'S': 47, 'T': 48, 'U': 49, 'V': 50, 'W': 51, 'X': 52, 'Y': 53, 'Z': 54,
                 "'": 55, '-': 56}
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
